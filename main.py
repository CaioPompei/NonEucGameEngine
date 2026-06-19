from pathlib import Path

import glfw
import numpy as np
import pyrr

from engine.camera import Camera
from engine.menu import Menu
from engine.shader import Shader
from engine.text_overlay import TextOverlay
from engine.window import Window
from game.collision import CollisionWorld
from game.level_loader import LevelLoader
from game.player import Player
from game.portal_renderer import PortalRenderer

PROJECT_ROOT = Path(__file__).resolve().parent
SHADERS_DIR = PROJECT_ROOT / "shaders"
LEVELS_DIR = PROJECT_ROOT / "levels"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 720


def make_projection(width, height):
    # near is intentionally tiny: the player can put the camera within a few
    # millimeters of a portal quad before traversing, and any geometry closer
    # than `near` is clipped — which would leave visible holes on the portal
    # edges right before the teleport ("cut" effect).
    return pyrr.matrix44.create_perspective_projection(
        90.0, width / height, 0.01, 100.0, np.float32
    )


def main():
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, "What is There?")
    camera = Camera(position=(0.0, 0.3, 3.0))
    player = Player(camera)

    def on_mouse(win, x, y):
        # Only steer the camera while actually playing. In the menu the cursor
        # is free, and moving it must not rotate the level's view.
        if state == STATE_PLAYING:
            camera.process_mouse_movement(x, y)

    window.set_callback_mouse(on_mouse)

    phong_shader = Shader.from_files(SHADERS_DIR / "phong.vert",
                                     SHADERS_DIR / "phong.frag")
    simple_shader = Shader.from_files(SHADERS_DIR / "simple.vert",
                                      SHADERS_DIR / "simple.frag")
    depth_shader = Shader.from_files(SHADERS_DIR / "depth.vert",
                                     SHADERS_DIR / "depth.frag")

    loader = LevelLoader()

    def setup_level(path):
        """Load a level and build everything tied to it (collision world,
        baked shadows, portal renderer). Also resets the player onto the new
        spawn so motion state doesn't carry over between levels."""
        lvl = loader.load(path)
        world = CollisionWorld.from_scene(lvl.scene)
        # Merge static geometry before baking: both the shadow bake and every
        # per-frame / per-portal scene draw then cost one call per texture+color
        # group instead of one per entity.
        lvl.scene.build_static_batch()
        lvl.scene.bake_shadows(depth_shader)
        renderer = PortalRenderer(
            portals=lvl.portals,
            scene=lvl.scene,
            scene_shader=phong_shader,
            stencil_shader=simple_shader,
            max_depth=1,
            skybox=lvl.skybox,
        )
        player.reset(lvl.player_start, lvl.player_start_dir)
        return lvl, world, renderer

    # Game runtime, built lazily the first time the player leaves the menu so
    # the (expensive) shadow bake doesn't run until a level is actually played.
    level = collision_world = portal_renderer = None

    # Initial menu. Actions are matched in the main loop below.
    main_menu = Menu(
        "NonEucGameEngine",
        [("Iniciar", "start"),
         ("Sair", "quit")],
        WINDOW_WIDTH, WINDOW_HEIGHT)

    # "debug mode" label — fixed text, shown only while in FREECAM (toggle V).
    debug_label_overlay = TextOverlay(
        "debug mode", WINDOW_WIDTH, WINDOW_HEIGHT,
        font_size=20, color=(255, 230, 80, 255),
        padding=10, margin_px=20, corner="top-left")

    # Stats panel — fps + position + orientation. Toggled with `´`
    # (KEY_LEFT_BRACKET in BR-ABNT2 layouts). Updates dynamically.
    stats_overlay = TextOverlay(
        "", WINDOW_WIDTH, WINDOW_HEIGHT,
        font_size=18, color=(180, 255, 180, 255),
        padding=10, margin_px=20, corner="top-right")

    # Crosshair — small "+" dead center, no background box.
    crosshair_overlay = TextOverlay(
        "+", WINDOW_WIDTH, WINDOW_HEIGHT,
        font_size=18, color=(255, 255, 255, 200),
        padding=2, corner="center", background=(0, 0, 0, 0))

    projection = make_projection(WINDOW_WIDTH, WINDOW_HEIGHT)

    def on_resize(width, height):
        """Adapt everything that depends on the framebuffer size: the camera
        projection (aspect ratio) and every screen-space overlay."""
        nonlocal projection
        projection = make_projection(width, height)
        main_menu.resize(width, height)
        debug_label_overlay.resize(width, height)
        stats_overlay.resize(width, height)
        crosshair_overlay.resize(width, height)

    window.on_resize(on_resize)
    # Sync once to the real framebuffer size (may differ from the requested
    # size on HiDPI displays) before the first frame.
    on_resize(*window.get_size())

    previous_time = glfw.get_time()
    stats_next_update = 0.0
    STATS_UPDATE_INTERVAL = 0.1  # 10 Hz is plenty for the readout

    # FPS counter: accumulate frames over a window so the readout is stable.
    fps_count = 0
    fps_time_acc = 0.0
    fps_display = 0.0
    FPS_WINDOW = 0.25  # seconds

    # Stats panel toggle (edge-triggered on `´` / KEY_LEFT_BRACKET).
    show_stats = False
    stats_key_was_pressed = False

    # ── Game state ────────────────────────────────────────────────────────────
    # MENU shows the initial menu (free cursor); PLAYING runs a level (captured
    # cursor). ESC toggles PLAYING -> MENU; in MENU it quits.
    STATE_MENU = "menu"
    STATE_PLAYING = "playing"
    state = STATE_MENU
    window.set_cursor_captured(False)
    esc_was_pressed = False
    f11_was_pressed = False  # F11 toggles fullscreen, edge-triggered

    while not window.window_close():
        real_time = glfw.get_time()
        delta_time = real_time - previous_time
        previous_time = real_time

        window.process_events()

        # Central ESC handling (edge-triggered): the menu owns "quit", the game
        # uses it to return to the menu.
        esc_now = glfw.get_key(window.get_handle(),
                               glfw.KEY_ESCAPE) == glfw.PRESS
        esc_pressed = esc_now and not esc_was_pressed
        esc_was_pressed = esc_now

        # F11 toggles fullscreen in any state. Toggling re-creates the window's
        # surface, so re-apply the cursor mode and reset the look state to avoid
        # a camera jump, and discard the dt spike from the (brief) stall.
        f11_now = glfw.get_key(window.get_handle(),
                               glfw.KEY_F11) == glfw.PRESS
        if f11_now and not f11_was_pressed:
            window.toggle_fullscreen()
            window.set_cursor_captured(state == STATE_PLAYING)
            camera.first_mouse = True
            previous_time = glfw.get_time()
        f11_was_pressed = f11_now

        if state == STATE_MENU:
            if esc_pressed:
                break
            action = main_menu.update(window.get_handle())
            if action == "quit":
                break
            if action == "start":
                if level is None:
                    level, collision_world, portal_renderer = setup_level(
                        LEVELS_DIR / "level_01.json")
                camera.first_mouse = True  # avoid a look jump on re-capture
                window.set_cursor_captured(True)
                state = STATE_PLAYING
                previous_time = glfw.get_time()  # discard the menu->play dt
                continue

            window.clear((0.0, 0.0, 0.0, 1.0))
            main_menu.draw()
            window.show()
            continue

        # ── PLAYING ────────────────────────────────────────────────────────────
        if esc_pressed:
            window.set_cursor_captured(False)
            main_menu.reset()
            state = STATE_MENU
            continue

        fps_count += 1
        fps_time_acc += delta_time
        if fps_time_acc >= FPS_WINDOW:
            fps_display = fps_count / fps_time_acc
            fps_count = 0
            fps_time_acc = 0.0

        player.process_input(window.get_handle(), delta_time,
                             collision_world, level.portals)

        # Edge-triggered toggle for the stats panel (`´` in BR-ABNT2).
        stats_key_now = glfw.get_key(window.get_handle(),
                                     glfw.KEY_LEFT_BRACKET) == glfw.PRESS
        if stats_key_now and not stats_key_was_pressed:
            show_stats = not show_stats
        stats_key_was_pressed = stats_key_now

        for trigger in level.triggers:
            event = trigger.check(camera.position)
            if event is not None:
                level.puzzle.dispatch(event)

        # Level transition: when the puzzle is solved and points to a next
        # level, swap the whole runtime and skip the rest of this frame.
        if level.puzzle.should_transition():
            level, collision_world, portal_renderer = setup_level(
                LEVELS_DIR / level.puzzle.next_level)
            previous_time = glfw.get_time()  # avoid a huge dt after the bake
            continue

        view = camera.get_view_matrix()

        window.clear()

        phong_shader.use()
        phong_shader.set_matrix4("view", view)
        phong_shader.set_matrix4("projection", projection)
        phong_shader.set_vec3("cameraPos", camera.position)
        level.scene.bind_lights(phong_shader, first_texture_unit=1)
        level.scene.draw(phong_shader)

        portal_renderer.render(view, projection, camera.position)

        # Sky last: it's drawn at the far plane and only fills pixels the scene
        # and portals didn't already cover.
        if level.skybox is not None:
            level.skybox.draw(view, projection)

        # Crosshair — only while actually playing (WALK), not in debug fly-cam.
        if player.mode == Player.MODE_WALK:
            crosshair_overlay.draw()

        # "debug mode" label — only while FREECAM is active.
        if player.mode == Player.MODE_FREECAM:
            debug_label_overlay.draw()

        # Stats panel — independent toggle (`´`).
        if show_stats:
            if real_time >= stats_next_update:
                x, y, z = camera.position
                mode_label = ("FREECAM" if player.mode == Player.MODE_FREECAM
                              else "WALK")
                speed = float(np.linalg.norm(player.velocity))
                ground_label = "yes" if player.on_ground else "no"
                stats_overlay.update_text(
                    f"fps     {fps_display:6.1f}\n"
                    f"mode    {mode_label}\n"
                    f"x       {x:7.2f}\n"
                    f"y       {y:7.2f}\n"
                    f"z       {z:7.2f}\n"
                    f"yaw     {camera.yaw:7.1f}\n"
                    f"pitch   {camera.pitch:7.1f}\n"
                    f"speed   {speed:7.2f}\n"
                    f"ground  {ground_label:>7}"
                )
                stats_next_update = real_time + STATS_UPDATE_INTERVAL
            stats_overlay.draw()

        window.show()

    window.close()


if __name__ == "__main__":
    main()
