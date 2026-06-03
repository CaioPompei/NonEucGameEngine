from pathlib import Path

import glfw
import numpy as np
import pyrr

from engine.camera import Camera
from engine.shader import Shader
from engine.text_overlay import TextOverlay
from engine.window import Window
from game.level_loader import LevelLoader
from game.player import Player
from game.portal_renderer import PortalRenderer

PROJECT_ROOT = Path(__file__).resolve().parent
SHADERS_DIR = PROJECT_ROOT / "shaders"
LEVELS_DIR = PROJECT_ROOT / "levels"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 720


def main():
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, "What is There?")
    camera = Camera(position=(0.0, 0.3, 3.0))
    player = Player(camera)

    window.get_mouse()
    window.set_callback_mouse(lambda win, x, y: camera.process_mouse_movement(x, y))

    phong_shader = Shader.from_files(SHADERS_DIR / "phong.vert",
                                     SHADERS_DIR / "phong.frag")
    simple_shader = Shader.from_files(SHADERS_DIR / "simple.vert",
                                      SHADERS_DIR / "simple.frag")
    depth_shader = Shader.from_files(SHADERS_DIR / "depth.vert",
                                     SHADERS_DIR / "depth.frag")

    loader = LevelLoader()
    level = loader.load(LEVELS_DIR / "level_01.json")

    camera.position[:] = level.player_start
    level.scene.bake_shadows(depth_shader)

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

    portal_renderer = PortalRenderer(
        portals=level.portals,
        scene=level.scene,
        scene_shader=phong_shader,
        stencil_shader=simple_shader,
        max_depth=3,
    )

    # near is intentionally tiny: the player can put the camera within a few
    # millimeters of a portal quad before traversing, and any geometry closer
    # than `near` is clipped — which would leave visible holes on the portal
    # edges right before the teleport ("cut" effect).
    projection = pyrr.matrix44.create_perspective_projection(
        90.0, WINDOW_WIDTH / WINDOW_HEIGHT, 0.01, 100.0, np.float32
    )

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

    while not window.window_close():
        real_time = glfw.get_time()
        delta_time = real_time - previous_time
        previous_time = real_time

        fps_count += 1
        fps_time_acc += delta_time
        if fps_time_acc >= FPS_WINDOW:
            fps_display = fps_count / fps_time_acc
            fps_count = 0
            fps_time_acc = 0.0

        window.process_events()
        player.process_input(window.get_handle(), delta_time, level.portals)

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

        view = camera.get_view_matrix()

        window.clear()

        phong_shader.use()
        phong_shader.set_matrix4("view", view)
        phong_shader.set_matrix4("projection", projection)
        phong_shader.set_vec3("cameraPos", camera.position)
        level.scene.bind_lights(phong_shader, first_texture_unit=1)
        level.scene.draw(phong_shader)

        portal_renderer.render(view, projection, camera.position)

        # "debug mode" label — only while FREECAM is active.
        if player.mode == Player.MODE_FREECAM:
            debug_label_overlay.draw()

        # Stats panel — independent toggle (`´`).
        if show_stats:
            if real_time >= stats_next_update:
                x, y, z = camera.position
                mode_label = ("FREECAM" if player.mode == Player.MODE_FREECAM
                              else "WALK")
                stats_overlay.update_text(
                    f"fps    {fps_display:6.1f}\n"
                    f"mode   {mode_label}\n"
                    f"x      {x:7.2f}\n"
                    f"y      {y:7.2f}\n"
                    f"z      {z:7.2f}\n"
                    f"yaw    {camera.yaw:7.1f}\n"
                    f"pitch  {camera.pitch:7.1f}"
                )
                stats_next_update = real_time + STATS_UPDATE_INTERVAL
            stats_overlay.draw()

        window.show()

    window.close()


if __name__ == "__main__":
    main()
