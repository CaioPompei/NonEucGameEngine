from pathlib import Path

import glfw
import numpy as np
import pyrr

from engine.camera import Camera
from engine.light import PointLight
from engine.shader import Shader
from engine.text_overlay import TextOverlay
from engine.window import Window
from game.level import create_room
from game.player import Player
from game.portal import Portal
from game.portal_renderer import PortalRenderer

SHADERS_DIR = Path(__file__).resolve().parent / "shaders"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 720


def main():
    window = Window(WINDOW_WIDTH, WINDOW_HEIGHT, "What is There?")
    camera = Camera(position=(0.0, 0.5, 3.0))
    player = Player(camera)

    window.get_mouse()
    window.set_callback_mouse(lambda win, x, y: camera.process_mouse_movement(x, y))

    phong_shader = Shader.from_files(SHADERS_DIR / "phong.vert",
                                     SHADERS_DIR / "phong.frag")
    simple_shader = Shader.from_files(SHADERS_DIR / "simple.vert",
                                      SHADERS_DIR / "simple.frag")
    depth_shader = Shader.from_files(SHADERS_DIR / "depth.vert",
                                     SHADERS_DIR / "depth.frag")

    scene = create_room()

    # Lights: instantiate, add to scene, bake once. Cena + luzes estáticas
    # → o custo das shadow maps cai pra zero por frame depois do bake.
    scene.add_light(PointLight(position=(0.0, 4.0, 0.0),
                               color=(1.0, 1.0, 1.0),
                               intensity=1.0,
                               range=35.0))
    scene.bake_shadows(depth_shader)

    debug_overlay = TextOverlay("debug mode", WINDOW_WIDTH, WINDOW_HEIGHT,
                                font_size=22,
                                color=(255, 230, 80, 255))

    # Portal A on the north wall, Portal B on the south wall — paired.
    portal_a = Portal(position=(0.0, 0.0, -9.7), rotation=0.0, color=(1.0, 0.5, 0.5))
    portal_b = Portal(position=(0.0, 0.0, 9.7), rotation=180.0, color=(0.2, 0.5, 1.0))
    portal_a.link_to(portal_b)
    portals = [portal_a, portal_b]

    portal_renderer = PortalRenderer(
        portals=portals,
        scene=scene,
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

    while not window.window_close():
        real_time = glfw.get_time()
        delta_time = real_time - previous_time
        previous_time = real_time

        window.process_events()
        player.process_input(window.get_handle(), delta_time, portals)
        view = camera.get_view_matrix()

        window.clear()

        phong_shader.use()
        phong_shader.set_matrix4("view", view)
        phong_shader.set_matrix4("projection", projection)
        phong_shader.set_vec3("cameraPos", camera.position)
        scene.bind_lights(phong_shader, first_texture_unit=1)
        scene.draw(phong_shader)

        portal_renderer.render(view, projection, camera.position)

        if player.mode == Player.MODE_FREECAM:
            debug_overlay.draw()

        window.show()

    window.close()


if __name__ == "__main__":
    main()
