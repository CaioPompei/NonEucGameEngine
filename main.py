import glfw
import numpy as np
import pyrr
from core.camera import Camera
from core.window import Window
from core.shader import Shader
from core.mesh import Mesh
from game.level import create_room

# ── Shaders ───────────────────────────────────────────────────────────────────

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 fragColor;
uniform vec3 uColor;
void main() {
    fragColor = vec4(uColor, 1.0);
}
"""

# ── Geometria ─────────────────────────────────────────────────────────────────

VERTICES_CUBO = np.array([
    -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
     0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5,
     0.5,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,
    -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
     0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5,  0.5,
     0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5, -0.5, -0.5,
    -0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,
     0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5,
], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    window = Window(1200, 720, "What is There?")
    camera = Camera(position=(0.0, 0.5, 3.0))

    window.get_mouse()

    window.set_callback_mouse(lambda win, x, y: camera.process_mouse_movement(x, y))

    shader = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
    # cubo   = Mesh(VERTICES_CUBO)
    scene = create_room()

    projection = pyrr.matrix44.create_perspective_projection(
        90.0,
        1200 / 720,
        0.1,
        100.0,
        np.float32
    )
    model = pyrr.matrix44.create_identity(np.float32)

    previous_time = glfw.get_time()

    while not window.window_close():

        real_time = glfw.get_time() 
        delta_time = real_time - previous_time
        previous_time = real_time
    
        window.process_events()
    
        camera.process_input(window.get_handle(), delta_time)
        view = camera.get_view_matrix()

        window.clear()

        shader.use()
        shader.set_matrix4("model",      model)
        shader.set_matrix4("view",       view)
        shader.set_matrix4("projection", projection)

        scene.draw(shader)

        window.show()

    window.close()


if __name__ == "__main__":
    main()