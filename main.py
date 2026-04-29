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
layout (location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_pos;
out vec3 frag_normal;

void main() {
    // Position in world space.
    // We need it to calculate the lighting.
    frag_pos = vec3(model * vec4(position, 1.0));

    // Normal needs to be transformed by the model matrix.
    // Without translation, normal is direction only, not position
    frag_normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 frag_pos;
in vec3 frag_normal;

out vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform vec3 cameraPos;

void main() {
    // Ambient
    float ambientStrength = 0.15; 
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 normal = normalize(frag_normal);
    vec3 lightDir = normalize(lightPos - frag_pos);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(cameraPos - frag_pos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    fragColor = vec4(result, 1.0);
}
"""

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
    #model = pyrr.matrix44.create_identity(np.float32)

    light_pos = np.array([0.0, 4, 0.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32) # RGB white light

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
        # shader.set_matrix4("model",      model)
        shader.set_matrix4("view",       view)
        shader.set_matrix4("projection", projection)
        shader.set_vec3("lightPos", light_pos)
        shader.set_vec3("lightColor", light_color)
        shader.set_vec3("cameraPos", camera.position)

        scene.draw(shader)

        window.show()

    window.close()


if __name__ == "__main__":
    main()