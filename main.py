import glfw
import numpy as np
import pyrr
from OpenGL.GL import *
from core.camera import Camera
from core.player import Player
from core.portal import Portal
from core.window import Window
from core.shader import Shader
from core.mesh import Mesh
from game.level import create_room

# ── Shaders ───────────────────────────────────────────────────────────────────

VERTEX_SIMPLE_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

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

FRAGMENT_SIMPLE_SHADER = """
#version 330 core
out vec4 fragColor;
uniform vec3 objectColor;
void main() {
    fragColor = vec4(objectColor, 1.0); // Red color
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

def render_portal(portals, shader_phong, simple_shader, real_view,
                  projection, light_pos, light_color, camera_pos, scene):
    for i, portal in enumerate(portals):
        if portal.destiny is None:
            continue

        # glClear(GL_DEPTH_BUFFER_BIT) ignores stencil test on desktop GL,
        # so the previous portal iteration wiped the whole depth buffer.
        # Repopulate it with the real scene before marking this portal's
        # stencil — otherwise the quad would pass the depth test everywhere.
        if i > 0:
            glDisable(GL_STENCIL_TEST)
            glClear(GL_DEPTH_BUFFER_BIT)
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
            glDepthMask(GL_TRUE)
            shader_phong.use()
            shader_phong.set_matrix4("view",       real_view)
            shader_phong.set_matrix4("projection", projection)
            shader_phong.set_vec3("lightPos",   light_pos)
            shader_phong.set_vec3("lightColor", light_color)
            shader_phong.set_vec3("cameraPos",  camera_pos)
            scene.draw(shader_phong)
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        glEnable(GL_STENCIL_TEST)
        glClear(GL_STENCIL_BUFFER_BIT)

        simple_shader.use()
        simple_shader.set_matrix4("view",       real_view)
        simple_shader.set_matrix4("projection", projection)

        portal.draw_stencil(simple_shader)

        glStencilFunc(GL_EQUAL, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)

        glDepthMask(GL_TRUE)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glClear(GL_DEPTH_BUFFER_BIT)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        virtual_view = portal.calculate_virtual_view(real_view)
        oblique_projection = portal.calculate_oblique_projection(
            virtual_view, projection)

        shader_phong.use()
        shader_phong.set_matrix4("view",       virtual_view)
        shader_phong.set_matrix4("projection", oblique_projection)
        shader_phong.set_vec3("lightPos",   light_pos)
        shader_phong.set_vec3("lightColor", light_color)
        shader_phong.set_vec3("cameraPos",  camera_pos)
        scene.draw(shader_phong)

    glDisable(GL_STENCIL_TEST)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    window = Window(1200, 720, "What is There?")
    camera = Camera(position=(0.0, 0.5, 3.0))
    player = Player(camera)

    window.get_mouse()
    window.set_callback_mouse(lambda win, x, y: camera.process_mouse_movement(x, y))

    phong_shader = Shader(VERTEX_SHADER, FRAGMENT_SHADER)
    simple_shader = Shader(VERTEX_SIMPLE_SHADER, FRAGMENT_SIMPLE_SHADER)
    # cubo   = Mesh(VERTICES_CUBO)
    scene = create_room()

    # Create Portals
    # Portal A - North Wall
    # Portal B - South Wall
    portal_a = Portal(position=(0.0, 0.0, -9.7), rotation=0.0, color=(1.0, 0.5, 0.5))
    portal_b = Portal(position=(0.0, 0.0, 9.7), rotation=0.0, color=(0.2, 0.5, 1.0))

    # Connects the portals
    portal_a.destiny = portal_b
    portal_b.destiny = portal_a

    portals = [portal_a, portal_b]

    print(f"Portal A: {portal_a.position}, destino: {portal_a.destiny is not None}")
    print(f"Portal B: {portal_b.position}, destino: {portal_b.destiny is not None}")

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
    
        player.process_input(window.get_handle(), delta_time)
        view = camera.get_view_matrix()

        window.clear()

        phong_shader.use()
        # shader.set_matrix4("model",      model)
        phong_shader.set_matrix4("view",       view)
        phong_shader.set_matrix4("projection", projection)
        phong_shader.set_vec3("lightPos", light_pos)
        phong_shader.set_vec3("lightColor", light_color)
        phong_shader.set_vec3("cameraPos", camera.position)
        scene.draw(phong_shader)

        render_portal(portals, phong_shader, simple_shader, view, projection, light_pos, light_color, camera.position, scene)

        window.show()

    

    window.close()


if __name__ == "__main__":
    main()