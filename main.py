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
from core.text_overlay import TextOverlay
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

PORTAL_RECURSION_DEPTH = 3  # how many nested portal levels to render


def _draw_portal_quad(portal, shader, view, projection):
    shader.set_matrix4("view", view)
    shader.set_matrix4("projection", projection)
    shader.set_matrix4("model", portal.get_model_matrix())
    Portal.mesh_quad.draw()


def render_portal(portals, shader_phong, simple_shader, real_view,
                  projection, light_pos, light_color, scene):
    glEnable(GL_STENCIL_TEST)
    glClear(GL_STENCIL_BUFFER_BIT)
    _render_portal_recursive(
        portals, shader_phong, simple_shader,
        real_view, projection,
        light_pos, light_color, scene,
        depth=0,
    )
    glDisable(GL_STENCIL_TEST)


def _render_portal_recursive(portals, shader_phong, simple_shader,
                             view, projection,
                             light_pos, light_color, scene, depth):
    # Camera position for this view — needed for the "facing" check.
    inv_view = np.linalg.inv(view)
    cam_pos = inv_view[3, :3]

    for portal in portals:
        if portal.destiny is None:
            continue
        if not portal.is_camera_in_front(cam_pos):
            continue

        # 1. Mark stencil: increment from `depth` to `depth+1` where this
        #    portal is visible in the current view (depth test active).
        glStencilFunc(GL_EQUAL, depth, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        simple_shader.use()
        _draw_portal_quad(portal, simple_shader, view, projection)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

        # 2. Virtual view/projection through this portal.
        new_view = portal.calculate_virtual_view(view)
        new_proj = portal.calculate_oblique_projection(new_view, projection)

        # 3. Reset depth to far inside the new stencil region.
        #    glClear ignores stencil test, so we draw the portal quad with
        #    glDepthRange(1,1) + GL_ALWAYS to force depth=1.0 there.
        glStencilFunc(GL_EQUAL, depth + 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_ALWAYS)
        glDepthRange(1.0, 1.0)
        _draw_portal_quad(portal, simple_shader, view, projection)
        glDepthRange(0.0, 1.0)
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        # 4. Draw the scene at this nested level, constrained to stencil == depth+1.
        new_cam_pos = np.linalg.inv(new_view)[3, :3]
        glStencilFunc(GL_EQUAL, depth + 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        shader_phong.use()
        shader_phong.set_matrix4("view", new_view)
        shader_phong.set_matrix4("projection", new_proj)
        shader_phong.set_vec3("lightPos", light_pos)
        shader_phong.set_vec3("lightColor", light_color)
        shader_phong.set_vec3("cameraPos", new_cam_pos)
        scene.draw(shader_phong)

        # 5. Recurse into the deeper level so portals seen *inside* the virtual
        #    scene also open up.
        if depth + 1 < PORTAL_RECURSION_DEPTH:
            _render_portal_recursive(
                portals, shader_phong, simple_shader,
                new_view, new_proj,
                light_pos, light_color, scene,
                depth + 1,
            )

        # 6. Decrement stencil back to `depth` so siblings at this level
        #    aren't masked out by the work we just did.
        glStencilFunc(GL_EQUAL, depth + 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_DECR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        simple_shader.use()
        _draw_portal_quad(portal, simple_shader, view, projection)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

        # 7. Restore depth at the portal quad position in this level's stencil,
        #    so the next sibling portal sees a consistent depth buffer.
        glStencilFunc(GL_EQUAL, depth, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_ALWAYS)
        _draw_portal_quad(portal, simple_shader, view, projection)
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

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

    debug_overlay = TextOverlay("debug mode", 1200, 720,
                                font_size=22,
                                color=(255, 230, 80, 255))

    # Create Portals
    # Portal A - North Wall
    # Portal B - South Wall
    portal_a = Portal(position=(0.0, 0.0, -9.7), rotation=0.0, color=(1.0, 0.5, 0.5))
    portal_b = Portal(position=(0.0, 0.0, 9.7), rotation=180.0, color=(0.2, 0.5, 1.0))

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

        render_portal(portals, phong_shader, simple_shader, view, projection, light_pos, light_color, scene)

        if player.mode == Player.MODE_FREECAM:
            debug_overlay.draw()

        window.show()

    

    window.close()


if __name__ == "__main__":
    main()