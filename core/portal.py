import numpy as np
import pyrr
import math
from OpenGL.GL import *
from core.mesh import Mesh
from core.shader import Shader
from math3d.portal_math import (
    calculate_virtual_view as _calculate_virtual_view,
    calculate_oblique_projection as _calculate_oblique_projection,
    portal_normal_world as _portal_normal_world,
)

def create_portal_mesh() -> Mesh:
    """
        A simple quad on XY, centered at the origin.
        Represents the portal surface.
        Size: 1X2 units.
    """
    vertices = np.array([
        # pos               normal (points to +Z - Portal Front)
        -0.5, -1.0, 0.0,   0.0, 0.0, 1.0,
         0.5, -1.0, 0.0,   0.0, 0.0, 1.0,
         0.5,  1.0, 0.0,   0.0, 0.0, 1.0,
         0.5,  1.0, 0.0,   0.0, 0.0, 1.0,
        -0.5,  1.0, 0.0,   0.0, 0.0, 1.0,
        -0.5, -1.0, 0.0,   0.0, 0.0, 1.0,
    ], dtype=np.float32)
    return Mesh(vertices, True)

class Portal:
    """
        Represents a portal in the scene.
        Each portal has a pair. Together they define a "doorway" between two locations.

        Responsabylities:
        - Keep Position and orientation
        - Calculate virtual camera rendering
        - Draw the portal surface on stencil buffer    
    """
    mesh_quad: Mesh = None  # Shared mesh for all portals, a simple quad

    def __init__(self, position, rotation, color):
        """
            position: (x, y, z) world coordinates of the portal center
            rotation: degrees - Where the portal is facing
            color: RGB color for debugging (0.0 to 1.0)
        """
        # Later create a scale to pass as self.scale on S in get_model_matrix
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation
        self.color = np.array(color, dtype=np.float32)
        self.destiny: 'Portal' = None

        if Portal.mesh_quad is None:
            Portal.mesh_quad = create_portal_mesh()

    def get_model_matrix(self) -> np.ndarray:
        """
        Builds the Model Matrix for the portal.
        Combines translation and Y rotation.
        Scale: width=1, height=2 (reasonable size for a character to pass through).
        """
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)

        R = pyrr.matrix44.create_from_y_rotation(
            math.radians(self.rotation), dtype=np.float32)

        S = pyrr.matrix44.create_from_scale(
            np.array([12, 5.0, 5.0], dtype=np.float32), dtype=np.float32)
        # Create below a print that show the position, rotation and scale of the portal for debugging
        # print(f"Portal Model Matrix:\nPosition: {self.position}, Rotation: {self.rotation}, Scale: [1.5, 2.0, 1.0]\n{S @ R @ T}")

        # Keep the same transform convention used by Entity:
        # scale -> rotation -> translation.
        return (S @ R @ T).astype(np.float32)

    def get_portal_transform(self) -> np.ndarray:
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)
        R = pyrr.matrix44.create_from_y_rotation(math.radians(self.rotation), dtype=np.float32)
        return (R @ T).astype(np.float32)  #rotação + translação

    def is_camera_in_front(self, camera_pos) -> bool:
        """
        True se a câmera está no semi-espaço para onde a normal frontal do
        portal aponta. Evita que o portal seja aberto quando visto por trás.
        """
        normal = _portal_normal_world(self.rotation)
        portal_to_cam = np.asarray(camera_pos, dtype=np.float32) - self.position
        return float(np.dot(normal, portal_to_cam)) > 0.0

    def calculate_virtual_view(self, realView: np.ndarray) -> np.ndarray:
        """
        View matrix da câmera virtual no lado destino do portal.
        Delegação para math3d.portal_math.
        """
        return _calculate_virtual_view(
            realView,
            self.get_portal_transform(),
            self.destiny.get_portal_transform(),
        )

    def calculate_oblique_projection(self,
                                     virtual_view: np.ndarray,
                                     projection: np.ndarray) -> np.ndarray:
        """
        Projeção com near plane oblíquo coincidindo com o plano do portal
        destino. Evita que geometria entre a câmera virtual e o portal
        destino "vaze" para dentro da máscara do stencil.
        """
        return _calculate_oblique_projection(
            projection,
            virtual_view,
            self.destiny.position,
            self.destiny.rotation,
        )
    
    # Rendering

    def draw_stencil(self, shader: Shader):
        """
            Draw the opening of the portal, Only on the stencil buffer.
            Don't write Color or Depth
            Stencil select the pixels that belong to the portal surface.
        """

        # Desables color and depth writing.
        # Thats important because we only want to mark the portal area on the stencil buffer.
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)

        # Configures stencil to write 1 where quads are drawn.
        glStencilFunc(GL_ALWAYS, 1, 0xFF)  # Always pass stencil test, reference value = 1
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)  # Replace stencil value with ref on depth pass

        # Draw the portal quad
        shader.set_matrix4("model", self.get_model_matrix())
        shader.set_vec3("color", self.color)
        Portal.mesh_quad.draw()

        # Restore Color and Depth writing for the rest of the scene.
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def draw_portal_border(self, shader: Shader):
        """
            Draw a border around the portal for debugging.
            This is drawn on the color buffer, not stencil.
        """
        shader.set_matrix4("model", self.get_model_matrix())
        shader.set_vec3("objectColor", self.color)
        Portal.mesh_quad.draw()
