import numpy as np
import pyrr
import math
from OpenGL.GL import *
from core.mesh import Mesh
from core.shader import Shader

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
            np.array([1.5, 2.0, 1.0], dtype=np.float32), dtype=np.float32)
        # Create below a print that show the position, rotation and scale of the portal for debugging
        # print(f"Portal Model Matrix:\nPosition: {self.position}, Rotation: {self.rotation}, Scale: [1.5, 2.0, 1.0]\n{S @ R @ T}")

        # Keep the same transform convention used by Entity:
        # scale -> rotation -> translation.
        return (S @ R @ T).astype(np.float32)

    def get_portal_transform(self) -> np.ndarray:
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)
        R = pyrr.matrix44.create_from_y_rotation(math.radians(self.rotation), dtype=np.float32)
        return (R @ T).astype(np.float32)  #rotação + translação

    def calculate_virtual_view(self, realView: np.ndarray) -> np.ndarray:
        """ 
        Calculate the View Matrix for the virtual camera on the destination side.

        The formula is:
            realView = real_view_do_destino
                         = view × inv(model_origem) × rotation_180 × destinyModel

        Rotation Y is necessary because the portals face each other:
        entering through the front of A means exiting through the front of B,
        so the camera needs to be rotated 180° upon arrival.
        """
        M_Origin = self.get_portal_transform()
        M_destiny = self.destiny.get_portal_transform()
        inv_destiny = np.linalg.inv(M_destiny)
        virtualView = inv_destiny @ M_Origin @ realView
        return virtualView.astype(np.float32)
    
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

        glEnable(GL_POLYGON_OFFSET_FILL)  # Avoid z-fighting with the portal border
        glPolygonOffset(-1.0, -1.0)

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

        glDisable(GL_POLYGON_OFFSET_FILL)
