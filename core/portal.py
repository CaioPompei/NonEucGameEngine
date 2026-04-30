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
    return Mesh(vertices)

class Portal:
    """
        Represents a portal in the scene.
        Each portal has a pair. Together they define a "doorway" between two locations.

        Responsabylities:
        - Keep Position and orientation
        - Calculate virtual camera rendering
        - Draw the portal surface on stencil buffer    
    """

    def __init__(self, position, rotation, color):
        """
            position: (x, y, z) world coordinates of the portal center
            rotation: degrees - Where the portal is facing
            color: RGB color for debugging (0.0 to 1.0)
        """
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation
        self.color = np.array(color, dtype=np.float32)
        self.destiny: 'Portal' = None

        if Portal.mesh_quad is None:
            Portal.mesh_quad = create_portal_mesh()

    def calculate_virtual_view(self, realView: np.ndarray) -> np.ndarray:
        """ Do it in english!
        Calculate the View Matrix for the virtual camera on the destination side.

        The formula is:
            realView = real_view_do_destino
                         = view × inv(model_origem) × rotation_180 × destinyModel

        Rotation Y is necessary because the portals face each other:
        entering through the front of A means exiting through the front of B,
        so the camera needs to be rotated 180° upon arrival.
        """
        # Rotation 180° around Y
        rot_180 = pyrr.matrix44.create_from_y_rotation(
            math.radians(180.0), dtype=np.float32)

        M_Origin = self.get_model_matrix()
        M_dstiny = self.destiny.get_model_matrix()

        # virtualView = realView @ np.linalg.inv(M_Origin) @ rot_180 @ M_dstiny
        inv_origin = np.linalg.inv(M_Origin)
        virtualView = realView @ inv_origin @ rot_180 @ M_dstiny
        return virtualView.astype(np.float32)
    
    # Rendering

    def Draw_stencil(self, shader: Shader):
        pass 
