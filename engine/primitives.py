import numpy as np
from engine.mesh import Mesh

def create_cube_mesh():
    """
    Unit cube centered with normals per face. 36 vertices (6 faces * 2 triangles * 3 vertices).
    Each group of 6 vertices corresponds to a face, a face = 2 triangles = 6 vertices.
    Format per vertex: x, y, z, nx, ny, nz (position + normal)
    """
    vertices = np.array([
        # pos                normal
        # Face frontal — normal points for +Z
        -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,   0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,   0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,   0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,   0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,   0.0,  0.0,  1.0,

        # Face traseira — normal points for -Z
        -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,   0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,   0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,   0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,   0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,   0.0,  0.0, -1.0,

        # Face esquerda — normal points for -X
        -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,  -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,  -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5,  -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5,  -1.0,  0.0,  0.0,

        # Face direita — normal points for +X
         0.5,  0.5,  0.5,   1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,   1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,   1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,   1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,   1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,   1.0,  0.0,  0.0,

        # Face inferior — normal points for -Y
        -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,   0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,   0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,   0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,   0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,   0.0, -1.0,  0.0,

        # Face superior — normal points for +Y
        -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,   0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,   0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,   0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,   0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,   0.0,  1.0,  0.0,
    ], dtype=np.float32)

    return Mesh(vertices, have_normals=True)