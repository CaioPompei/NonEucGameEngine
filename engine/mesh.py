import ctypes

import numpy as np
from OpenGL.GL import *


class Mesh:
    """
    Responsability: send geometry to the GPU and draw it.
    Manages the VAO and VBO. Doesn't know anything about shaders or camera.

    Vertex layout (interleaved floats), by flags:
        position only            -> x, y, z                       (loc 0)
        have_normals             -> x, y, z, nx, ny, nz           (loc 0, 1)
        have_normals + have_uv   -> x, y, z, nx, ny, nz, u, v     (loc 0, 1, 2)
    """

    def __init__(self, vertices: np.ndarray,
                 have_normals: bool = False,
                 have_uv: bool = False):
        self.have_normals = have_normals
        self.have_uv = have_uv
        self.floats_per_vertex = 3 + (3 if have_normals else 0) + (2 if have_uv else 0)
        # Keep the source vertices around so static batching can read this
        # mesh's geometry (positions/normals/uv) and bake it into a merged
        # buffer. Cheap for the small primitives used here.
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self._num_vertices = len(vertices) // self.floats_per_vertex
        self._vao = self._create_vao(vertices, have_normals, have_uv)

    def draw(self):
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._num_vertices)
        glBindVertexArray(0)

    # VAO = Vertex Array Object = sets up how the vertex data is stored in the GPU
    def _create_vao(self, vertices, have_normals, have_uv):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        floats_per_vertex = 3 + (3 if have_normals else 0) + (2 if have_uv else 0)
        stride = floats_per_vertex * 4

        # location 0 = position (3 floats)
        offset = 0
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(0)
        offset += 3 * 4

        if have_normals:
            # location 1 = normal (3 floats)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
            glEnableVertexAttribArray(1)
            offset += 3 * 4

        if have_uv:
            # location 2 = texture coordinates (2 floats)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
            glEnableVertexAttribArray(2)
            offset += 2 * 4

        glBindVertexArray(0)
        return vao