import numpy as np
from OpenGL.GL import *


class Mesh:
    """
    Responsability: send geometry to the GPU and draw it.
    Manages the VAO and VBO. Doesn't know anything about shaders or camera.
    """

    def __init__(self, vertices: np.ndarray, have_normals: bool = False):
        """
        vertices: numpy array float32 with the vertices already organized.
        For now, it assumes each vertex has 3 floats (x, y, z).
        """
        self._num_vertices = len(vertices) // 3
        self._vao = self._create_vao(vertices, have_normals)

    def draw(self):
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._num_vertices)
        glBindVertexArray(0)

    # VAO = Vertex Array Object = sets up how the vertex data is stored in the GPU
    def _create_vao(self, vertices, have_normals):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        if have_normals:
            stride = 6 * 4  # 6 floats per vertex (3 for position, 3 for normal), 4 bytes per float

            # location = 0 first 3 floats are position
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
            glEnableVertexAttribArray(0)

            #location = 1 normal, next 3 floats
            import ctypes
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)
        else:
            # no normals, like before
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
            glEnableVertexAttribArray(0)

        glBindVertexArray(0)
        return vao