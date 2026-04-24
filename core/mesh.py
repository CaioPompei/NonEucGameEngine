import numpy as np
from OpenGL.GL import *


class Mesh:
    """
    Responsability: send geometry to the GPU and draw it.
    Manages the VAO and VBO. Doesn't know anything about shaders or camera.
    """

    def __init__(self, vertices: np.ndarray):
        """
        vertices: numpy array float32 with the vertices already organized.
        For now, it assumes each vertex has 3 floats (x, y, z).
        """
        self._num_vertices = len(vertices) // 3
        self._vao = self._create_vao(vertices)

    def draw(self):
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._num_vertices)
        glBindVertexArray(0)

    # VAO = Vertex Array Object = sets up how the vertex data is stored in the GPU
    def _create_vao(self, vertices):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, None)
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)
        return vao