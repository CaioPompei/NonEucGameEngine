"""
Skybox — an infinitely-distant cube of sky around the world.

A reusable engine tool: give it six images (one per cube face) and it draws
them as the background of the scene. It knows nothing about levels or JSON —
the game layer resolves paths and decides which skybox a level uses.

Two tricks make it read as "infinitely far away":

  * The camera's translation is stripped from the view matrix, so the cube is
    always centered on the eye — you can never walk up to the wall.
  * It is drawn with depth forced to the far plane (gl_Position = pos.xyww)
    and GL_LEQUAL, so any real geometry already in the depth buffer covers it.
    Draw it AFTER the scene to skip every pixel that's already painted.

The face images are sampled by a 3D direction vector (a cubemap), so corners
are seamless — unlike pasting a flat photo on each face. Cubemaps use a
top-left texel origin, so faces are uploaded WITHOUT the vertical flip that
engine.texture.Texture applies to 2D textures.
"""

import ctypes

import numpy as np
from OpenGL.GL import *
from PIL import Image

from engine.shader import Shader

# Face order required by OpenGL's cubemap targets.
_FACE_TARGETS = (
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,  # right
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,  # left
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,  # top
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,  # bottom
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,  # front
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,  # back
)

# Unit cube, positions only. The vertex position doubles as the sampling
# direction, so no UVs or normals are needed. Winding is irrelevant because
# the engine does not enable face culling.
_CUBE_VERTICES = np.array([
    -1,  1, -1,  -1, -1, -1,   1, -1, -1,   1, -1, -1,   1,  1, -1,  -1,  1, -1,
    -1, -1,  1,  -1, -1, -1,  -1,  1, -1,  -1,  1, -1,  -1,  1,  1,  -1, -1,  1,
     1, -1, -1,   1, -1,  1,   1,  1,  1,   1,  1,  1,   1,  1, -1,   1, -1, -1,
    -1, -1,  1,  -1,  1,  1,   1,  1,  1,   1,  1,  1,   1, -1,  1,  -1, -1,  1,
    -1,  1, -1,   1,  1, -1,   1,  1,  1,   1,  1,  1,  -1,  1,  1,  -1,  1, -1,
    -1, -1, -1,  -1, -1,  1,   1, -1, -1,   1, -1, -1,  -1, -1,  1,   1, -1,  1,
], dtype=np.float32)

_VS = """
#version 330 core
layout (location = 0) in vec3 position;
out vec3 sample_dir;
uniform mat4 view;
uniform mat4 projection;
void main() {
    sample_dir = position;
    vec4 pos = projection * view * vec4(position, 1.0);
    gl_Position = pos.xyww;   // z/w == 1.0 -> always at the far plane
}
"""

_FS = """
#version 330 core
in vec3 sample_dir;
out vec4 fragColor;
uniform samplerCube skybox;
void main() {
    fragColor = texture(skybox, sample_dir);
}
"""


class Skybox:
    """A cubemap drawn around the world. Construct after the GL context exists.

    `face_paths` is six image paths in the order:
        right (+X), left (-X), top (+Y), bottom (-Y), front (+Z), back (-Z).
    """

    # Shared across every Skybox instance: the GLSL program never changes, so
    # there's no reason to recompile it on each level transition.
    _shared_shader: Shader | None = None

    def __init__(self, face_paths):
        faces = list(face_paths)
        if len(faces) != 6:
            raise ValueError(
                f"Skybox needs exactly 6 face paths, got {len(faces)}")

        self._texture = self._load_cubemap(faces)
        self._vao = self._build_cube()

        if Skybox._shared_shader is None:
            Skybox._shared_shader = Shader(_VS, _FS)
        self._shader = Skybox._shared_shader

    # ── Public API ───────────────────────────────────────────────────────────

    def draw(self, view, projection):
        """Render the sky. Call after the scene so covered pixels are skipped.

        `view` is the camera view matrix; its translation is stripped here so
        the box stays centered on the eye.
        """
        # Strip translation: pyrr stores it in the last row (cols 0..2).
        view_static = np.array(view, dtype=np.float32, copy=True)
        view_static[3, 0:3] = 0.0

        glDepthFunc(GL_LEQUAL)        # depth == far plane must pass
        glDisable(GL_STENCIL_TEST)    # ignore whatever the portals left set

        self._shader.use()
        self._shader.set_matrix4("view", view_static)
        self._shader.set_matrix4("projection", projection)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._texture)
        self._shader.set_int("skybox", 0)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        glDepthFunc(GL_LESS)          # restore the engine default

    # ── Internals ────────────────────────────────────────────────────────────

    def _load_cubemap(self, face_paths):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        for target, path in zip(_FACE_TARGETS, face_paths):
            # No vertical flip: cubemaps sample from a top-left texel origin.
            image = Image.open(path).convert("RGB")
            width, height = image.size
            glTexImage2D(target, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, image.tobytes())

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        return texture

    def _build_cube(self):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, _CUBE_VERTICES.nbytes,
                     _CUBE_VERTICES, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)
        return vao
