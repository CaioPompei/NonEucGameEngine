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

# A horizontal-cross cubemap is one image laid out 4 columns x 3 rows:
#                [ +Y ]
#         [ -X ][ +Z ][ +X ][ -Z ]
#                [ -Y ]
# Cell (column, row) of each face, in the SAME order as _FACE_TARGETS, so a
# single PNG can be sliced into the six faces without pre-splitting it.
_CROSS_COLS, _CROSS_ROWS = 4, 3
_CROSS_CELLS = (
    (2, 1),  # right  (+X)
    (0, 1),  # left   (-X)
    (1, 0),  # top    (+Y)
    (1, 2),  # bottom (-Y)
    (1, 1),  # front  (+Z)
    (3, 1),  # back   (-Z)
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

    Build it with one of the constructors:
        Skybox.from_cross(path)   one PNG laid out as a 4x3 horizontal cross
        Skybox.from_faces(paths)  six separate face images

    Faces are ordered right (+X), left (-X), top (+Y), bottom (-Y),
    front (+Z), back (-Z) — matching `_FACE_TARGETS`.
    """

    # Shared across every Skybox instance: the GLSL program never changes, so
    # there's no reason to recompile it on each level transition.
    _shared_shader: Shader | None = None

    def __init__(self, face_images):
        """`face_images` is six PIL Images (RGB) in `_FACE_TARGETS` order.
        Most callers want `from_cross` or `from_faces` instead."""
        faces = list(face_images)
        if len(faces) != 6:
            raise ValueError(
                f"Skybox needs exactly 6 face images, got {len(faces)}")

        self._texture = self._load_cubemap(faces)
        self._vao = self._build_cube()

        if Skybox._shared_shader is None:
            Skybox._shared_shader = Shader(_VS, _FS)
        self._shader = Skybox._shared_shader

    @classmethod
    def from_faces(cls, face_paths):
        """Build from six image files (one per face), in `_FACE_TARGETS` order."""
        paths = list(face_paths)
        if len(paths) != 6:
            raise ValueError(
                f"Skybox needs exactly 6 face paths, got {len(paths)}")
        return cls([Image.open(p).convert("RGB") for p in paths])

    @classmethod
    def from_cross(cls, path):
        """Build from a single horizontal-cross cubemap image (4 cols x 3 rows).

        The image must have a 4:3 aspect (e.g. 2048x1536); it is sliced into the
        six square faces in-memory, so a level can point at one PNG with no
        pre-splitting. Faces use a top-left origin (no vertical flip).
        """
        image = Image.open(path).convert("RGB")
        w, h = image.size
        if w * _CROSS_ROWS != h * _CROSS_COLS:
            raise ValueError(
                f"Skybox cross image must be {_CROSS_COLS}:{_CROSS_ROWS} "
                f"(e.g. 2048x1536), got {w}x{h}: {path}")
        f = w // _CROSS_COLS
        faces = [image.crop((c * f, r * f, (c + 1) * f, (r + 1) * f))
                 for (c, r) in _CROSS_CELLS]
        return cls(faces)

    # ── Public API ───────────────────────────────────────────────────────────

    def draw(self, view, projection, stencil_ref=None):
        """Render the sky. Call after the scene so covered pixels are skipped.

        `view` is the camera view matrix; its translation is stripped here so
        the box stays centered on the eye.

        `stencil_ref`: when None (the main pass), the stencil test is disabled
        so the sky fills every uncovered far-plane pixel. When set — drawn
        inside a portal's virtual view — the stencil test is kept so the sky
        only fills that portal's masked region (stencil == stencil_ref).
        """
        # Strip translation: pyrr stores it in the last row (cols 0..2).
        view_static = np.array(view, dtype=np.float32, copy=True)
        view_static[3, 0:3] = 0.0

        glDepthFunc(GL_LEQUAL)        # depth == far plane must pass
        if stencil_ref is None:
            glDisable(GL_STENCIL_TEST)    # main pass: ignore portal leftovers
        else:
            glEnable(GL_STENCIL_TEST)
            glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)

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

    def _load_cubemap(self, face_images):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        for target, image in zip(_FACE_TARGETS, face_images):
            # No vertical flip: cubemaps sample from a top-left texel origin.
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
