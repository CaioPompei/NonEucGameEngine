"""
Cubemap shadow map for omnidirectional point lights.

Implementation: depth-only cubemap rendered via 6 sequential passes (one per
face). A geometry-shader-based single-pass layered approach exists but the
6-pass version is simpler and only runs at bake time, so the cost is
irrelevant in practice (the scene is static).

The cubemap stores normalized depth in [0, 1]; the depth shader writes
`gl_FragDepth = length(world_pos - light_pos) / far_plane`. The main shader
multiplies the sampled value by `far_plane` to recover the actual distance
for comparison.
"""

from __future__ import annotations

import math

import numpy as np
import pyrr
from OpenGL.GL import *


# Six (target, up) pairs for the cubemap faces. The "up" vectors look
# inverted because OpenGL cubemaps were defined with the RenderMan
# convention (top-left origin per face), so face textures are sampled
# upside-down relative to a normal view.
_FACE_TARGETS = (
    (( 1.0,  0.0,  0.0), (0.0, -1.0,  0.0)),  # +X
    ((-1.0,  0.0,  0.0), (0.0, -1.0,  0.0)),  # -X
    (( 0.0,  1.0,  0.0), (0.0,  0.0,  1.0)),  # +Y
    (( 0.0, -1.0,  0.0), (0.0,  0.0, -1.0)),  # -Y
    (( 0.0,  0.0,  1.0), (0.0, -1.0,  0.0)),  # +Z
    (( 0.0,  0.0, -1.0), (0.0, -1.0,  0.0)),  # -Z
)


class CubeShadowMap:
    """Owns one depth-cubemap + FBO. Reusable across bakes."""

    def __init__(self, resolution: int = 1024):
        self.resolution = int(resolution)
        self._cubemap = self._create_cubemap(self.resolution)
        self._fbo = self._create_fbo(self._cubemap)

    # ── Bake ─────────────────────────────────────────────────────────────────

    def bake(self,
             light_pos: np.ndarray,
             far_plane: float,
             scene,
             depth_shader) -> None:
        """
        Render `scene` depth from `light_pos` into the 6 cube faces.
        `depth_shader` must have uniforms `view`, `projection`, `lightPos`,
        `far_plane` and use the same vertex layout as the regular scene
        shader (position + normal).
        """
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_cull = glIsEnabled(GL_CULL_FACE)
        prev_cull_mode = glGetIntegerv(GL_CULL_FACE_MODE)

        glViewport(0, 0, self.resolution, self.resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Front-face culling during the depth pass mitigates self-shadowing
        # ("peter panning" is preferred over shadow acne on closed meshes).
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        projection = pyrr.matrix44.create_perspective_projection(
            90.0, 1.0, 0.1, float(far_plane), dtype=np.float32)

        depth_shader.use()
        depth_shader.set_matrix4("projection", projection)
        depth_shader.set_vec3("lightPos", light_pos)
        depth_shader.set_float("far_plane", float(far_plane))

        for face, (target, up) in enumerate(_FACE_TARGETS):
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                self._cubemap, 0)
            glClear(GL_DEPTH_BUFFER_BIT)

            eye = light_pos
            tgt = eye + np.array(target, dtype=np.float32)
            view = pyrr.matrix44.create_look_at(
                eye=eye, target=tgt,
                up=np.array(up, dtype=np.float32),
                dtype=np.float32)
            depth_shader.set_matrix4("view", view)

            # `scene.draw_depth` doesn't bind colors/normals — just geometry.
            scene.draw_depth(depth_shader)

        # Restore previous GL state.
        glCullFace(prev_cull_mode)
        if not prev_cull:
            glDisable(GL_CULL_FACE)
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glViewport(prev_viewport[0], prev_viewport[1],
                   prev_viewport[2], prev_viewport[3])

    def bind(self, texture_unit: int) -> None:
        """Bind this cubemap to `GL_TEXTUREi` for sampling by the main shader."""
        glActiveTexture(GL_TEXTURE0 + texture_unit)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap)

    # ── GL object creation ───────────────────────────────────────────────────

    @staticmethod
    def _create_cubemap(resolution: int) -> int:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
        for face in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
                         GL_DEPTH_COMPONENT24,
                         resolution, resolution, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        return tex

    @staticmethod
    def _create_fbo(cubemap: int) -> int:
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        # Initial attachment (face will be replaced per-face during bake).
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X, cubemap, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Shadow FBO incomplete: 0x{status:x}")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return fbo


# A 1x1 "null" cubemap used to fill unused shadow slots. Sampler arrays in
# GLSL 330 must all be bound to a valid texture even if logically disabled.
_NULL_CUBEMAP: int | None = None


def get_null_cubemap() -> int:
    """Lazily create (once) and return a 1×1 depth cubemap that always
    samples 1.0 (== max distance, never in shadow). Used to fill unused
    light slots."""
    global _NULL_CUBEMAP
    if _NULL_CUBEMAP is not None:
        return _NULL_CUBEMAP

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)
    one = (np.ones((1, 1), dtype=np.float32)).tobytes()
    for face in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0,
                     GL_DEPTH_COMPONENT24, 1, 1, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, one)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    _NULL_CUBEMAP = tex
    return tex


def bind_null_cubemap(texture_unit: int) -> None:
    glActiveTexture(GL_TEXTURE0 + texture_unit)
    glBindTexture(GL_TEXTURE_CUBE_MAP, get_null_cubemap())
