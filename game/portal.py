import math
from typing import Optional

import numpy as np
import pyrr
from OpenGL.GL import *

from engine.mesh import Mesh
from engine.shader import Shader
from math3d.portal_math import (
    calculate_virtual_view_cached as _calculate_virtual_view_cached,
    calculate_oblique_projection as _calculate_oblique_projection,
    calculate_traversal_transform as _calculate_traversal_transform,
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


_PORTAL_SCALE = np.array([12.0, 5.0, 5.0], dtype=np.float32)
# Half extents of the source mesh (before scaling): X in [-0.5, 0.5], Y in [-1, 1].
_MESH_HALF_X = 0.5
_MESH_HALF_Y = 1.0


class Portal:
    """
    A static portal in the scene.

    Pre-computed on construction (rotation/position never change):
        - portal_transform (R @ T) and its inverse
        - model matrix (S @ R @ T)
        - world-space front normal
        - cos/sin of -rotation for fast world→local XY conversion
        - rectangular opening half-extents

    Pre-computed on `link_to()`:
        - traversal_transform: applied to a world-space point/direction to
          carry it from the front of this portal to the front of the linked
          one (with the 180° Y flip that paired portals require).
    """

    mesh_quad: Mesh = None  # Shared mesh for all portals.

    def __init__(self, position, rotation, color):
        """
            position: (x, y, z) world coordinates of the portal center
            rotation: degrees - Where the portal is facing (around Y)
            color:    RGB color (0..1) for debug/border drawing
        """
        self.position = np.array(position, dtype=np.float32)
        self.rotation = float(rotation)
        self.color = np.array(color, dtype=np.float32)
        self.destiny: 'Portal' = None

        # --- static caches (portal never moves after construction) ---
        self._normal_world = _portal_normal_world(self.rotation)
        self._portal_transform = self._compute_portal_transform()
        self._inv_portal_transform = np.linalg.inv(
            self._portal_transform).astype(np.float32)
        self._model_matrix = self._compute_model_matrix()
        self._half_extents = np.array(
            [_MESH_HALF_X * _PORTAL_SCALE[0],
             _MESH_HALF_Y * _PORTAL_SCALE[1]],
            dtype=np.float32,
        )
        # cos/sin of -rotation: rotates a world-space delta into portal-local XY.
        inv_theta = -math.radians(self.rotation)
        self._cos_inv_rot = math.cos(inv_theta)
        self._sin_inv_rot = math.sin(inv_theta)

        # Filled in by link_to():
        self._traversal_transform: Optional[np.ndarray] = None

        if Portal.mesh_quad is None:
            Portal.mesh_quad = create_portal_mesh()

    # ── Construction helpers ─────────────────────────────────────────────────

    def _compute_portal_transform(self) -> np.ndarray:
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)
        R = pyrr.matrix44.create_from_y_rotation(
            math.radians(self.rotation), dtype=np.float32)
        return (R @ T).astype(np.float32)

    def _compute_model_matrix(self) -> np.ndarray:
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)
        R = pyrr.matrix44.create_from_y_rotation(
            math.radians(self.rotation), dtype=np.float32)
        S = pyrr.matrix44.create_from_scale(_PORTAL_SCALE, dtype=np.float32)
        return (S @ R @ T).astype(np.float32)

    # ── Public accessors ─────────────────────────────────────────────────────

    def get_model_matrix(self) -> np.ndarray:
        return self._model_matrix

    def get_portal_transform(self) -> np.ndarray:
        return self._portal_transform

    # ── Pairing ──────────────────────────────────────────────────────────────

    def link_to(self, other: 'Portal') -> None:
        """
        Pair this portal with `other`. Sets `.destiny` on both sides and
        pre-computes the traversal transform for each direction (so the
        teleport at run-time is a single mat-vec multiply).
        """
        self.destiny = other
        other.destiny = self
        self._traversal_transform = _calculate_traversal_transform(
            self._portal_transform, other._portal_transform)
        other._traversal_transform = _calculate_traversal_transform(
            other._portal_transform, self._portal_transform)

    # ── Geometric queries ────────────────────────────────────────────────────

    def signed_distance(self, point) -> float:
        """
        Dot product of the world-space front normal with (point - portal_pos).
        > 0  → point is in front of the portal,
        < 0  → behind, = 0 on the plane.
        """
        p = np.asarray(point, dtype=np.float32) - self.position
        return float(self._normal_world.dot(p))

    def is_camera_in_front(self, camera_pos) -> bool:
        return self.signed_distance(camera_pos) > 0.0

    def _intersection_in_opening(self,
                                 prev_pos: np.ndarray,
                                 curr_pos: np.ndarray,
                                 prev_sd: float,
                                 curr_sd: float) -> bool:
        """
        Pre-condition: prev_sd > 0 and curr_sd <= 0 (segment crossed the
        plane front→back). Returns True iff the intersection point lies
        inside the rectangular opening.
        """
        denom = prev_sd - curr_sd
        if denom < 1e-6:
            return False
        t = prev_sd / denom
        hit = prev_pos + t * (curr_pos - prev_pos)

        # World-space delta → portal-local XY (rotation by -theta around Y).
        delta = hit - self.position
        local_x = self._cos_inv_rot * delta[0] + self._sin_inv_rot * delta[2]
        local_y = delta[1]

        return (abs(local_x) <= self._half_extents[0] and
                abs(local_y) <= self._half_extents[1])

    def attempt_traversal(self,
                          prev_pos: np.ndarray,
                          prev_sd: Optional[float],
                          curr_pos: np.ndarray
                          ):
        """
        O(1) traversal check. Returns (traversal_transform, curr_sd).

        - `traversal_transform` is None when the segment did NOT cross the
          portal from the front through its opening.
        - `curr_sd` is returned so the caller can cache it for next frame
          without recomputing the dot product.

        `prev_sd` is the cached signed distance from the previous frame
        (None on the very first call).
        """
        curr_sd = self.signed_distance(curr_pos)
        if (self._traversal_transform is None
                or prev_sd is None
                or prev_sd <= 0.0
                or curr_sd > 0.0):
            return None, curr_sd
        prev = np.asarray(prev_pos, dtype=np.float32)
        curr = np.asarray(curr_pos, dtype=np.float32)
        if not self._intersection_in_opening(prev, curr, prev_sd, curr_sd):
            return None, curr_sd
        return self._traversal_transform, curr_sd

    # ── Rendering ────────────────────────────────────────────────────────────

    def calculate_virtual_view(self, realView: np.ndarray) -> np.ndarray:
        """
        View matrix da câmera virtual no lado destino do portal.
        Usa a inversa pré-cacheada do destino — evita `np.linalg.inv` por
        frame (importante porque o renderer recursa até `max_depth`).
        """
        return _calculate_virtual_view_cached(
            realView,
            self._portal_transform,
            self.destiny._inv_portal_transform,
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

    def draw_stencil(self, shader: Shader):
        """
            Draw the opening of the portal, Only on the stencil buffer.
            Don't write Color or Depth
            Stencil select the pixels that belong to the portal surface.
        """
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)

        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)

        shader.set_matrix4("model", self._model_matrix)
        shader.set_vec3("color", self.color)
        Portal.mesh_quad.draw()

        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def draw_portal_border(self, shader: Shader):
        """Draws the portal quad as a colored border (debug)."""
        shader.set_matrix4("model", self._model_matrix)
        shader.set_vec3("objectColor", self.color)
        Portal.mesh_quad.draw()
