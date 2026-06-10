"""
Axis-aligned static collision world.

The level is built entirely from axis-aligned cube entities, so every
collider is an AABB. The player is also modelled as an AABB (a box around
the camera). `CollisionWorld.slide()` moves that box by a displacement and
resolves it against the static geometry with a per-axis "move and slide"
sweep — the blocked component is cancelled while the others keep going, so
the player glides along walls instead of sticking.

Rotation on collidable entities is ignored (all level geometry is
axis-aligned). If a rotated solid is ever added, its AABB will be the
axis-aligned bound of the rotated box, which is a conservative over-cover.
"""

from __future__ import annotations

import numpy as np

# Sub-step length (world units). Displacements longer than this are split so
# the player can't tunnel through the thin (0.25u) walls in a single frame.
_MAX_STEP = 0.1
_EPS = 1e-4


class CollisionWorld:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray):
        self.mins = np.asarray(mins, dtype=np.float32).reshape(-1, 3)
        self.maxs = np.asarray(maxs, dtype=np.float32).reshape(-1, 3)

    @classmethod
    def from_scene(cls, scene) -> "CollisionWorld":
        """Build the world from every `solid` entity in the scene."""
        mins, maxs = [], []
        for entity in scene.entities:
            if not getattr(entity, "solid", True):
                continue
            mn, mx = entity.world_aabb()
            mins.append(mn)
            maxs.append(mx)
        if not mins:
            return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))
        return cls(np.asarray(mins, np.float32), np.asarray(maxs, np.float32))

    def slide(self, center, half, displacement):
        """
        Move an AABB (defined by `center` and `half`-extents) by
        `displacement`, resolving collisions against the static world.

        Returns `(new_center, flags)` where `flags` is a dict with booleans:
            on_ground  — landed on a surface (moving down hit something)
            ceiling    — bonked head (moving up hit something)
            blocked_x  — horizontal X movement was stopped
            blocked_z  — horizontal Z movement was stopped
        """
        center = np.array(center, dtype=np.float32)
        half = np.asarray(half, dtype=np.float32)
        disp = np.asarray(displacement, dtype=np.float32)

        flags = {"on_ground": False, "ceiling": False,
                 "blocked_x": False, "blocked_z": False}

        if len(self.mins) == 0:
            return center + disp, flags

        steps = int(max(1, np.ceil(np.max(np.abs(disp)) / _MAX_STEP)))
        step = disp / steps
        for _ in range(steps):
            # Resolve horizontal axes first, then vertical, so grounding is
            # detected against a fully-resolved horizontal position.
            self._resolve_axis(center, half, step[0], 0, flags)
            self._resolve_axis(center, half, step[2], 2, flags)
            self._resolve_axis(center, half, step[1], 1, flags)

        return center, flags

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resolve_axis(self, center, half, d, axis, flags):
        if d == 0.0:
            return
        center[axis] += d

        pmin = center - half
        pmax = center + half
        # Overlap on all three axes simultaneously (vectorized over all boxes).
        overlap = np.all((pmin < self.maxs) & (pmax > self.mins), axis=1)
        if not overlap.any():
            return

        if d > 0.0:
            # Leading face is +; snap behind the closest box's min face.
            center[axis] = float(np.min(self.mins[overlap, axis])) - half[axis] - _EPS
            if axis == 1:
                flags["ceiling"] = True
        else:
            # Leading face is -; snap above the closest box's max face.
            center[axis] = float(np.max(self.maxs[overlap, axis])) + half[axis] + _EPS
            if axis == 1:
                flags["on_ground"] = True

        if axis == 0:
            flags["blocked_x"] = True
        elif axis == 2:
            flags["blocked_z"] = True
