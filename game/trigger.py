"""
Axis-aligned bounding-box trigger zones.

A trigger fires `on_enter` (an event name string) exactly once, the first
frame the player's position enters the AABB. To rearm it, call `reset()`.
This is enough for the 3 phases described in the spec; if you need
enter/exit/stay semantics, extend `check()` to return distinct events.
"""

from __future__ import annotations

import numpy as np


class Trigger:
    def __init__(self,
                 id: str,
                 aabb_min,
                 aabb_max,
                 on_enter: str):
        self.id = id
        self.aabb_min = np.asarray(aabb_min, dtype=np.float32)
        self.aabb_max = np.asarray(aabb_max, dtype=np.float32)
        self.on_enter = on_enter
        self._activated = False

    def reset(self) -> None:
        self._activated = False

    def check(self, position) -> str | None:
        """Return `on_enter` if the position just entered the AABB, else None."""
        if self._activated:
            return None
        p = np.asarray(position, dtype=np.float32)
        if np.all(p >= self.aabb_min) and np.all(p <= self.aabb_max):
            self._activated = True
            return self.on_enter
        return None
