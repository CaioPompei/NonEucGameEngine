"""
Per-level puzzle state.

A `PuzzleManager` collects events fired by triggers and decides when the
puzzle is complete. The simplest case (`completion_event`) covers fase 01:
"enter zone X → done". Compound puzzles (multiple buttons, ordered
sequence, etc.) can extend `dispatch()` and `is_complete()` without
breaking the `main.py` loop, which only needs `should_transition()` and
the resulting `next_level`.
"""

from __future__ import annotations


class PuzzleManager:
    def __init__(self,
                 objective: str = "",
                 completion_event: str | None = None,
                 next_level: str | None = None):
        self.objective = objective
        self.completion_event = completion_event
        self.next_level = next_level

        self.completed = False
        self.flags: dict[str, bool] = {}

    def dispatch(self, event: str) -> None:
        """Record an event. Triggers call this via main.py's loop."""
        self.flags[event] = True
        if self.completion_event is not None and event == self.completion_event:
            if not self.completed:
                self.completed = True
                self._on_complete()

    def should_transition(self) -> bool:
        return self.completed and self.next_level is not None

    # ── Hooks ────────────────────────────────────────────────────────────────

    def _on_complete(self) -> None:
        """Override or monkey-patch from main.py to plug UI/SFX."""
        msg = self.objective or "puzzle completo"
        print(f"[Puzzle] OK: {msg}")
        if self.next_level:
            print(f"[Puzzle] proxima fase: {self.next_level}")
