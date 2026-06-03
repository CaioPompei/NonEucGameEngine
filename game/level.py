"""
A loaded level — purely data. Built by `LevelLoader.load()`.

Doesn't hold game-loop logic; that lives in main.py and the systems
(PortalRenderer, PuzzleManager, Collision) that consume the Level.
"""

from dataclasses import dataclass, field

from engine.scene import Scene
from game.portal import Portal
from game.puzzle import PuzzleManager
from game.trigger import Trigger


@dataclass
class Level:
    name: str
    scene: Scene
    portals: list[Portal] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)
    puzzle: PuzzleManager = field(default_factory=PuzzleManager)
    player_start: tuple[float, float, float] = (0.0, 0.0, 0.0)
