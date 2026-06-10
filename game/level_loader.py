"""
JSON → Level.

Schema (every field optional unless noted; missing fields fall back to
sensible defaults so partial JSONs still load):

    {
        "name": "Fase 01",                          # required
        "player_start": [x, y, z],
        "ambient_color": [r, g, b],

        "entities": [
            {
                "mesh": "cube",                     # MeshRegistry key
                "position": [x, y, z],
                "rotation": [x, y, z],              # Euler degrees
                "scale": [x, y, z],
                "color": [r, g, b],                 # tint (multiplies texture)
                "solid": true,                      # player collision (default true)
                "texture": "textures/wall.png",     # optional, path from repo root
                "texture_scale": [u, v]             # optional UV tiling (default 1,1)
            }, ...
        ],

        "lights": [
            {
                "type": "point",
                "position": [x, y, z],
                "color": [r, g, b],
                "intensity": 1.0,
                "range": 35.0,
                "cast_shadows": true,
                "shadow_resolution": 1024
            }, ...
        ],

        "portals": [
            {
                "id": "A",
                "linked_to": "B",                   # destination id; "" / omit
                                                    # = inert exit (no teleport).
                                                    # one-directional: set on
                                                    # both ids for a two-way pair
                "position": [x, y, z],
                "rotation": 0.0,                    # Y-axis degrees
                "color": [r, g, b],
                "scale": [width, height]            # optional opening size
            }, ...
        ],

        "triggers": [
            {
                "id": "win",
                "aabb_min": [x, y, z],
                "aabb_max": [x, y, z],
                "on_enter": "complete_puzzle"       # event name dispatched
            }, ...
        ],

        "puzzle": {
            "objective": "texto exibido ao vencer",
            "completion_event": "complete_puzzle",  # event that completes it
            "next_level": "level_02.json"           # null if last fase
        }
    }
"""

import json
from pathlib import Path

from engine.entity import Entity
from engine.light import PointLight
from engine.scene import Scene
from engine.texture import TextureRegistry
from game.level import Level
from game.mesh_registry import MeshRegistry
from game.portal import Portal
from game.puzzle import PuzzleManager
from game.trigger import Trigger

# Repo root: texture paths in level JSON are resolved relative to this
# (game/ -> project root), unless the path is already absolute.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LevelLoader:
    def __init__(self,
                 mesh_registry: MeshRegistry | None = None,
                 texture_registry: TextureRegistry | None = None):
        self.mesh_registry = mesh_registry or MeshRegistry()
        # Cached across levels so a shared texture is uploaded only once.
        self.texture_registry = texture_registry or TextureRegistry()

    def load(self, path: str | Path) -> Level:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._build(data, source=path)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build(self, data: dict, source: Path) -> Level:
        if "name" not in data:
            raise ValueError(f"{source}: missing required field 'name'")

        scene = Scene(ambient_color=tuple(
            data.get("ambient_color", (0.05, 0.05, 0.05))))

        for entity_data in data.get("entities", []):
            scene.add_entity(self._build_entity(entity_data, source))

        for light_data in data.get("lights", []):
            scene.add_light(self._build_light(light_data, source))

        portals, portal_by_id = self._build_portals(
            data.get("portals", []), source)

        triggers = [self._build_trigger(t, source)
                    for t in data.get("triggers", [])]
        puzzle = self._build_puzzle(data.get("puzzle", {}))

        return Level(
            name=data["name"],
            scene=scene,
            portals=portals,
            triggers=triggers,
            puzzle=puzzle,
            player_start=tuple(data.get("player_start", (0.0, 0.0, 0.0))),
        )

    def _build_entity(self, e: dict, source: Path) -> Entity:
        if "mesh" not in e:
            raise ValueError(f"{source}: entity missing 'mesh' field: {e}")
        mesh = self.mesh_registry.get(e["mesh"])
        return Entity(
            mesh=mesh,
            position=tuple(e.get("position", (0.0, 0.0, 0.0))),
            rotation=tuple(e.get("rotation", (0.0, 0.0, 0.0))),
            scale=tuple(e.get("scale", (1.0, 1.0, 1.0))),
            color=tuple(e.get("color", (1.0, 1.0, 1.0))),
            solid=bool(e.get("solid", True)),
            texture=self._load_texture(e.get("texture"), source),
            texture_scale=tuple(e.get("texture_scale", (1.0, 1.0))),
        )

    def _load_texture(self, rel_path, source: Path):
        """Resolve a texture path (relative to repo root) and load it, or
        return None when the entity declares no texture."""
        if not rel_path:
            return None
        path = Path(rel_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"{source}: texture not found: {path}")
        return self.texture_registry.get(path)

    def _build_light(self, l: dict, source: Path) -> PointLight:
        light_type = l.get("type", "point")
        if light_type != "point":
            raise ValueError(
                f"{source}: light type '{light_type}' not supported yet "
                f"(only 'point' for now)")
        return PointLight(
            position=tuple(l.get("position", (0.0, 0.0, 0.0))),
            color=tuple(l.get("color", (1.0, 1.0, 1.0))),
            intensity=float(l.get("intensity", 1.0)),
            range=float(l.get("range", 30.0)),
            cast_shadows=bool(l.get("cast_shadows", True)),
            shadow_resolution=int(l.get("shadow_resolution", 1024)),
        )

    def _build_trigger(self, t: dict, source: Path) -> Trigger:
        for required in ("id", "aabb_min", "aabb_max", "on_enter"):
            if required not in t:
                raise ValueError(
                    f"{source}: trigger missing '{required}': {t}")
        return Trigger(
            id=t["id"],
            aabb_min=tuple(t["aabb_min"]),
            aabb_max=tuple(t["aabb_max"]),
            on_enter=t["on_enter"],
        )

    def _build_puzzle(self, p: dict) -> PuzzleManager:
        return PuzzleManager(
            objective=p.get("objective", ""),
            completion_event=p.get("completion_event"),
            next_level=p.get("next_level"),
        )

    def _build_portals(self, portal_list: list[dict], source: Path
                       ) -> tuple[list[Portal], dict[str, Portal]]:
        # Two-pass: instantiate all, then resolve `linked_to` references.
        portals: list[Portal] = []
        by_id: dict[str, Portal] = {}

        for p in portal_list:
            if "id" not in p:
                raise ValueError(f"{source}: portal missing 'id': {p}")
            portal = Portal(
                position=tuple(p.get("position", (0.0, 0.0, 0.0))),
                rotation=float(p.get("rotation", 0.0)),
                color=tuple(p.get("color", (1.0, 1.0, 1.0))),
                scale=p.get("scale"),  # None -> Portal's default opening size
            )
            if p["id"] in by_id:
                raise ValueError(f"{source}: duplicate portal id '{p['id']}'")
            by_id[p["id"]] = portal
            portals.append(portal)

        # Links are one-directional: each portal independently points at its
        # destination. Declare `linked_to` on both portals for a two-way pair,
        # or on just one for a one-way portal (the other is an inert exit).
        # An empty / null / missing `linked_to` means "no destination".
        for p in portal_list:
            target = p.get("linked_to")
            if not target:  # None or "" -> inert exit, no teleport, no view
                continue
            if target == p["id"]:
                raise ValueError(
                    f"{source}: portal '{p['id']}' links to itself")
            if target not in by_id:
                raise ValueError(
                    f"{source}: portal '{p['id']}' links to unknown id "
                    f"'{target}'")
            by_id[p["id"]].set_destination(by_id[target])

        return portals, by_id
