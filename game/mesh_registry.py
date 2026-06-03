"""
Resolves mesh names (strings from level JSON) to shared Mesh instances.

A single VAO/VBO is cached per mesh name and reused across every entity
that requests it. Adding a new mesh type:

    registry.register("sphere", create_sphere_mesh)
"""

from typing import Callable

from engine.mesh import Mesh
from engine.primitives import create_cube_mesh


MeshFactory = Callable[[], Mesh]


class MeshRegistry:
    def __init__(self):
        self._factories: dict[str, MeshFactory] = {
            "cube": create_cube_mesh,
        }
        self._cache: dict[str, Mesh] = {}

    def register(self, name: str, factory: MeshFactory) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> Mesh:
        if name in self._cache:
            return self._cache[name]
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"Unknown mesh '{name}'. Registered: "
                f"{sorted(self._factories.keys())}")
        mesh = factory()
        self._cache[name] = mesh
        return mesh
