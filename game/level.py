import numpy as np
from core.mesh import Mesh
from core.entity import Entity
from core.scene import Scene
from game.meshes import create_cube_mesh


def create_room(width=10.0, height=4.0, depth=10.0) -> Scene:
    """
        Create a closed room using 6 faces.
    """
    scene = Scene()
    cube = create_cube_mesh()

    W, H, D, T = width, height, depth, 0.25

    # Floor — dark gray
    scene.add_entity(Entity(cube, position=(0.0, -H*2, 0.0), scale=(W, T, D), color=(0.4, 0.4, 0.4)))
    # Ceiling — light gray
    scene.add_entity(Entity(cube, position=(0.0, H*2, 0.0), scale=(W, T, D), color=(0.8, 0.8, 0.8)))
    # Front wall — bluish gray
    scene.add_entity(Entity(cube, position=(0.0, 0.0, -D*2), scale=(W, H, T), color=(0.5, 0.6, 0.7)))
    # Back wall
    scene.add_entity(Entity(cube, position=(0.0, 0.0, D*2), scale=(W, H, T), color=(0.5, 0.6, 0.7)))
    # Left wall — slightly different tone
    scene.add_entity(Entity(cube, position=(-W*2, 0.0, 0.0), scale=(T, H, D), color=(0.6, 0.5, 0.5)))
    # Right wall
    scene.add_entity(Entity(cube, position=(W*2, 0.0, 0.0), scale=(T, H, D), color=(0.6, 0.5, 0.5)))

    return scene