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

    # Remeber: scale() order is: X, Y, Z.
    # Floor — dark gray
    scene.add_entity(Entity(cube, position=(0.0, -H, 0.0), scale=(W*2, T, D*2), color=(0.4, 0.4, 0.4)))
    # Ceiling — light gray
    scene.add_entity(Entity(cube, position=(0.0, H, 0.0), scale=(W*2, T, D*2), color=(0.8, 0.8, 0.8)))
    # Front wall — bluish gray
    scene.add_entity(Entity(cube, position=(0.0, 0.0, -D), scale=(W*2, H*2, T), color=(0.5, 0.6, 0.7)))
    # Back wall
    scene.add_entity(Entity(cube, position=(0.0, 0.0, D), scale=(W*2, H*2, T), color=(0.5, 0.6, 0.7)))
    # Left wall — slightly different tone
    scene.add_entity(Entity(cube, position=(-W, 0.0, 0.0), scale=(T, H*2, D*2), color=(0.6, 0.5, 0.5)))
    # Right wall
    scene.add_entity(Entity(cube, position=(W, 0.0, 0.0), scale=(T, H*2, D*2), color=(0.6, 0.5, 0.5)))
    # Divider wall Left - red
    scene.add_entity(Entity(cube, position=(W/2, 0, -D/2.5), scale=(T, H*2, 12), color=(0.7, 0.3, 0.3)))
    # Divider wall Right - red
    scene.add_entity(Entity(cube, position=(W/2, 0, D/1.05), scale=(T, H*2, D/8), color=(0.7, 0.3, 0.3)))
    # Divider ceiling - red
    scene.add_entity(Entity(cube, position=(W/2, H*0.9, D/2 + 0.4), scale=(T, H*0.3, -D/2 -2), color=(0.7, 0.3, 0.3)))
    # Central Pillar - green
    scene.add_entity(Entity(cube, position=(0, -H/2, 0), scale=(W/4, H, D/4), color=(0.3, 0.7, 0.3)))


    return scene