import numpy as np
from engine.entity import Entity
from engine.primitives import create_cube_mesh
from engine.scene import Scene


def create_room(width=30.0, height=6.0, depth=30.0) -> Scene:
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
    scene.add_entity(Entity(cube, position=(W/2.5, 0, D/5), scale=(T, H*2, 12), color=(0.7, 0.3, 0.3)))
    # Divider wall Right - red
    scene.add_entity(Entity(cube, position=(W/2.5, 0, D/1.05), scale=(T, H*2, D/8), color=(0.7, 0.3, 0.3)))
    # Divider ceiling - red
    scene.add_entity(Entity(cube, position=(W/2.5, H*0.9, D/1.5), scale=(T, H*0.3, -D/2 -2), color=(0.7, 0.3, 0.3)))
    # Divider Wall X - red
    scene.add_entity(Entity(cube, position=(W/1.371, 0, 0), scale=(D/1.5, H*2, T), color=(0.7, 0.3, 0.3)))

    # Central Pillar - green
    scene.add_entity(Entity(cube, position=(0, -H/2, 0), scale=(1, H*3, 1), color=(0.3, 0.7, 0.3)))


    return scene