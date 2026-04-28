import numpy as np
from core.mesh import Mesh
from core.entity import Entity
from core.scene import Scene


def _create_cube_mesh():
    """
        Create a unique cube on origin
        Reuse same mesh for all walls
        Diference between position/scale stay on entity, not on geometry
        Thats is efficient: An only VBO in GPU for everithing
    """
    vertices = np.array([
        -0.5, -0.5,  0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,
         0.5,  0.5,  0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5,
        -0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5,
         0.5,  0.5, -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,
        -0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5, -0.5, -0.5,  0.5, -0.5,  0.5,  0.5,
         0.5,  0.5,  0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5,
         0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5, -0.5,  0.5,
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5,  0.5,
         0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5, -0.5, -0.5,
        -0.5,  0.5, -0.5,  0.5,  0.5,  0.5,  0.5,  0.5, -0.5,
         0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5,  0.5,  0.5,
    ], dtype=np.float32)
    return Mesh(vertices)

def create_room(width=10.0, height=4.0, depth=10.0) -> Scene:
    """
        Create a closed room using 6 faces
        All faces share the same cube mesh, but with diferent position and scale

        The room is centred on the origin, with size 10X4X10.
    """
    scene = Scene()
    cube = _create_cube_mesh()

    W = width 
    H = height
    D = depth
    T = 0.2
    floor_color = (0.35, 0.30, 0.22)
    wall_color = (0.72, 0.77, 0.82)
    ceiling_color = (0.90, 0.90, 0.95)

    # Floor (centered at -H/2 so room is centered on origin)
    scene.add_entity(Entity(cube,
                             position=(0.0, -H*2, 0.0,),
                             scale=(W, T, D),
                             color=floor_color))

    # Ceiling (centered at +H/2)
    scene.add_entity(Entity(cube,
                             position=(0.0, +H*2, 0.0,),
                             scale=(W, T, D),
                             color=ceiling_color))
    
    # Front Wall (centered vertically)
    scene.add_entity(Entity(cube,
                             position=(0.0, 0.0, -D*2),
                             scale=(W, H, T),
                             color=wall_color))
    
    # Back Wall (centered vertically)
    scene.add_entity(Entity(cube,
                             position=(0.0, 0.0, D*2,),
                             scale=(W, H, T),
                             color=wall_color))
    
    # Left Wall (centered vertically)
    scene.add_entity(Entity(cube,
                             position=(-W*2, 0.0, 0.0,),
                             scale=(T, H, D),
                             color=wall_color))
    
    # Right Wall (centered vertically)
    scene.add_entity(Entity(cube,
                            position=(W*2, 0.0, 0.0,),
                            scale=(T, H, D),
                            color=wall_color))
    return scene    