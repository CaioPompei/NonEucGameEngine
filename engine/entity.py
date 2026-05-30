import numpy as np
import pyrr
from engine.mesh import Mesh

class Entity:
    """
        Represents an object in the world
        Transform(position, rotation, scale) and a mesh to render
        Doesn't know anything about shaders or scene - only about itself
    """

    def __init__(self, mesh: Mesh, 
                 position=(0.0, 0.0, 0.0),
                 rotation=(0.0, 0.0, 0.0),
                 scale=   (1.0, 1.0, 1.0),
                 color=(1.0, 1.0, 1.0)):
        self.mesh = mesh
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)

    def get_model_matrix(self):
        """
            Create Model combined from position, rotation and scale
            Order = Scale -> Rotation -> Translation
        """
        # Translation Matrix - Move the object to its position
        T = pyrr.matrix44.create_from_translation(self.position, dtype=np.float32)

        # Rotation Matrices - One per axis, in degrees
        Rx = pyrr.matrix44.create_from_x_rotation(
            np.radians(self.rotation[0]), dtype=np.float32)
        Ry = pyrr.matrix44.create_from_y_rotation(
            np.radians(self.rotation[1]), dtype=np.float32)
        Rz = pyrr.matrix44.create_from_z_rotation(
            np.radians(self.rotation[2]), dtype=np.float32)
        
        # Scale Matrix - Scale the object by its scale factors
        S = pyrr.matrix44.create_from_scale(self.scale, dtype=np.float32)

        # With the current matrix convention used in this project, the transform chain
        # must be composed left-to-right as Scale -> Rotation -> Translation.
        model = S @ Rx @ Ry @ Rz @ T
        return model.astype(np.float32)
    
    def draw(self, shader):
        """
            Send model matrix to shader and draw the mesh
            Shader must be active.
        """
        shader.set_matrix4("model", self.get_model_matrix())
        shader.set_vec3("objectColor", self.color)
        self.mesh.draw()
            