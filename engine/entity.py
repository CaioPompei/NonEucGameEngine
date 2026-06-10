import numpy as np
import pyrr
from engine.mesh import Mesh

class Entity:
    """
        Represents an object in the world
        Transform(position, rotation, scale) and a mesh to render
        Doesn't know anything about shaders or scene - only about itself

        The model matrix is cached on first access and reused until
        `mark_dirty()` is called. All current entities in the scene are
        static, so this avoids hundreds of matrix reconstructions per frame
        (scene.draw runs once per portal recursion level).
    """

    def __init__(self, mesh: Mesh,
                 position=(0.0, 0.0, 0.0),
                 rotation=(0.0, 0.0, 0.0),
                 scale=   (1.0, 1.0, 1.0),
                 color=(1.0, 1.0, 1.0),
                 solid=True,
                 texture=None,
                 texture_scale=(1.0, 1.0)):
        self.mesh = mesh
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        # Whether the player collides with this entity (see game/collision.py).
        self.solid = bool(solid)
        # Optional diffuse texture (engine.texture.Texture) + UV tiling factor.
        self.texture = texture
        self.texture_scale = np.array(texture_scale, dtype=np.float32)
        self._cached_model: np.ndarray | None = None

    def mark_dirty(self):
        """Force recomputation of the model matrix on next access."""
        self._cached_model = None

    def world_aabb(self):
        """
        Axis-aligned bounding box in world space as `(min, max)` numpy arrays.

        Assumes the base mesh is the unit cube (half-extent 0.5 per axis), so
        the world half-size is `abs(scale) * 0.5`. Rotation is ignored — all
        collidable level geometry is axis-aligned. `abs` guards against the
        negative scale factors some level entities use as a modelling trick.
        """
        half = np.abs(self.scale) * 0.5
        return self.position - half, self.position + half

    def get_model_matrix(self):
        """
            Create Model combined from position, rotation and scale
            Order = Scale -> Rotation -> Translation
        """
        if self._cached_model is not None:
            return self._cached_model

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
        self._cached_model = model.astype(np.float32)
        return self._cached_model

    def draw(self, shader):
        """
            Send model matrix to shader and draw the mesh
            Shader must be active.

            Binds the diffuse texture to unit 0 when present (the scene's
            shadow cubemaps live on units 1+). `useTexture` is set every draw
            so a textured entity doesn't leak its texture onto the next.
        """
        shader.set_matrix4("model", self.get_model_matrix())
        shader.set_vec3("objectColor", self.color)

        if self.texture is not None:
            self.texture.bind(unit=0)
            shader.set_int("diffuseTexture", 0)
            shader.set_int("useTexture", 1)
            shader.set_vec2("uvScale", self.texture_scale)
        else:
            shader.set_int("useTexture", 0)

        self.mesh.draw()
