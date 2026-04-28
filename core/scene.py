from core.entity import Entity
from core.shader import Shader

class Scene:
    """
        All entities in the world, and the shader to render them with
        Responsability: - Manage entities (add/remove)
                        - Manage shader (set active, set uniforms)
        Doesn't know anything about OpenGL - Only delegates to entities and shader
    """

    def __init__(self):
        self.entities: list[Entity] = []

    # Add an entity to the scene and return it
    def add_entity(self, entity: Entity) -> Entity:
        self.entities.append(entity)
        return entity
    
     # Remove an entity from the scene
    def remove_entity(self, entity: Entity):
        self.entities.remove(entity)

    def draw(self, shader: Shader):
        # run across all entities and draw them
        for entity in self.entities:
            entity.draw(shader)