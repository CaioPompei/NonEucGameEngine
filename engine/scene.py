from engine.entity import Entity
from engine.light import PointLight
from engine.shader import Shader
from engine.shadow_map import bind_null_cubemap

# Must match MAX_LIGHTS in shaders/phong.frag.
MAX_LIGHTS = 8


class Scene:
    """
        All entities + lights in the world.
        Responsability: - Manage entities (add/remove)
                        - Manage lights and their shadow maps
        Doesn't know anything about OpenGL state — only delegates draws.
    """

    def __init__(self, ambient_color=(0.05, 0.05, 0.05)):
        self.entities: list[Entity] = []
        self.lights: list[PointLight] = []
        self.ambient_color = tuple(ambient_color)
        self._shadows_baked = False

    # ── Entity management ────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        self.entities.append(entity)
        self._shadows_baked = False  # geometry changed → bake invalidated
        return entity

    def remove_entity(self, entity: Entity):
        self.entities.remove(entity)
        self._shadows_baked = False

    # ── Light management ─────────────────────────────────────────────────────

    def add_light(self, light: PointLight) -> PointLight:
        if len(self.lights) >= MAX_LIGHTS:
            raise RuntimeError(
                f"Scene already has {MAX_LIGHTS} lights — bump MAX_LIGHTS in "
                f"engine/scene.py AND shaders/phong.frag to add more.")
        self.lights.append(light)
        self._shadows_baked = False
        return light

    # ── Drawing ──────────────────────────────────────────────────────────────

    def draw(self, shader: Shader):
        """Draw all entities with the scene shader (lighting + shadows)."""
        for entity in self.entities:
            entity.draw(shader)

    def draw_depth(self, depth_shader: Shader):
        """
        Draw all entities with the depth-only shader (shadow bake).
        Skips color/normal uniforms — only the model matrix matters.
        """
        for entity in self.entities:
            depth_shader.set_matrix4("model", entity.get_model_matrix())
            entity.mesh.draw()

    # ── Lighting uniforms (called per frame per shader use) ──────────────────

    def bind_lights(self, shader: Shader, first_texture_unit: int = 1):
        """
        Push all light uniforms + bind their shadow cubemaps to consecutive
        texture units starting at `first_texture_unit`. Unused slots get a
        null cubemap so the sampler array stays valid.
        """
        shader.set_int("numLights", len(self.lights))
        shader.set_vec3("ambientColor", self.ambient_color)
        for i, light in enumerate(self.lights):
            light.bind_to_shader(shader, slot=i,
                                 texture_unit=first_texture_unit + i)
        # Fill unused slots so samplerCube uniforms aren't undefined.
        for i in range(len(self.lights), MAX_LIGHTS):
            bind_null_cubemap(first_texture_unit + i)
            shader.set_int(f"shadowMaps[{i}]", first_texture_unit + i)

    # ── Shadow bake ──────────────────────────────────────────────────────────

    def bake_shadows(self, depth_shader: Shader, force: bool = False):
        """
        Render the depth cubemap for every shadow-casting light. Call once
        after the scene + lights are set up. Re-run if you move entities or
        lights (or pass `force=True` to bypass the cache flag).
        """
        if self._shadows_baked and not force:
            return
        for light in self.lights:
            light.bake_shadow(self, depth_shader)
        self._shadows_baked = True
