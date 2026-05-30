"""
Light sources for the engine.

The engine renders shadows via shadow mapping. Each light owns its own
shadow map (a cubemap for omnidirectional point lights) which is baked
once when `Scene.bake_shadows()` is called. The cost per frame is just
sampling the cubemap in the fragment shader; the expensive 6-face render
happens only at bake time.

To add new light types (directional, spot), follow the PointLight contract:
    - hold the light's transform/parameters
    - own a shadow map object that knows how to bake itself from a scene
    - expose `bind_to_shader(shader, slot, texture_unit)` to wire uniforms
"""

from __future__ import annotations

import numpy as np

from engine.shadow_map import CubeShadowMap


class PointLight:
    """
    Omnidirectional point light with cubemap shadows.

    `range` controls attenuation (cor ∝ 1 / (1 + d/range)²) and also acts
    as the far plane of the shadow projection — anything farther than
    `range` from the light won't cast nor receive shadows from it.
    """

    def __init__(self,
                 position,
                 color=(1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 range: float = 30.0,
                 shadow_resolution: int = 1024,
                 cast_shadows: bool = True,
                 shadow_bias: float = 0.05):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.intensity = float(intensity)
        self.range = float(range)
        self.shadow_resolution = int(shadow_resolution)
        self.cast_shadows = bool(cast_shadows)
        self.shadow_bias = float(shadow_bias)

        self.shadow_map: CubeShadowMap | None = None
        if self.cast_shadows:
            self.shadow_map = CubeShadowMap(self.shadow_resolution)

    @property
    def far_plane(self) -> float:
        return self.range

    def bake_shadow(self, scene, depth_shader) -> None:
        """Render the scene depth from this light into its cubemap."""
        if self.shadow_map is None:
            return
        self.shadow_map.bake(self.position, self.far_plane,
                             scene, depth_shader)

    def bind_to_shader(self, shader, slot: int, texture_unit: int) -> None:
        """
        Push the light's parameters + bind its shadow cubemap to the
        given texture unit. `slot` is the index in the shader's
        pointLights[] array.
        """
        prefix = f"pointLights[{slot}]"
        shader.set_vec3(f"{prefix}.position", self.position)
        shader.set_vec3(f"{prefix}.color", self.color * self.intensity)
        shader.set_float(f"{prefix}.range", self.range)
        shader.set_float(f"{prefix}.far_plane", self.far_plane)
        shader.set_float(f"{prefix}.bias", self.shadow_bias)
        shader.set_int(f"{prefix}.cast_shadows", 1 if self.cast_shadows else 0)

        if self.shadow_map is not None:
            self.shadow_map.bind(texture_unit)
        shader.set_int(f"shadowMaps[{slot}]", texture_unit)
