"""
Light sources for the engine.

Three types, all binding into the shared `lights[]` array in the phong
shader (the `.type` field selects the lighting model per light):

    PointLight        omnidirectional, position + range falloff. The only
                      type with shadows (a baked depth cubemap).
    DirectionalLight  parallel "sun"; a direction, no position/falloff.
    SpotLight         position + direction + cone (inner/outer angles).

Shadows currently exist for point lights only — directional/spot would need
2D shadow maps (ortho / perspective), which the cubemap infrastructure here
doesn't cover. They light the scene but don't cast shadows yet.

Light contract (so Scene can treat them uniformly):
    - `intensity`, `color`             (read by portal light transport)
    - `shadow_map`                     (None when the light casts no shadow)
    - `bake_shadow(scene, depth_shader)`
    - `bind_to_shader(shader, slot, texture_unit)`
"""

from __future__ import annotations

import math

import numpy as np

from engine.shadow_map import CubeShadowMap, bind_null_cubemap

# Must match the LIGHT_* defines in shaders/phong.frag.
TYPE_POINT = 0
TYPE_DIRECTIONAL = 1
TYPE_SPOT = 2


def _normalize(vec, fallback=(0.0, 0.0, -1.0)) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (v / n).astype(np.float32)


def _bind_struct(shader, slot, *, type, position=(0.0, 0.0, 0.0),
                 direction=(0.0, 0.0, -1.0), color=(0.0, 0.0, 0.0),
                 range=1.0, far_plane=1.0, bias=0.0, cast_shadows=0,
                 inner_cos=1.0, outer_cos=-1.0) -> None:
    """Set every field of `lights[slot]` so no stale value leaks across the
    different light types sharing the array."""
    p = f"lights[{slot}]"
    shader.set_int(f"{p}.type", type)
    shader.set_vec3(f"{p}.position", position)
    shader.set_vec3(f"{p}.direction", direction)
    shader.set_vec3(f"{p}.color", color)
    shader.set_float(f"{p}.range", range)
    shader.set_float(f"{p}.far_plane", far_plane)
    shader.set_float(f"{p}.bias", bias)
    shader.set_int(f"{p}.cast_shadows", cast_shadows)
    shader.set_float(f"{p}.inner_cos", inner_cos)
    shader.set_float(f"{p}.outer_cos", outer_cos)


def _bind_shadow_slot(shader, slot, texture_unit, shadow_map) -> None:
    """Bind a cubemap (or the null one) to keep the samplerCube array valid."""
    if shadow_map is not None:
        shadow_map.bind(texture_unit)
    else:
        bind_null_cubemap(texture_unit)
    shader.set_int(f"shadowMaps[{slot}]", texture_unit)


class PointLight:
    """
    Omnidirectional point light with cubemap shadows.

    `range` controls attenuation (cor ∝ (1 - d/range)²) and also acts as the
    far plane of the shadow projection — anything farther than `range` from
    the light won't cast nor receive shadows from it.
    """

    type = TYPE_POINT

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
        if self.shadow_map is None:
            return
        self.shadow_map.bake(self.position, self.far_plane, scene, depth_shader)

    def bind_to_shader(self, shader, slot: int, texture_unit: int) -> None:
        _bind_struct(shader, slot,
                     type=TYPE_POINT,
                     position=self.position,
                     color=self.color * self.intensity,
                     range=self.range,
                     far_plane=self.far_plane,
                     bias=self.shadow_bias,
                     cast_shadows=1 if self.cast_shadows else 0)
        _bind_shadow_slot(shader, slot, texture_unit, self.shadow_map)


class DirectionalLight:
    """Parallel light (a sun): a travel direction, no position or falloff."""

    type = TYPE_DIRECTIONAL

    def __init__(self,
                 direction,
                 color=(1.0, 1.0, 1.0),
                 intensity: float = 1.0):
        self.direction = _normalize(direction)
        self.color = np.array(color, dtype=np.float32)
        self.intensity = float(intensity)
        self.shadow_map = None  # no shadows for directional lights (yet)

    def bake_shadow(self, scene, depth_shader) -> None:
        return  # no shadow map to bake

    def bind_to_shader(self, shader, slot: int, texture_unit: int) -> None:
        _bind_struct(shader, slot,
                     type=TYPE_DIRECTIONAL,
                     direction=self.direction,
                     color=self.color * self.intensity)
        _bind_shadow_slot(shader, slot, texture_unit, None)


class SpotLight:
    """
    Cone light: position + direction + inner/outer cone angles (degrees,
    measured from the axis). Full intensity within `inner_angle`, fading to
    zero at `outer_angle`. Uses the same distance falloff as a point light.
    No shadows yet.
    """

    type = TYPE_SPOT

    def __init__(self,
                 position,
                 direction,
                 color=(1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 range: float = 30.0,
                 inner_angle: float = 20.0,
                 outer_angle: float = 30.0):
        self.position = np.array(position, dtype=np.float32)
        self.direction = _normalize(direction)
        self.color = np.array(color, dtype=np.float32)
        self.intensity = float(intensity)
        self.range = float(range)
        # Keep inner <= outer so the cone fades the right way.
        self.inner_angle = float(min(inner_angle, outer_angle))
        self.outer_angle = float(max(inner_angle, outer_angle))
        self.shadow_map = None

    def bake_shadow(self, scene, depth_shader) -> None:
        return

    def bind_to_shader(self, shader, slot: int, texture_unit: int) -> None:
        _bind_struct(shader, slot,
                     type=TYPE_SPOT,
                     position=self.position,
                     direction=self.direction,
                     color=self.color * self.intensity,
                     range=self.range,
                     inner_cos=math.cos(math.radians(self.inner_angle)),
                     outer_cos=math.cos(math.radians(self.outer_angle)))
        _bind_shadow_slot(shader, slot, texture_unit, None)
