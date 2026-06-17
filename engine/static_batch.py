"""
Static geometry batching.

The scene is made of many small static entities (mostly textured cubes), each
drawn with its own model matrix, uniforms and texture bind. Under PyOpenGL the
per-call overhead dominates, and the scene is redrawn once per portal recursion
level on top of the main pass — so the draw-call count is the bottleneck, not
the GPU.

A `StaticBatch` bakes every static entity into the GPU once: positions and
normals are pre-transformed to world space, UVs are pre-multiplied by each
entity's `texture_scale`, and entities sharing the same (texture, color) are
merged into a single mesh. Drawing the whole scene then costs one draw call per
distinct (texture, color) group — for the current levels that's ~2 instead of
~55. Because the model matrix is baked in, every group draws with model =
identity and uvScale = (1, 1).

Only meshes carrying both normals and UVs are merged (the vertex layout the
scene shader expects); anything else falls back to per-entity drawing.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import *

from engine.mesh import Mesh

_IDENTITY = np.identity(4, dtype=np.float32)


class _Group:
    __slots__ = ("mesh", "texture", "color", "use_texture")

    def __init__(self, mesh, texture, color, use_texture):
        self.mesh = mesh
        self.texture = texture
        self.color = color
        self.use_texture = use_texture


class StaticBatch:
    """Merged, pre-transformed static scene geometry. Build once per level."""

    def __init__(self, entities):
        self._groups: list[_Group] = []
        self._unbatched = []  # meshes without normals+uv: drawn per-entity
        self._build(entities)

    # ── Build ────────────────────────────────────────────────────────────────

    def _build(self, entities):
        # Bucket entities by what makes them share a draw call: the texture
        # object identity and the tint color (objectColor). useTexture follows
        # from whether a texture is present.
        buckets: dict[tuple, list] = {}
        for e in entities:
            mesh = e.mesh
            if not (mesh.have_normals and mesh.have_uv):
                self._unbatched.append(e)
                continue
            tex_key = id(e.texture) if e.texture is not None else None
            color_key = tuple(np.round(e.color, 5))
            buckets.setdefault((tex_key, color_key), []).append(e)

        for (tex_key, _), group_entities in buckets.items():
            # Flatten to a 1-D float buffer: Mesh derives the vertex count from
            # len(vertices) // floats_per_vertex, so it must be the raw float
            # stream (not a 2-D (n_vertices, 8) array).
            merged = np.concatenate(
                [self._bake_entity(e) for e in group_entities]).reshape(-1)
            mesh = Mesh(merged, have_normals=True, have_uv=True)
            sample = group_entities[0]
            self._groups.append(_Group(
                mesh=mesh,
                texture=sample.texture,
                color=sample.color,
                use_texture=sample.texture is not None,
            ))

    @staticmethod
    def _bake_entity(entity) -> np.ndarray:
        """Return the entity's vertices transformed into world space, with
        UVs scaled by its texture_scale. Shape (n_vertices, 8)."""
        verts = entity.mesh.vertices.reshape(-1, entity.mesh.floats_per_vertex)
        pos = verts[:, 0:3]
        normal = verts[:, 3:6]
        uv = verts[:, 6:8]

        model = entity.get_model_matrix()  # pyrr row-vector: p_world = p @ model

        # Positions: homogeneous row-vector multiply.
        ones = np.ones((pos.shape[0], 1), dtype=np.float32)
        pos_world = (np.hstack([pos, ones]) @ model)[:, :3]

        # Normals: row-vector form of the inverse-transpose. The scene shader
        # computes n_world_col = inverse(model_3x3) @ n_col, so for row vectors
        # n_world = n @ inverse(model_3x3).T. Re-normalize (non-uniform scale).
        inv3 = np.linalg.inv(model[:3, :3])
        normal_world = normal @ inv3.T
        lengths = np.linalg.norm(normal_world, axis=1, keepdims=True)
        normal_world = normal_world / np.maximum(lengths, 1e-8)

        uv_scaled = uv * entity.texture_scale

        out = np.empty((pos.shape[0], 8), dtype=np.float32)
        out[:, 0:3] = pos_world
        out[:, 3:6] = normal_world
        out[:, 6:8] = uv_scaled
        return out

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, shader):
        """Draw the whole static scene with the lighting shader (must be in
        use, with view/projection/lights already bound)."""
        shader.set_matrix4("model", _IDENTITY)
        for g in self._groups:
            shader.set_vec3("objectColor", g.color)
            if g.use_texture:
                g.texture.bind(unit=0)
                shader.set_int("diffuseTexture", 0)
                shader.set_int("useTexture", 1)
                shader.set_vec2("uvScale", (1.0, 1.0))
            else:
                shader.set_int("useTexture", 0)
            g.mesh.draw()

        for e in self._unbatched:
            e.draw(shader)

    def draw_depth(self, depth_shader):
        """Draw the static scene for a shadow/depth pass (positions only)."""
        depth_shader.set_matrix4("model", _IDENTITY)
        for g in self._groups:
            g.mesh.draw()

        for e in self._unbatched:
            depth_shader.set_matrix4("model", e.get_model_matrix())
            e.mesh.draw()
