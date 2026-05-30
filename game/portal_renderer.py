import numpy as np
from OpenGL.GL import *

from engine.scene import Scene
from engine.shader import Shader
from game.portal import Portal


class PortalRenderer:
    """
    Renders a scene through a set of paired portals, supporting recursive
    nested views (one portal seen inside another, up to `max_depth`).

    Algorithm (Lengyel-style stencil portals):
      - Stencil starts at 0 everywhere.
      - For each portal facing the current view:
          1. Increment stencil from `depth` to `depth+1` where the portal
             quad passes the depth test (i.e. is actually visible).
          2. Reset depth to far inside that new stencil region.
          3. Render the scene with the virtual view/projection through the
             portal, masked to stencil == depth+1.
          4. Recurse into depth+1 so portals visible in the virtual scene
             also open up.
          5. Decrement stencil back to `depth` so sibling portals can mark
             their own area without interference.
          6. Restore the depth of the portal quad at this level so siblings
             see a consistent depth buffer.

    The scene at depth 0 must be drawn by the caller before invoking
    `render()`, since this class only paints the inside of portals.
    """

    def __init__(self,
                 portals: list,
                 scene: Scene,
                 scene_shader: Shader,
                 stencil_shader: Shader,
                 max_depth: int = 3):
        self.portals = portals
        self.scene = scene
        self.scene_shader = scene_shader
        self.stencil_shader = stencil_shader
        self.max_depth = max_depth

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, view, projection, cam_pos):
        glEnable(GL_STENCIL_TEST)
        glClear(GL_STENCIL_BUFFER_BIT)
        self._render_recursive(view, projection, depth=0, cam_pos=cam_pos)
        glDisable(GL_STENCIL_TEST)

    # ── Recursive core ────────────────────────────────────────────────────────

    def _render_recursive(self, view, projection, depth, cam_pos):
        for portal in self.portals:
            if portal.destiny is None:
                continue
            if not portal.is_camera_in_front(cam_pos):
                continue

            # The four stencil/depth passes below all draw the same quad
            # with the same model/view/projection. Bind the stencil shader
            # and its uniforms once before the first pair of passes.
            self._bind_quad_uniforms(portal, view, projection)
            self._mark_stencil_inc(depth)
            self._reset_depth_in_stencil(depth + 1)

            new_view = portal.calculate_virtual_view(view)
            new_proj = portal.calculate_oblique_projection(new_view, projection)
            # One matrix inverse per recursion level, instead of several.
            new_cam_pos = np.linalg.inv(new_view)[3, :3]

            self._draw_scene(new_view, new_proj, depth + 1, new_cam_pos)

            if depth + 1 < self.max_depth:
                self._render_recursive(new_view, new_proj,
                                       depth + 1, new_cam_pos)

            # _draw_scene and the recursion switched programs and uniforms;
            # rebind the stencil shader and quad uniforms before the
            # decrement/restore pair.
            self._bind_quad_uniforms(portal, view, projection)
            self._mark_stencil_dec(depth + 1)
            self._restore_depth_at_portal(depth)

    # ── Stencil / depth state helpers ─────────────────────────────────────────

    def _bind_quad_uniforms(self, portal, view, projection):
        self.stencil_shader.use()
        self.stencil_shader.set_matrix4("view", view)
        self.stencil_shader.set_matrix4("projection", projection)
        self.stencil_shader.set_matrix4("model", portal.get_model_matrix())

    def _mark_stencil_inc(self, depth):
        """Increment stencil where the portal quad is visible at this depth."""
        glStencilFunc(GL_EQUAL, depth, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        Portal.mesh_quad.draw()
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def _mark_stencil_dec(self, stencil_ref):
        """Undo the matching increment so siblings aren't masked out."""
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_DECR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        Portal.mesh_quad.draw()
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def _reset_depth_in_stencil(self, stencil_ref):
        """
        Force depth=1.0 inside stencil==stencil_ref. glClear ignores the
        stencil test, so we draw the portal quad with glDepthRange(1,1)
        and GL_ALWAYS to "clear" only the masked region.
        """
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_ALWAYS)
        glDepthRange(1.0, 1.0)
        Portal.mesh_quad.draw()
        glDepthRange(0.0, 1.0)
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

    def _restore_depth_at_portal(self, stencil_ref):
        """
        Write the portal quad's real depth inside stencil==stencil_ref so the
        next sibling portal at this depth sees a consistent depth buffer.
        """
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_ALWAYS)
        Portal.mesh_quad.draw()
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

    # ── Drawing primitives ────────────────────────────────────────────────────

    def _draw_scene(self, view, projection, stencil_ref, cam_pos):
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        self.scene_shader.use()
        self.scene_shader.set_matrix4("view", view)
        self.scene_shader.set_matrix4("projection", projection)
        self.scene_shader.set_vec3("cameraPos", cam_pos)
        # Lights + shadow cubemaps are bound once per frame in main.py and
        # remain on their texture units; no re-bind needed per recursion.
        self.scene.draw(self.scene_shader)
