import numpy as np
from OpenGL.GL import *

from core.portal import Portal
from core.scene import Scene
from core.shader import Shader


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

    def render(self, view, projection, light_pos, light_color):
        glEnable(GL_STENCIL_TEST)
        glClear(GL_STENCIL_BUFFER_BIT)
        self._render_recursive(view, projection, light_pos, light_color, depth=0)
        glDisable(GL_STENCIL_TEST)

    # ── Recursive core ────────────────────────────────────────────────────────

    def _render_recursive(self, view, projection, light_pos, light_color, depth):
        cam_pos = self._camera_position(view)

        for portal in self.portals:
            if portal.destiny is None:
                continue
            if not portal.is_camera_in_front(cam_pos):
                continue

            self._mark_stencil_inc(portal, view, projection, depth)

            new_view = portal.calculate_virtual_view(view)
            new_proj = portal.calculate_oblique_projection(new_view, projection)

            self._reset_depth_in_stencil(portal, view, projection, depth + 1)
            self._draw_scene(new_view, new_proj, light_pos, light_color, depth + 1)

            if depth + 1 < self.max_depth:
                self._render_recursive(new_view, new_proj,
                                       light_pos, light_color, depth + 1)

            self._mark_stencil_dec(portal, view, projection, depth + 1)
            self._restore_depth_at_portal(portal, view, projection, depth)

    # ── Stencil / depth state helpers ─────────────────────────────────────────

    def _mark_stencil_inc(self, portal, view, projection, depth):
        """Increment stencil where the portal quad is visible at this depth."""
        glStencilFunc(GL_EQUAL, depth, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        self.stencil_shader.use()
        self._draw_quad(portal, view, projection)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def _mark_stencil_dec(self, portal, view, projection, stencil_ref):
        """Undo the matching increment so siblings aren't masked out."""
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_DECR)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_FALSE)
        self.stencil_shader.use()
        self._draw_quad(portal, view, projection)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)
        glDepthMask(GL_TRUE)

    def _reset_depth_in_stencil(self, portal, view, projection, stencil_ref):
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
        self.stencil_shader.use()
        self._draw_quad(portal, view, projection)
        glDepthRange(0.0, 1.0)
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

    def _restore_depth_at_portal(self, portal, view, projection, stencil_ref):
        """
        Write the portal quad's real depth inside stencil==stencil_ref so the
        next sibling portal at this depth sees a consistent depth buffer.
        """
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_ALWAYS)
        self.stencil_shader.use()
        self._draw_quad(portal, view, projection)
        glDepthFunc(GL_LESS)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

    # ── Drawing primitives ────────────────────────────────────────────────────

    def _draw_quad(self, portal, view, projection):
        self.stencil_shader.set_matrix4("view", view)
        self.stencil_shader.set_matrix4("projection", projection)
        self.stencil_shader.set_matrix4("model", portal.get_model_matrix())
        Portal.mesh_quad.draw()

    def _draw_scene(self, view, projection, light_pos, light_color, stencil_ref):
        cam_pos = self._camera_position(view)
        glStencilFunc(GL_EQUAL, stencil_ref, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        self.scene_shader.use()
        self.scene_shader.set_matrix4("view", view)
        self.scene_shader.set_matrix4("projection", projection)
        self.scene_shader.set_vec3("lightPos", light_pos)
        self.scene_shader.set_vec3("lightColor", light_color)
        self.scene_shader.set_vec3("cameraPos", cam_pos)
        self.scene.draw(self.scene_shader)

    @staticmethod
    def _camera_position(view):
        """Extract the world-space camera position from a view matrix."""
        return np.linalg.inv(view)[3, :3]
