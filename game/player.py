import glfw
import numpy as np

from engine.camera import Camera
from math3d.portal_math import transform_point, transform_direction


class Player:
    """
        Responsability: Player physics (gravity, jump, ground detection),
        mode switching between WALK and FREE_CAM, and portal traversal.
        Wraps a Camera but does not replace it.

        Portal traversal is O(1) per portal per frame:
            - 1 dot product (signed distance to plane)
            - rectangle test only when the sign flipped front → back
            - teleport reuses a pre-baked traversal_transform matrix
    """

    MODE_WALK = 0
    MODE_FREECAM = 1

    def __init__(self, camera: Camera,
                 eye_height=6,
                 ground_y=-5.875,
                 gravity=-20.0,
                 jump_speed=9.0,
                 move_speed=5.0):
        self.camera = camera

        # Floor top in world coordinates; eye sits at ground_y + eye_height
        self.eye_height = eye_height
        self.ground_y = ground_y

        self.gravity = gravity
        self.jump_speed = jump_speed
        self.move_speed = move_speed

        self.mode = Player.MODE_WALK
        self.velocity_y = 0.0
        self.on_ground = False

        # Edge detection for the V toggle
        self._v_was_pressed = False

        # Snap to ground at start
        self.camera.position[1] = self.ground_y + self.eye_height

        # Portal traversal tracking.
        # Position at the end of the previous frame (segment start for the
        # cross-test) and cached signed distance per portal.
        self._last_pos = self.camera.position.copy()
        self._last_sd: dict[int, float] = {}

    def process_input(self, window, delta_time, portals=()):

        # ESC closes
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # V toggles WALK <-> FREE_CAM (edge-triggered)
        v_pressed = glfw.get_key(window, glfw.KEY_V) == glfw.PRESS
        if v_pressed and not self._v_was_pressed:
            self._toggle_mode()
        self._v_was_pressed = v_pressed

        if self.mode == Player.MODE_FREECAM:
            self.camera.process_input(window, delta_time)
        else:
            self._walk_input(window, delta_time)

        # Portal traversal: run after all movement so the segment is the
        # full frame displacement.
        self._handle_portal_traversal(portals)

        # Remember the post-frame position for next frame's segment test.
        self._last_pos = self.camera.position.copy()

    def _toggle_mode(self):
        if self.mode == Player.MODE_WALK:
            self.mode = Player.MODE_FREECAM
            print("[Player] Mode: FREE CAM")
        else:
            self.mode = Player.MODE_WALK
            self.velocity_y = 0.0
            print("[Player] Mode: WALK")

    def _walk_input(self, window, delta_time):
        speed = self.move_speed * delta_time
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 2

        # Horizontal movement only: project camera vectors onto XZ plane
        front = self.camera.front.copy()
        front[1] = 0.0
        front_norm = np.linalg.norm(front)
        if front_norm > 1e-6:
            front /= front_norm

        right = self.camera.right.copy()
        right[1] = 0.0
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right /= right_norm

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.camera.position += front * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.camera.position -= front * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.camera.position -= right * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.camera.position += right * speed

        # Jump (only when grounded)
        if self.on_ground and glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            self.velocity_y = self.jump_speed
            self.on_ground = False

        # Gravity integration
        self.velocity_y += self.gravity * delta_time
        self.camera.position[1] += self.velocity_y * delta_time

        # Ground collision
        floor_eye = self.ground_y + self.eye_height
        if self.camera.position[1] <= floor_eye:
            self.camera.position[1] = floor_eye
            self.velocity_y = 0.0
            self.on_ground = True
        else:
            self.on_ground = False

    # ── Portal traversal ─────────────────────────────────────────────────────

    def _handle_portal_traversal(self, portals):
        if not portals:
            return

        prev_pos = self._last_pos
        curr_pos = self.camera.position

        for portal in portals:
            key = id(portal)
            prev_sd = self._last_sd.get(key)
            T, curr_sd = portal.attempt_traversal(prev_pos, prev_sd, curr_pos)

            if T is not None:
                self._teleport(T)
                # Resync ALL portals against the new position so the
                # destination portal doesn't fire on the very next frame.
                new_pos = self.camera.position
                for p in portals:
                    self._last_sd[id(p)] = p.signed_distance(new_pos)
                return

            self._last_sd[key] = curr_sd

    def _teleport(self, traversal_transform):
        """
        Apply the pre-baked traversal matrix to position and view direction,
        then rebuild yaw/pitch on the camera. Vertical velocity is preserved
        as-is because portals only rotate around Y (so Y is invariant).
        """
        new_pos = transform_point(traversal_transform, self.camera.position)
        new_front = transform_direction(traversal_transform, self.camera.front)

        self.camera.position[:] = new_pos
        self.camera.set_orientation_from_front(new_front)
