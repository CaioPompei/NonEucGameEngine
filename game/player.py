import glfw
import numpy as np

from engine.camera import Camera
from math3d.portal_math import transform_point, transform_direction


class Player:
    """
    Player physics and movement.

    WALK mode is a proper character controller:
      - acceleration / friction based horizontal movement (no instant
        stop/start — gives weight without feeling sluggish),
      - real gravity integrated every frame; the player lands on top of
        whatever solid surface is below (floor, a box, …) via the
        CollisionWorld, not a hardcoded ground plane,
      - AABB collision with wall-slide (blocked axis is cancelled, the
        others keep moving),
      - coyote time + jump buffering so jumps feel responsive.

    FREECAM mode (toggle `V`) is a noclip 6-DOF fly camera for debugging.

    The camera position is the player's *eye*. The collision box hangs below
    it: feet at `eye_y - EYE_HEIGHT`, top a little above the eye. Keeping the
    eye as the single source of truth means portal traversal and the view
    matrix need no special-casing.
    """

    MODE_WALK = 0
    MODE_FREECAM = 1

    # ── Body dimensions (world units; the scene is modelled large) ───────────
    RADIUS = 0.2         # half-width on X/Z
    EYE_HEIGHT = 1.0      # feet → eye
    HEAD_CLEARANCE = 0.3  # eye → top of head
    HEIGHT = EYE_HEIGHT + HEAD_CLEARANCE

    # ── Head bob (visual-only camera sway while walking) ─────────────────────
    BOB_FREQUENCY = 4.5   # phase advance per world unit travelled
    BOB_AMP_VERTICAL = 0.08
    BOB_AMP_SIDE = 0.0
    BOB_BLEND_RATE = 9.0  # how fast the bob fades in/out (per second)
    BOB_MIN_SPEED = 0.5   # below this we treat the player as standing still

    def __init__(self, camera: Camera,
                 gravity=-20.0,
                 jump_speed=5.0,
                 walk_speed=3.0,
                 sprint_multiplier=1.8,
                 ground_accel=70.0,
                 air_accel=14.0,
                 friction=55.0):
        self.camera = camera

        self.gravity = gravity
        self.jump_speed = jump_speed
        self.walk_speed = walk_speed
        self.sprint_multiplier = sprint_multiplier
        self.ground_accel = ground_accel
        self.air_accel = air_accel
        self.friction = friction

        self.mode = Player.MODE_WALK

        # Horizontal velocity lives in XZ of this vector (Y stays 0); vertical
        # velocity is tracked separately so gravity/jump are easy to reason about.
        self.velocity = np.zeros(3, dtype=np.float32)
        self.velocity_y = 0.0
        self.on_ground = False

        # Responsiveness helpers (seconds).
        self._coyote = 0.0          # time left to still jump after leaving ground
        self._jump_buffer = 0.0     # time left for a pre-pressed jump to fire
        self.COYOTE_TIME = 0.12
        self.JUMP_BUFFER_TIME = 0.12

        # Head-bob state.
        self._bob_phase = 0.0       # advances with distance walked
        self._bob_blend = 0.0       # 0 = no bob, 1 = full bob (eased)

        # Edge detection.
        self._v_was_pressed = False
        self._space_was_pressed = False

        # Portal traversal tracking.
        self._last_pos = self.camera.position.copy()
        self._last_sd: dict[int, float] = {}

    @property
    def half_extents(self):
        return np.array([Player.RADIUS, Player.HEIGHT * 0.5, Player.RADIUS],
                        dtype=np.float32)

    def reset(self, position):
        """
        Place the player at `position` (eye position) and clear all motion
        state. Called on spawn and whenever a new level is loaded so velocity,
        grounding, and portal-traversal caches don't leak across levels.
        """
        self.camera.position[:] = position
        self.velocity[:] = 0.0
        self.velocity_y = 0.0
        self.on_ground = False
        self._coyote = 0.0
        self._jump_buffer = 0.0
        self._space_was_pressed = False
        self._bob_phase = 0.0
        self._bob_blend = 0.0
        self.camera.view_offset[:] = 0.0
        self._last_pos = self.camera.position.copy()
        self._last_sd.clear()

    def process_input(self, window, delta_time, world=None, portals=()):
        # ESC closes
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        # V toggles WALK <-> FREECAM (edge-triggered)
        v_pressed = glfw.get_key(window, glfw.KEY_V) == glfw.PRESS
        if v_pressed and not self._v_was_pressed:
            self._toggle_mode()
        self._v_was_pressed = v_pressed

        if self.mode == Player.MODE_FREECAM:
            self.camera.process_input(window, delta_time)
            self.camera.view_offset[:] = 0.0  # no head bob in debug fly-cam
        else:
            self._walk(window, delta_time, world)
            self._update_head_bob(delta_time)

        # Portal traversal runs after all movement so the segment is the full
        # frame displacement.
        self._handle_portal_traversal(portals)
        self._last_pos = self.camera.position.copy()

    def _toggle_mode(self):
        if self.mode == Player.MODE_WALK:
            self.mode = Player.MODE_FREECAM
            print("[Player] Mode: FREE CAM")
        else:
            self.mode = Player.MODE_WALK
            self.velocity[:] = 0.0
            self.velocity_y = 0.0
            print("[Player] Mode: WALK")

    # ── WALK mode ────────────────────────────────────────────────────────────

    def _walk(self, window, delta_time, world):
        dt = delta_time
        self._coyote = max(0.0, self._coyote - dt)
        self._jump_buffer = max(0.0, self._jump_buffer - dt)

        wish_dir = self._wish_direction(window)
        sprinting = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        target_speed = self.walk_speed * (self.sprint_multiplier if sprinting else 1.0)

        self._apply_horizontal_acceleration(wish_dir, target_speed, dt)

        # Buffer a jump press; fire it if grounded (or within coyote window).
        space = glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
        if space and not self._space_was_pressed:
            self._jump_buffer = self.JUMP_BUFFER_TIME
        self._space_was_pressed = space

        if self._jump_buffer > 0.0 and (self.on_ground or self._coyote > 0.0):
            self.velocity_y = self.jump_speed
            self.on_ground = False
            self._coyote = 0.0
            self._jump_buffer = 0.0

        # Gravity.
        self.velocity_y += self.gravity * dt

        # Assemble this frame's displacement and resolve against the world.
        displacement = self.velocity * dt
        displacement[1] = self.velocity_y * dt

        if world is None:
            self.camera.position += displacement
            self.on_ground = False
            return

        center = self._box_center()
        new_center, flags = world.slide(center, self.half_extents, displacement)
        self._set_box_center(new_center)

        # Cancel velocity components that hit something so we don't accumulate
        # speed into a wall / the floor.
        if flags["blocked_x"]:
            self.velocity[0] = 0.0
        if flags["blocked_z"]:
            self.velocity[2] = 0.0
        if flags["ceiling"] and self.velocity_y > 0.0:
            self.velocity_y = 0.0
        if flags["on_ground"]:
            self.velocity_y = 0.0

        was_grounded = self.on_ground
        self.on_ground = flags["on_ground"]
        if was_grounded and not self.on_ground:
            self._coyote = self.COYOTE_TIME  # just walked off a ledge

    def _update_head_bob(self, dt):
        """
        Advance a sine-based bob from the distance walked and write it into
        `camera.view_offset` (visual only). The amplitude eases in/out via
        `_bob_blend` so starting, stopping, jumping and landing stay smooth.
        Vertical bob is the classic up/down; a slower side sway along the
        camera's right axis adds the walk roll.
        """
        horizontal = self.velocity.copy()
        horizontal[1] = 0.0
        speed = float(np.linalg.norm(horizontal))
        walking = self.on_ground and speed > self.BOB_MIN_SPEED

        target = 1.0 if walking else 0.0
        self._bob_blend += (target - self._bob_blend) * min(1.0,
                                                            self.BOB_BLEND_RATE * dt)

        if walking:
            # Cap the cadence: sprinting lengthens the stride (bigger amplitude
            # below) rather than speeding the bob up, which otherwise looks
            # frantic. Cadence tops out just above walk speed.
            cadence_speed = min(speed, self.walk_speed * 1.15)
            self._bob_phase += cadence_speed * dt * self.BOB_FREQUENCY

        amp_scale = min(speed / self.walk_speed, 1.4) if self.walk_speed > 0 else 1.0
        vert = np.sin(self._bob_phase) * self.BOB_AMP_VERTICAL * amp_scale
        side = np.cos(self._bob_phase * 0.5) * self.BOB_AMP_SIDE * amp_scale

        right = self.camera.right.copy()
        right[1] = 0.0
        n = np.linalg.norm(right)
        if n > 1e-6:
            right /= n

        b = self._bob_blend
        self.camera.view_offset[0] = right[0] * side * b
        self.camera.view_offset[1] = vert * b
        self.camera.view_offset[2] = right[2] * side * b

    def _wish_direction(self, window):
        """Desired horizontal move direction (unit vector in XZ, or zeros)."""
        front = self.camera.front.copy()
        front[1] = 0.0
        n = np.linalg.norm(front)
        if n > 1e-6:
            front /= n

        right = self.camera.right.copy()
        right[1] = 0.0
        n = np.linalg.norm(right)
        if n > 1e-6:
            right /= n

        wish = np.zeros(3, dtype=np.float32)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            wish += front
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            wish -= front
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            wish += right
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            wish -= right

        n = np.linalg.norm(wish)
        if n > 1e-6:
            wish /= n
        return wish

    def _apply_horizontal_acceleration(self, wish_dir, target_speed, dt):
        vel = self.velocity.copy()
        vel[1] = 0.0
        speed = np.linalg.norm(vel)

        moving = np.linalg.norm(wish_dir) > 1e-6
        if moving:
            accel = self.ground_accel if self.on_ground else self.air_accel
            vel += wish_dir * accel * dt
            # Clamp to the target speed (only scales down, so sprint→walk
            # bleeds off smoothly via friction rather than snapping).
            new_speed = np.linalg.norm(vel)
            if new_speed > target_speed:
                vel *= target_speed / new_speed
        elif self.on_ground and speed > 0.0:
            # Friction: decelerate toward a full stop.
            drop = self.friction * dt
            factor = max(0.0, speed - drop) / speed
            vel *= factor

        vel[1] = 0.0
        self.velocity = vel

    # ── Collision box <-> eye conversion ─────────────────────────────────────

    def _box_center(self):
        eye = self.camera.position
        feet_y = eye[1] - Player.EYE_HEIGHT
        return np.array([eye[0], feet_y + Player.HEIGHT * 0.5, eye[2]],
                        dtype=np.float32)

    def _set_box_center(self, center):
        feet_y = center[1] - Player.HEIGHT * 0.5
        self.camera.position[0] = center[0]
        self.camera.position[1] = feet_y + Player.EYE_HEIGHT
        self.camera.position[2] = center[2]

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
                new_pos = self.camera.position
                for p in portals:
                    self._last_sd[id(p)] = p.signed_distance(new_pos)
                return

            self._last_sd[key] = curr_sd

    def _teleport(self, traversal_transform):
        """
        Apply the pre-baked traversal matrix to position, view direction, and
        the horizontal velocity, then rebuild yaw/pitch. Vertical velocity is
        preserved as-is because portals only rotate around Y.
        """
        new_pos = transform_point(traversal_transform, self.camera.position)
        new_front = transform_direction(traversal_transform, self.camera.front)
        new_vel = transform_direction(traversal_transform, self.velocity)

        self.camera.position[:] = new_pos
        self.velocity[:] = new_vel
        self.velocity[1] = 0.0
        self.camera.set_orientation_from_front(new_front)
