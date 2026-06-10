import numpy as np
import pyrr
import glfw
import math


class Camera:
    """
        Responsability: Keep position and camera orientation.
        Process input to move the camera
    """

    def __init__(self, position=(0.0,0.0,3.0), speed=5.0, sensitivity=0.5):
        #World position
        self.position = np.array(position, dtype=np.float32)

        # Rotation angles in degrees
        self.yaw = -90.0
        self.pitch = 0.0

        # camera options
        self.speed = speed
        self.sensitivity = sensitivity

        # Direction vectors
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Visual-only eye offset (e.g. head bob). Added when building the view
        # matrix but NOT part of `position`, so physics, collision and portal
        # traversal keep using the true eye position.
        self.view_offset = np.zeros(3, dtype=np.float32)

        # Mouse tracking
        self.last_x = 0.0
        self.last_y = 0.0
        self.first_mouse = True

        self.update_vectors()

    """
        Read keyboard pressed keys and move the camera
        delta_time grants constant speed independent of frame rate
    """
    def process_input(self, window, delta_time):

        speed =  self.speed * delta_time
        if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 2

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.position += self.front * speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.position -= self.front * speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.position -= self.right * speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.position += self.right * speed

        # Q / E to move up and down
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            self.position += self.up * speed
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            self.position -= self.up * speed

        # ESC to close the window
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: 
            glfw.set_window_should_close(window, True)

    def process_mouse_movement(self, x_pos, y_pos):
        if self.first_mouse:
            self.last_x = x_pos
            self.last_y = y_pos
            self.first_mouse = False
            return
        
        dx = (x_pos - self.last_x) * self.sensitivity
        dy = (self.last_y - y_pos) * self.sensitivity  # Invert y-axis

        self.last_x = x_pos
        self.last_y = y_pos

        self.yaw += dx
        self.pitch += dy

        self.pitch = max(-89.0, min(89.0, self.pitch))  # Limit pitch to prevent gimbal lock
        self.update_vectors()

    def set_orientation_from_front(self, front):
        """
        Reconstrói yaw/pitch a partir de um vetor front (após uma travessia
        de portal, p.ex.). Inverte a fórmula usada em `update_vectors`:
            front = (cos(yaw)cos(pitch), sin(pitch), sin(yaw)cos(pitch))
        """
        f = np.asarray(front, dtype=np.float32)
        n = float(np.linalg.norm(f))
        if n < 1e-6:
            return
        f = f / n
        pitch_r = math.asin(max(-1.0, min(1.0, float(f[1]))))
        yaw_r = math.atan2(float(f[2]), float(f[0]))
        self.pitch = math.degrees(pitch_r)
        self.yaw = math.degrees(yaw_r)
        self.update_vectors()

    def get_view_matrix(self):
        """Returns the current view matrix to send to the shader"""
        # Eye includes the visual-only view_offset (head bob); target tracks it
        # so the look direction is unchanged.
        eye = self.position + self.view_offset
        return pyrr.matrix44.create_look_at(
            eye=eye,
            target=eye + self.front,
            up=self.up,
            dtype=np.float32
        )

    def update_vectors(self):
        """
            Recalculate the front, right and up vectors
        """
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)

        # Convert angles to a front vector
        front = np.array([
            math.cos(yaw_r) * math.cos(pitch_r),
            math.sin(pitch_r),
            math.sin(yaw_r) * math.cos(pitch_r),
        ], dtype=np.float32)

        # Grants that the vector weight is 1, so the camera moves at the same speed in all directions
        self.front = front / np.linalg.norm(front) 

        global_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.cross(self.front, global_up)
        self.right /= np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.front)
        self.up /= np.linalg.norm(self.up)