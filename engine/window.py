import glfw
from OpenGL.GL import *


class Window:
    """
    Responsability: create and manage the OpenGL window and its context.
    It just opens the window.
    """

    def __init__(self, Width, Height, title):
        if not glfw.init():
            raise Exception("glfw can not be initialized")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.STENCIL_BITS, 8)


        # Create the window; Width, Height, Title, Monitor (none = Windowed)
        self._window = glfw.create_window(Width, Height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise Exception("glfw window can not be created")

        # Make the window's context current
        # In OpenGL, we need to say which window we want to draw in.
        glfw.make_context_current(self._window)

        glEnable(GL_DEPTH_TEST)

        # Track the *framebuffer* size (in pixels), which can differ from the
        # window size on HiDPI displays. The viewport and the aspect ratio must
        # follow this, not the requested Width/Height.
        self.largura, self.altura = glfw.get_framebuffer_size(self._window)
        glViewport(0, 0, self.largura, self.altura)

        # Callbacks notified (with the new width, height) on every resize.
        self._resize_callbacks = []
        glfw.set_framebuffer_size_callback(self._window,
                                           self._on_framebuffer_size)

        # Fullscreen bookkeeping: remember the windowed placement so we can
        # restore it when leaving fullscreen.
        self._is_fullscreen = False
        self._windowed_rect = (0, 0, Width, Height)  # x, y, w, h

    # ── Resize / fullscreen ────────────────────────────────────────────────

    def _on_framebuffer_size(self, _window, width, height):
        """GLFW callback: keep the GL viewport matching the framebuffer and let
        listeners (projection, overlays) adapt to the new size."""
        # A minimized window reports a 0x0 framebuffer. Ignore it to avoid a
        # zero-area viewport and a division-by-zero in the aspect-ratio math.
        if width == 0 or height == 0:
            return
        self.largura = width
        self.altura = height
        glViewport(0, 0, width, height)
        for callback in self._resize_callbacks:
            callback(width, height)

    def on_resize(self, callback):
        """Register `callback(width, height)`, called whenever the framebuffer
        resizes (window drag, maximize, or fullscreen toggle)."""
        self._resize_callbacks.append(callback)

    def get_size(self):
        """Current framebuffer size in pixels (width, height)."""
        return self.largura, self.altura

    def is_fullscreen(self):
        return self._is_fullscreen

    def toggle_fullscreen(self):
        """Switch between a borderless fullscreen on the primary monitor and the
        previous windowed placement. The framebuffer-size callback fires as a
        side effect, so the viewport/projection/overlays update automatically."""
        if self._is_fullscreen:
            x, y, w, h = self._windowed_rect
            glfw.set_window_monitor(self._window, None, x, y, w, h, 0)
            self._is_fullscreen = False
        else:
            # Remember where the window was so we can come back to it.
            x, y = glfw.get_window_pos(self._window)
            w, h = glfw.get_window_size(self._window)
            self._windowed_rect = (x, y, w, h)

            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self._window, monitor, 0, 0,
                mode.size.width, mode.size.height, mode.refresh_rate)
            self._is_fullscreen = True

    def window_close(self):
        return glfw.window_should_close(self._window)

    def process_events(self):
        glfw.poll_events()

    def clear(self, color=(0.1, 0.1, 0.2, 1.0)):
        glClearColor(*color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

    def show(self):
        glfw.swap_buffers(self._window)

    def close(self):
        glfw.terminate()

    # Camera input processing

    def get_handle(self):
        """Retorna o objeto janela do GLFW (necessário para input)."""
        return self._window

    def get_mouse(self):
        """
        Trava o cursor no centro da janela e o esconde.
        Modo padrão para jogos FPS.
        """
        glfw.set_input_mode(self._window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def set_cursor_captured(self, captured: bool):
        """
        Trava+esconde o cursor para controle FPS (captured=True) ou o libera
        para telas de menu/UI (captured=False).
        """
        mode = glfw.CURSOR_DISABLED if captured else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self._window, glfw.CURSOR, mode)

    def set_callback_mouse(self, funcao):
        """Registra a função que será chamada quando o mouse mover."""
        glfw.set_cursor_pos_callback(self._window, funcao)