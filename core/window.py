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

        # Create the window; Width, Height, Title, Monitor (none = Windowed)
        self._window = glfw.create_window(Width, Height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise Exception("glfw window can not be created")

        # Make the window's context current
        # In OpenGL, we need to say which window we want to draw in.    
        glfw.make_context_current(self._window)

        glEnable(GL_DEPTH_TEST)

        self.largura = Width
        self.altura = Height

    def window_close(self):
        return glfw.window_should_close(self._window)

    def process_events(self):
        glfw.poll_events()

    def clear(self):
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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

    def set_callback_mouse(self, funcao):
        """Registra a função que será chamada quando o mouse mover."""
        glfw.set_cursor_pos_callback(self._window, funcao)