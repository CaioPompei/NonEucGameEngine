from pathlib import Path

from OpenGL.GL import *


class Shader:
    """
    Responsability: compile GLSL shaders and send data (uniforms) to the GPU.
    It knows nothing about windows or geometry.
    """

    def __init__(self, vertex_src, fragment_src):
        self._programa = self._create_program(vertex_src, fragment_src)
        # glGetUniformLocation is a driver-side string lookup; calling it
        # once per uniform per draw adds up fast under recursive portal
        # rendering. Cache locations on first use.
        self._uniform_locations: dict[str, int] = {}

    @classmethod
    def from_files(cls, vertex_path, fragment_path) -> "Shader":
        """Load and compile a shader pair from GLSL files on disk."""
        vert_src = Path(vertex_path).read_text(encoding="utf-8")
        frag_src = Path(fragment_path).read_text(encoding="utf-8")
        return cls(vert_src, frag_src)

    # ── Use ──────────────────────────────────────────────────────────────

    def use(self):
        """Ativa este shader para os próximos draw calls."""
        glUseProgram(self._programa)

    def _location(self, name):
        loc = self._uniform_locations.get(name)
        if loc is None:
            loc = glGetUniformLocation(self._programa, name)
            self._uniform_locations[name] = loc
        return loc

    def set_matrix4(self, name, matrix):
        """Envia uma matriz 4x4 para um uniform do shader."""
        glUniformMatrix4fv(self._location(name), 1, GL_FALSE, matrix)

    def set_float(self, name, value):
        glUniform1f(self._location(name), value)

    def set_vec2(self, name, vector):
        glUniform2f(self._location(name), *vector)

    def set_vec3(self, name, vector):
        glUniform3f(self._location(name), *vector)

    def set_int(self, name, value):
        glUniform1i(self._location(name), int(value))

    # ── Internos ─────────────────────────────────────────────────────────

    def _compile(self, source, shader_type):
        shader = glCreateShader(shader_type)    # Create a shader object
        glShaderSource(shader, source)          # Send the source code to the shader object 
        glCompileShader(shader)                 # Compile the shader on GPU

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            erro = glGetShaderInfoLog(shader).decode()
            raise Exception(f"Shader compilation failed: {erro}")
        return shader

    """
    A "program" in OpenGL = vertex shader + fragment shader linked together
    The GPU uses this program to render objects on the screen
    """
    def _create_program(self, vert_src, frag_src):
        vs = self._compile(vert_src, GL_VERTEX_SHADER)
        fs = self._compile(frag_src, GL_FRAGMENT_SHADER)

        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)

        if not glGetProgramiv(prog, GL_LINK_STATUS):
            erro = glGetProgramInfoLog(prog).decode()
            raise Exception(f"Program linking failed: {erro}")

        glDeleteShader(vs)
        glDeleteShader(fs)
        return prog