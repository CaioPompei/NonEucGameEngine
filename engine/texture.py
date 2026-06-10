"""
2D texture loading for the engine.

A `Texture` owns one GL_TEXTURE_2D. Images are loaded with Pillow, flipped
to match OpenGL's bottom-left UV origin, uploaded with mipmaps, and set to
GL_REPEAT so per-entity `texture_scale` (uvScale) can tile them across large
surfaces. Requires an active GL context (construct after the window exists).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OpenGL.GL import *
from PIL import Image


class Texture:
    def __init__(self, path):
        self.path = str(path)
        image = Image.open(path).convert("RGBA")
        # OpenGL samples with the origin at the bottom-left; PIL rows go
        # top-down, so flip vertically to keep textures upright.
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        pixels = np.frombuffer(image.tobytes(), dtype=np.uint8)
        width, height = image.size

        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

    def bind(self, unit: int = 0):
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.id)


class TextureRegistry:
    """Caches textures by resolved path so the same file is uploaded once."""

    def __init__(self):
        self._cache: dict[str, Texture] = {}

    def get(self, path) -> Texture:
        key = str(Path(path).resolve())
        tex = self._cache.get(key)
        if tex is None:
            tex = Texture(path)
            self._cache[key] = tex
        return tex
