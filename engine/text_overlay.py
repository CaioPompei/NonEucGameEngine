import ctypes

import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageDraw, ImageFont

from engine.shader import Shader


_VS = """
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 uv;
out vec2 frag_uv;
void main() {
    frag_uv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

_FS = """
#version 330 core
in vec2 frag_uv;
out vec4 fragColor;
uniform sampler2D tex;
void main() {
    vec4 c = texture(tex, frag_uv);
    if (c.a < 0.01) discard;
    fragColor = c;
}
"""


class TextOverlay:
    """
    Renders a text string as a textured quad in screen space.

    The text can be replaced at runtime via `update_text()`. Each update
    re-rasterizes through PIL and re-uploads the texture, so throttle
    callers (10-15Hz is plenty for HUD-style readouts).
    """

    def __init__(self, text, window_width, window_height,
                 font_size=16, color=(255, 255, 255, 255),
                 padding=10, margin_px=12, corner="top-left",
                 background=(0, 0, 0, 160),
                 offset_px=(0, 0), font_name="arial.ttf"):
        self._shader = Shader(_VS, _FS)
        self._window_size = (window_width, window_height)
        self._color = color
        self._padding = padding
        self._margin_px = margin_px
        self._corner = corner
        self._background = background
        # Extra screen-space shift applied to the quad, in pixels: +x right,
        # +y up. Lets callers place text away from a corner/center anchor
        # (e.g. stacking menu items vertically).
        self._offset_px = offset_px

        self._font = self._load_font(font_name, font_size)

        self._texture = glGenTextures(1)
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._img_size = (0, 0)
        self._text = ""

        self.update_text(text)

    # ── Public API ───────────────────────────────────────────────────────────

    def update_text(self, text: str) -> None:
        """Re-rasterize and re-upload the text texture + quad."""
        if text == self._text:
            return
        self._text = text
        self._rasterize_and_upload(text)
        self._build_quad(self._margin_px, self._corner)

    def draw(self):
        # Overlay must ignore the 3D depth/stencil state
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_STENCIL_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._shader.use()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._texture)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_font(font_name, font_size):
        """Try the requested font, then arial, then PIL's built-in bitmap
        font. On Windows, truetype() looks up bare names in the system fonts
        directory, so "segoeui.ttf" etc. resolve without a full path."""
        for candidate in (font_name, "arial.ttf"):
            if not candidate:
                continue
            try:
                return ImageFont.truetype(candidate, font_size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _rasterize_and_upload(self, text: str) -> None:
        # Multiline-aware: bbox via textbbox handles \n.
        dummy = Image.new("RGBA", (1, 1))
        dd = ImageDraw.Draw(dummy)
        bbox = dd.textbbox((0, 0), text, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        img_w = text_w + self._padding * 2
        img_h = text_h + self._padding * 2

        image = Image.new("RGBA", (img_w, img_h), self._background)
        draw = ImageDraw.Draw(image)
        draw.text((self._padding - bbox[0], self._padding - bbox[1]), text,
                  font=self._font, fill=self._color)

        # Flip vertically so OpenGL UVs (origin bottom-left) match the image.
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        pixels = np.array(image, dtype=np.uint8)

        glBindTexture(GL_TEXTURE_2D, self._texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_w, img_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._img_size = (img_w, img_h)

    def _build_quad(self, margin_px, corner):
        win_w, win_h = self._window_size
        img_w, img_h = self._img_size

        # Quad size in NDC
        w = 2.0 * img_w / win_w
        h = 2.0 * img_h / win_h
        mx = 2.0 * margin_px / win_w
        my = 2.0 * margin_px / win_h

        if corner == "center":
            x0, x1 = -w * 0.5, w * 0.5
            y0, y1 = -h * 0.5, h * 0.5
        elif corner == "top-left":
            x0, y1 = -1.0 + mx, 1.0 - my
            x1, y0 = x0 + w, y1 - h
        elif corner == "top-right":
            x1, y1 = 1.0 - mx, 1.0 - my
            x0, y0 = x1 - w, y1 - h
        elif corner == "bottom-left":
            x0, y0 = -1.0 + mx, -1.0 + my
            x1, y1 = x0 + w, y0 + h
        else:  # bottom-right
            x1, y0 = 1.0 - mx, -1.0 + my
            x0, y1 = x1 - w, y0 + h

        # Apply the screen-space offset (pixels -> NDC; +y is up).
        ox = 2.0 * self._offset_px[0] / win_w
        oy = 2.0 * self._offset_px[1] / win_h
        x0 += ox; x1 += ox
        y0 += oy; y1 += oy

        # Two triangles, with UVs
        vertices = np.array([
            x0, y0, 0.0, 0.0,
            x1, y0, 1.0, 0.0,
            x1, y1, 1.0, 1.0,
            x0, y0, 0.0, 0.0,
            x1, y1, 1.0, 1.0,
            x0, y1, 0.0, 1.0,
        ], dtype=np.float32)

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        stride = 4 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        glBindVertexArray(0)
