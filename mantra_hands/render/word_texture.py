"""
Rasterises the target word to a texture.

This runs when the word changes — once every few seconds at most, never inside
the frame loop. By the time a cast fires, the texture is already resident and the
burst uploads nothing.
"""

from __future__ import annotations

import moderngl
from PIL import Image, ImageDraw

from .. import fonts

TEX_W = 1024
TEX_H = 256


class WordTexture:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        #: Width/height of the drawn glyphs, for sizing the quad.
        self.aspect = 4.0
        self.word = ""

        self._image = Image.new("RGBA", (TEX_W, TEX_H), (0, 0, 0, 0))
        self.texture = ctx.texture((TEX_W, TEX_H), 4, dtype="f1")
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.set("READY")

    def set(self, word: str) -> None:
        if word == self.word:
            return
        self.word = word

        text = word.upper()
        self._image.paste((0, 0, 0, 0), (0, 0, TEX_W, TEX_H))
        draw = ImageDraw.Draw(self._image)

        # Fit to the box: measure once at a reference size, then scale. Long words
        # are width limited, short ones height limited.
        ref = 180
        ref_font = fonts.load("display", ref)
        left, _, right, _ = draw.textbbox((0, 0), text, font=ref_font)
        ref_width = max(1, right - left)
        size = int(min(ref * (TEX_W - 40) / ref_width, TEX_H - 40))
        font = fonts.load("display", max(8, size))

        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        drawn_width = min(TEX_W - 20, max(1, right - left))
        draw.text(
            (TEX_W // 2, int(TEX_H / 2 + size * 0.03)),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

        self.aspect = drawn_width / max(1.0, size * 0.78)

        # Row 0 of the image lands at v=0, which is what WORD_VS assumes when it
        # flips vUv.y — the same convention the browser build's canvas upload had.
        self.texture.write(self._image.tobytes())
        self.texture.build_mipmaps()
