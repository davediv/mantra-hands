"""
Rasterises the target word into a double-buffered texture.

The next prompt is uploaded to the inactive texture as soon as it is known. The
renderer can keep the prior word under its dissolving burst, then activate the
new one with a pointer/index swap. An early charge therefore never pays text
rasterisation, upload, or mipmap generation on its latency-sensitive frame.
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
        self._active = 0
        self._prepared_index = -1
        self._prepared_word = ""
        self._prepared_aspect = 4.0

        self._image = Image.new("RGBA", (TEX_W, TEX_H), (0, 0, 0, 0))
        self._textures = [
            ctx.texture((TEX_W, TEX_H), 4, dtype="f1"),
            ctx.texture((TEX_W, TEX_H), 4, dtype="f1"),
        ]
        for texture in self._textures:
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
        self.set("READY")

    @property
    def texture(self) -> moderngl.Texture:
        return self._textures[self._active]

    def prepare(self, word: str) -> None:
        """Rasterises and uploads into the texture that is not currently visible."""
        if word == self.word or word == self._prepared_word:
            return

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

        aspect = drawn_width / max(1.0, size * 0.78)

        # Row 0 of the image lands at v=0, which is what WORD_VS assumes when it
        # flips vUv.y — the same convention the browser build's canvas upload had.
        index = self._active ^ 1
        texture = self._textures[index]
        texture.write(self._image.tobytes())
        texture.build_mipmaps()
        self._prepared_index = index
        self._prepared_word = word
        self._prepared_aspect = aspect

    def activate(self, word: str) -> None:
        """Makes a prepared word visible without rasterisation or an upload."""
        if word == self.word:
            return
        if word != self._prepared_word:
            self.prepare(word)
        self._active = self._prepared_index
        self.word = self._prepared_word
        self.aspect = self._prepared_aspect
        self._prepared_index = -1
        self._prepared_word = ""

    def set(self, word: str) -> None:
        """Immediate compatibility path; interaction code uses prepare/activate."""
        self.prepare(word)
        self.activate(word)
