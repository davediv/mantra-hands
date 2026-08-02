"""
Every piece of chrome the app draws, behind one object.

The browser build put all of this in the DOM and let the compositor deal with it.
There is no DOM here, so it is rasterised with PIL into one RGBA texture and
drawn as a single quad after the composite pass — and, critically, only
re-rasterised when something it shows actually changes. A word lasts seconds; a
toast lasts three; the counters change once a cast. Rasterising a full-window
image every frame would cost more than the entire scene does.

The one thing that does change every frame is the charge bar, and that is why it
is *not* in the texture: it is two solid quads whose geometry is a uniform, which
is the same trick the browser build used when it moved the bar with a transform
rather than a width.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import moderngl
import numpy as np
from PIL import Image, ImageDraw

from .. import fonts
from ..config import COLORS, ELEMENT_LABEL
from ..content.deck import WordEntry
from ..render.gl import Program
from ..render.shaders.scene import OVERLAY_FS, OVERLAY_VS, RECT_FS, RECT_VS
from .progress import Settings

#: Failure microcopy. Every one of these is about the spell, never about the
#: speaker — the joke is that the runes are picky, not that the accent is wrong.
#: Nobody at a language camp needs a computer being clever about their English.
FIZZLE_LINES = (
    "The runes did not catch that one.",
    "Spell fizzled. Coba lagi!",
    "Almost — the circle cracked.",
    "Nope. The magic is fussy today.",
    "Missed it. One more time.",
)

ELEMENT_TINT = ((255, 71, 18), (15, 184, 255), (255, 224, 107))

#: The overlay is rasterised at most this wide and scaled up by the GPU. At 4K
#: the difference is invisible and it keeps a word change to a few milliseconds.
MAX_OVERLAY_W = 1920

KEY_HELP = (
    ("d", "performance readout"),
    ("s", "these settings"),
    ("m", "mirror the camera"),
    ("p", "low power mode"),
    ("[ ]", "effect intensity"),
    ("r", "record a clip"),
    ("q", "quit"),
)


def _rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


BASE = _rgb(COLORS.base)
TEXT = _rgb(COLORS.text)
GREEN = _rgb(COLORS.green)
YELLOW = _rgb(COLORS.yellow)
MAGENTA = _rgb(COLORS.magenta)


class Shell:
    def __init__(self, ctx: moderngl.Context, settings: Settings) -> None:
        self.ctx = ctx
        self.settings = settings
        self.on_settings_change: Optional[Callable[[Settings], None]] = None

        self._prog = Program(ctx, OVERLAY_VS, OVERLAY_FS, "overlay")
        self._rect = Program(ctx, RECT_VS, RECT_FS, "rect")
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
        self._vbo = ctx.buffer(quad)
        self._vao = self._prog.vao([(self._vbo, "2f", "aCorner")], ctx)
        self._rect_vao = self._rect.vao([(self._vbo, "2f", "aCorner")], ctx)

        self._w = 1
        self._h = 1
        self._image: Optional[Image.Image] = None
        self._texture: Optional[moderngl.Texture] = None
        self._dirty = True

        # Everything the overlay shows. Any assignment that changes one of these
        # sets _dirty; nothing else causes a re-rasterise.
        self._word: Optional[WordEntry] = None
        self._hint = "Open palm · say the word · push"
        self._combo = 0
        self._streak = 0
        self._best = 0
        self._empty = False
        self._heard = ""
        self._heard_ok = False
        self._heard_until = 0.0
        self._toast = ""
        self._toast_warn = False
        self._toast_until = 0.0
        self._fizzled = False
        self._fizzle_index = 0
        self._loading: Optional[str] = None
        self._fatal: Optional[str] = None
        self._explainer: Optional[str] = None
        self._settings_open = False
        self._deck_info = ""
        self._mic_note = ""
        self._countdown: Optional[str] = None
        self._recording = False

        # Drawn as quads, not rasterised: these change every frame.
        self._charge = 0.0
        self._armed = False

    # ----------------------------------------------------------------- size ---

    def resize(self, width: int, height: int) -> None:
        w = min(MAX_OVERLAY_W, max(1, width))
        h = max(1, round(height * w / max(1, width)))
        if w == self._w and h == self._h:
            return
        self._w, self._h = w, h
        if self._texture is not None:
            self._texture.release()
        self._image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self._texture = self.ctx.texture((w, h), 4, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.repeat_x = False
        self._texture.repeat_y = False
        self._dirty = True

    @property
    def _scale(self) -> float:
        return self._h / 720.0

    # -------------------------------------------------------------- screens ---

    def show_explainer(self, note: str) -> None:
        self._explainer = note
        self._dirty = True

    def hide_explainer(self) -> None:
        self._explainer = None
        self._dirty = True

    @property
    def explainer_open(self) -> bool:
        return self._explainer is not None

    def show_loading(self, msg: str) -> None:
        self._loading = msg
        self._dirty = True

    def hide_loading(self) -> None:
        self._loading = None
        self._dirty = True

    def show_fatal(self, msg: str) -> None:
        self._loading = None
        self._explainer = None
        self._fatal = msg
        self._dirty = True

    def refresh(self) -> None:
        """Forces one re-rasterise, for a change the shell cannot see itself —
        a settings value it reads from the app rather than owns."""
        self._dirty = True

    def toggle_settings(self, open_: bool) -> None:
        if open_ == self._settings_open:
            return
        self._settings_open = open_
        self._dirty = True

    @property
    def settings_open(self) -> bool:
        return self._settings_open

    def set_mic_note(self, text: str) -> None:
        self._mic_note = text
        self._dirty = self._dirty or self._settings_open

    def set_deck_info(self, text: str) -> None:
        self._deck_info = text
        self._dirty = self._dirty or self._settings_open

    def set_mic_enabled(self, on: bool) -> None:
        self.settings.mic = on
        self._dirty = self._dirty or self._settings_open

    def set_low_power_toggle(self, on: bool) -> None:
        self.settings.low_power = on
        self._dirty = self._dirty or self._settings_open

    def set_recording(self, on: bool) -> None:
        if on != self._recording:
            self._recording = on
            self._dirty = True

    def set_countdown(self, text: Optional[str]) -> None:
        if text != self._countdown:
            self._countdown = text
            self._dirty = True

    # --------------------------------------------------------------- prompt ---

    def set_word(self, entry: WordEntry) -> None:
        self._word = entry
        self._fizzled = False
        self._dirty = True

    def set_hint(self, text: str) -> None:
        if text == self._hint:
            return
        self._hint = text
        self._dirty = True

    def set_charge(self, charge01: float, armed: bool) -> None:
        """Called every frame. Two quads — never a re-rasterise."""
        self._charge = max(0.0, min(1.0, charge01))
        self._armed = armed

    def set_empty_state(self, show: bool) -> None:
        if show == self._empty:
            return
        self._empty = show
        self._dirty = True

    def set_counts(self, combo: int, streak: int, best: int) -> None:
        if (combo, streak, best) == (self._combo, self._streak, self._best):
            return
        self._combo, self._streak, self._best = combo, streak, best
        self._dirty = True

    def show_heard(self, text: str, correct: bool, now_ms: float) -> None:
        if not text:
            return
        self._heard = text
        self._heard_ok = correct
        self._heard_until = now_ms + 1800.0
        self._dirty = True

    def clear_heard(self) -> None:
        if self._heard:
            self._heard = ""
            self._dirty = True

    def fizzle(self) -> str:
        """The word stays on screen for a retry; only the styling says it missed."""
        self._fizzled = True
        self._dirty = True
        line = FIZZLE_LINES[self._fizzle_index % len(FIZZLE_LINES)]
        self._fizzle_index += 1
        return line

    def toast(self, text: str, now_ms: float, warn: bool = False) -> None:
        self._toast = text
        self._toast_warn = warn
        self._toast_until = now_ms + 3200.0
        self._dirty = True

    # ----------------------------------------------------------------- draw ---

    def draw(self, now_ms: float) -> int:
        """
        :returns: draw calls issued, so the HUD's count stays honest.
        """
        if self._toast and now_ms > self._toast_until:
            self._toast = ""
            self._dirty = True
        if self._heard and now_ms > self._heard_until:
            self._heard = ""
            self._dirty = True

        if self._dirty:
            self._rasterise()
            self._texture.write(self._image.tobytes())
            self._dirty = False

        calls = 0
        self._prog.use().tex("uLayer", 0, self._texture)
        self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        calls += 1

        # The bar is drawn after the texture, so anything that dims the screen
        # has to suppress it explicitly — a full-screen sheet with a charge bar
        # glowing through it reads as a rendering bug.
        covered = (
            self._loading is not None
            or self._explainer is not None
            or self._fatal is not None
            or self._settings_open
        )
        if self._word is not None and not covered:
            calls += self._draw_charge()
        return calls

    def _draw_charge(self) -> int:
        # Screen fractions, matching where _rasterise draws the track.
        cx, cy, w, h = 0.5, 0.962, 0.42, 0.007
        left = cx - w * 0.5
        fill_w = w * self._charge
        color = (1.0, 0.773, 0.192, 0.95) if self._armed else (0.247, 0.839, 0.549, 0.9)

        self._rect.use().vec4("uRect", cx * 2 - 1, 1 - cy * 2, w, h).vec4(
            "uColor", 1.0, 1.0, 1.0, 0.14
        )
        self._rect_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        if fill_w <= 0.0005:
            return 1

        fcx = left + fill_w * 0.5
        self._rect.use().vec4("uRect", fcx * 2 - 1, 1 - cy * 2, fill_w, h).vec4("uColor", *color)
        self._rect_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        return 2

    # ------------------------------------------------------------ rasterise ---

    def _rasterise(self) -> None:
        img = self._image
        img.paste((0, 0, 0, 0), (0, 0, self._w, self._h))
        d = ImageDraw.Draw(img)
        s = self._scale

        self._draw_topbar(d, s)
        if self._empty:
            self._draw_empty_state(d, s)
        if self._heard:
            self._draw_heard(d, s)
        if self._word is not None:
            self._draw_prompt(d, s)
        if self._toast:
            self._draw_toast(d, s)
        if self._countdown:
            self._draw_countdown(d, s)
        if self._settings_open:
            self._draw_settings(d, s)
        if self._explainer is not None:
            self._draw_explainer(d, s)
        if self._loading is not None:
            self._draw_sheet(d, s, "", self._loading)
        if self._fatal is not None:
            self._draw_sheet(d, s, "Stuck", self._fatal)

    def _draw_topbar(self, d: ImageDraw.ImageDraw, s: float) -> None:
        label_font = fonts.load("ui", max(9, int(11 * s)))
        value_font = fonts.load("ui-bold", max(14, int(21 * s)))
        x = 26 * s
        for label, value, hot in (
            ("COMBO", f"x{self._combo}", self._combo >= 3),
            ("STREAK", str(self._streak), self._streak >= 3),
            ("BEST", str(self._best), False),
        ):
            d.text((x, 18 * s), label, font=label_font, fill=TEXT + (110,))
            d.text((x, 32 * s), value, font=value_font, fill=(YELLOW if hot else TEXT) + (235,))
            x += 82 * s

        hint = "● REC" if self._recording else "d · s"
        d.text(
            (self._w - 26 * s, 24 * s),
            hint,
            font=fonts.load("ui-bold", max(11, int(14 * s))),
            fill=(MAGENTA if self._recording else TEXT) + (150,),
            anchor="ra",
        )

    def _draw_empty_state(self, d: ImageDraw.ImageDraw, s: float) -> None:
        cx, cy = self._w * 0.5, self._h * 0.42
        # A palm, drawn rather than set in an emoji font that may not be there.
        pw, ph = 26 * s, 34 * s
        d.rounded_rectangle(
            (cx - pw * 0.5, cy - ph * 0.2, cx + pw * 0.5, cy + ph * 0.6),
            radius=8 * s,
            fill=TEXT + (40,),
        )
        for i in range(4):
            fx = cx - pw * 0.5 + (i + 0.5) * (pw / 4)
            d.rounded_rectangle(
                (fx - 3.4 * s, cy - ph * (0.62 - abs(i - 1.5) * 0.07), fx + 3.4 * s, cy),
                radius=3 * s,
                fill=TEXT + (40,),
            )
        d.text(
            (cx, cy + ph * 0.95),
            "Show me your hand",
            font=fonts.load("ui-bold", max(15, int(23 * s))),
            fill=TEXT + (215,),
            anchor="ma",
        )
        d.text(
            (cx, cy + ph * 0.95 + 30 * s),
            "Open palm, facing the camera",
            font=fonts.load("ui", max(11, int(15 * s))),
            fill=TEXT + (130,),
            anchor="ma",
        )

    def _draw_heard(self, d: ImageDraw.ImageDraw, s: float) -> None:
        d.text(
            (self._w * 0.5, self._h * 0.70),
            f'"{self._heard}"',
            font=fonts.load("ui", max(13, int(19 * s))),
            fill=(GREEN if self._heard_ok else MAGENTA) + (230,),
            anchor="ma",
        )

    def _draw_prompt(self, d: ImageDraw.ImageDraw, s: float) -> None:
        entry = self._word
        cx = self._w * 0.5
        top = self._h - 176 * s

        # Element tag: the part of speech and what it casts, which is the lesson.
        tag = f"{entry.pos.upper()} · {ELEMENT_LABEL[int(entry.element)]}"
        tag_font = fonts.load("ui-bold", max(10, int(13 * s)))
        tw = d.textlength(tag, font=tag_font)
        pad = 11 * s
        tint = ELEMENT_TINT[int(entry.element)]
        d.rounded_rectangle(
            (cx - tw * 0.5 - pad, top, cx + tw * 0.5 + pad, top + 24 * s),
            radius=12 * s,
            fill=tint + (46,),
            outline=tint + (150,),
            width=max(1, int(s)),
        )
        d.text((cx, top + 12 * s), tag, font=tag_font, fill=tint + (255,), anchor="mm")

        word_font = fonts.load("display", max(24, int(64 * s)))
        d.text(
            (cx, top + 34 * s),
            entry.word,
            font=word_font,
            fill=(MAGENTA if self._fizzled else TEXT) + (245,),
            anchor="ma",
        )
        if entry.gloss:
            d.text(
                (cx, top + 104 * s),
                entry.gloss,
                font=fonts.load("ui", max(11, int(15 * s))),
                fill=TEXT + (140,),
                anchor="ma",
            )
        d.text(
            (cx, top + 130 * s),
            self._hint,
            font=fonts.load("ui-bold", max(10, int(13 * s))),
            fill=TEXT + (170,),
            anchor="ma",
        )

    def _draw_toast(self, d: ImageDraw.ImageDraw, s: float) -> None:
        font = fonts.load("ui-bold", max(11, int(15 * s)))
        cx, cy = self._w * 0.5, self._h * 0.845
        tw = d.textlength(self._toast, font=font)
        pad = 16 * s
        d.rounded_rectangle(
            (cx - tw * 0.5 - pad, cy - 16 * s, cx + tw * 0.5 + pad, cy + 16 * s),
            radius=16 * s,
            fill=BASE + (225,),
            outline=(YELLOW if self._toast_warn else GREEN) + (170,),
            width=max(1, int(s)),
        )
        d.text((cx, cy), self._toast, font=font, fill=TEXT + (245,), anchor="mm")

    def _draw_countdown(self, d: ImageDraw.ImageDraw, s: float) -> None:
        d.text(
            (self._w * 0.5, self._h * 0.44),
            self._countdown,
            font=fonts.load("display", max(60, int(190 * s))),
            fill=TEXT + (235,),
            anchor="mm",
        )

    def _panel(self, d: ImageDraw.ImageDraw, s: float, height: float) -> Tuple[float, float, float]:
        d.rectangle((0, 0, self._w, self._h), fill=BASE + (232,))
        w = min(self._w * 0.86, 560 * s)
        x0 = (self._w - w) * 0.5
        y0 = (self._h - height) * 0.5
        d.rounded_rectangle(
            (x0, y0, x0 + w, y0 + height),
            radius=18 * s,
            fill=(18, 24, 40, 245),
            outline=TEXT + (28,),
            width=max(1, int(s)),
        )
        return x0, y0, w

    def _draw_sheet(self, d: ImageDraw.ImageDraw, s: float, title: str, body: str) -> None:
        x0, y0, w = self._panel(d, s, 190 * s)
        cx = x0 + w * 0.5
        y = y0 + 46 * s
        if title:
            d.text(
                (cx, y),
                title,
                font=fonts.load("display", max(22, int(40 * s))),
                fill=TEXT + (245,),
                anchor="ma",
            )
            y += 56 * s
        self._wrapped(d, body, cx, y, w - 72 * s, fonts.load("ui", max(12, int(16 * s))), TEXT + (205,))

    def _draw_explainer(self, d: ImageDraw.ImageDraw, s: float) -> None:
        x0, y0, w = self._panel(d, s, 400 * s)
        cx = x0 + w * 0.5
        d.text(
            (cx, y0 + 40 * s),
            "Mantra Hands",
            font=fonts.load("display", max(26, int(52 * s))),
            fill=TEXT + (250,),
            anchor="ma",
        )
        y = y0 + 108 * s
        y = self._wrapped(
            d,
            "Charge a spell with your palm. Say the English word out loud. "
            "Push at the camera and watch it fire.",
            cx,
            y,
            w - 72 * s,
            fonts.load("ui", max(12, int(17 * s))),
            TEXT + (215,),
        )
        y += 18 * s
        body = fonts.load("ui", max(11, int(14 * s)))
        for line in (
            "Camera — to see your hand. The video never leaves this device.",
            "Microphone — optional. Skip it and the app still plays, offline.",
        ):
            y = self._wrapped(d, f"·  {line}", x0 + 36 * s, y, w - 72 * s, body, TEXT + (165,), "la")
        y += 26 * s
        d.text(
            (cx, y),
            "Enter — start with the microphone",
            font=fonts.load("ui-bold", max(12, int(16 * s))),
            fill=GREEN + (245,),
            anchor="ma",
        )
        d.text(
            (cx, y + 30 * s),
            "N — play without a mic",
            font=fonts.load("ui-bold", max(12, int(16 * s))),
            fill=TEXT + (190,),
            anchor="ma",
        )
        if self._explainer:
            self._wrapped(
                d, self._explainer, cx, y + 66 * s, w - 72 * s,
                fonts.load("ui", max(10, int(13 * s))), TEXT + (130,),
            )

    def _draw_settings(self, d: ImageDraw.ImageDraw, s: float) -> None:
        x0, y0, w = self._panel(d, s, 430 * s)
        d.text(
            (x0 + 34 * s, y0 + 28 * s),
            "Settings",
            font=fonts.load("display", max(20, int(34 * s))),
            fill=TEXT + (245,),
            anchor="la",
        )
        row = fonts.load("ui", max(12, int(16 * s)))
        key = fonts.load("mono", max(11, int(14 * s)))
        y = y0 + 86 * s
        for k, label, state in (
            ("m", "Mirror the camera", self.settings.mirror),
            ("p", "Low power mode", self.settings.low_power),
            ("[ ]", f"Effect intensity  {self.settings.intensity}%", None),
        ):
            d.text((x0 + 34 * s, y), k, font=key, fill=YELLOW + (220,), anchor="la")
            d.text((x0 + 86 * s, y), label, font=row, fill=TEXT + (215,), anchor="la")
            if state is not None:
                d.text(
                    (x0 + w - 34 * s, y),
                    "on" if state else "off",
                    font=row,
                    fill=(GREEN if state else TEXT) + (200 if state else 110,),
                    anchor="ra",
                )
            y += 32 * s

        y += 8 * s
        note = fonts.load("ui", max(10, int(13 * s)))
        y = self._wrapped(d, self._mic_note, x0 + 34 * s, y, w - 68 * s, note, TEXT + (130,), "la")
        y += 10 * s
        y = self._wrapped(d, self._deck_info, x0 + 34 * s, y, w - 68 * s, note, TEXT + (130,), "la")
        y += 18 * s
        for k, label in KEY_HELP:
            d.text((x0 + 34 * s, y), k, font=key, fill=TEXT + (150,), anchor="la")
            d.text((x0 + 86 * s, y), label, font=note, fill=TEXT + (120,), anchor="la")
            y += 22 * s

    def _wrapped(
        self,
        d: ImageDraw.ImageDraw,
        text: str,
        x: float,
        y: float,
        width: float,
        font,  # noqa: ANN001
        fill,  # noqa: ANN001
        anchor: str = "ma",
    ) -> float:
        if not text:
            return y
        words = text.split()
        lines: List[str] = []
        line = ""
        for word in words:
            probe = f"{line} {word}".strip()
            if d.textlength(probe, font=font) > width and line:
                lines.append(line)
                line = word
            else:
                line = probe
        if line:
            lines.append(line)
        step = (font.size if hasattr(font, "size") else 14) * 1.45
        for text_line in lines:
            d.text((x, y), text_line, font=font, fill=fill, anchor=anchor)
            y += step
        return y
