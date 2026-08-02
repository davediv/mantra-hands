"""
The performance readout, toggled with ``d`` and shipped in the normal build — a
perf HUD you have to rebuild to get is a perf HUD nobody uses when it matters,
which is on a borrowed laptop in another country.

Rasterised into its own small texture a few times a second rather than every
frame. Measuring the frame is not allowed to cost the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import moderngl
import numpy as np
from PIL import Image, ImageDraw

from .. import fonts
from ..config import COLORS, PERF
from ..perf import Perf
from ..render.gl import Program
from ..render.shaders.scene import PANEL_FS, PANEL_VS

PANEL_W = 340
PANEL_H = 360
SPARK_W = 300
SPARK_H = 54


@dataclass
class HudInput:
    draw_calls: int
    particles: int
    hands: int
    delegate: str
    phase: str
    element: str
    combo: int
    expand_rate: float
    facing: float
    renderer: str
    handedness: str
    scorer: str


def _rgb(hex_color: str):  # noqa: ANN202
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


TEXT = _rgb(COLORS.text)
GREEN = _rgb(COLORS.green)
MAGENTA = _rgb(COLORS.magenta)
YELLOW = _rgb(COLORS.yellow)


def _pad(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}".rjust(6)


class Hud:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.visible = False

        self._prog = Program(ctx, PANEL_VS, PANEL_FS, "hud")
        self._vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self._vao = self._prog.vao([(self._vbo, "2f", "aCorner")], ctx)

        self._image = Image.new("RGBA", (PANEL_W, PANEL_H), (0, 0, 0, 0))
        self._texture = ctx.texture((PANEL_W, PANEL_H), 4, dtype="f1")
        self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture.repeat_x = False
        self._texture.repeat_y = False
        self._last_paint = 0.0
        self._screen = (1280, 720)

    def toggle(self) -> None:
        self.visible = not self.visible

    def resize(self, width: int, height: int) -> None:
        self._screen = (max(1, width), max(1, height))

    def update(self, perf: Perf, data: HudInput, now: float) -> None:
        if not self.visible:
            return
        # Eight refreshes a second. Any faster and the numbers are unreadable
        # anyway, and rasterising is not free.
        if now - self._last_paint < 125:
            return
        self._last_paint = now

        img = self._image
        img.paste((0, 0, 0, 0), (0, 0, PANEL_W, PANEL_H))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, PANEL_W - 1, PANEL_H - 1), radius=10, fill=(6, 9, 16, 214))

        mono = fonts.load("mono", 13)
        y = 14.0

        def line(label: str, value: str, over: bool = False) -> None:
            nonlocal y
            d.text((14, y), label.ljust(9), font=mono, fill=TEXT + (140,))
            d.text((100, y), value, font=mono, fill=(MAGENTA if over else GREEN) + (240,))
            y += 19

        line("fps", _pad(perf.fps, 1) + ("  low power" if perf.low_power else ""))
        line("frame", f"{_pad(perf.average_ms)} / {PERF.budget_ms:.1f}", perf.average_ms > PERF.budget_ms)
        line("render", f"{_pad(perf.work_ms)} / 10.0", perf.work_ms > 10)
        line("detect", f"{_pad(perf.detect_ms)}   {perf.detect_hz:.0f} hz")
        line("draws", str(data.draw_calls).rjust(6))
        line("particles", str(data.particles).rjust(6))
        heap = perf.heap_mb
        line("rss", ("     -" if heap < 0 else _pad(heap, 1)) + " mb")

        y += 8
        for label, value in (
            ("hands", f"{data.hands}  {data.delegate}  {data.handedness}"),
            ("cast", f"{data.phase}  x{data.combo}"),
            ("element", data.element),
            ("expand", f"{data.expand_rate:.2f}/s  facing {data.facing:.2f}"),
            ("speech", data.scorer),
            ("gpu", data.renderer[:26]),
        ):
            d.text((14, y), label.ljust(9), font=mono, fill=TEXT + (140,))
            d.text((100, y), value, font=mono, fill=TEXT + (215,))
            y += 19

        y += 6
        d.text((14, y), "frame time, last 120", font=fonts.load("ui", 11), fill=TEXT + (110,))
        self._sparkline(d, perf, 14, y + 20)

        self._texture.write(img.tobytes())

    def _sparkline(self, d: ImageDraw.ImageDraw, perf: Perf, x: float, y: float) -> None:
        """
        120 frames of wall time. A sparkline shows a stutter that an average
        hides: one 40 ms spike every second reads as "60 fps" in a mean and as a
        visible hitch on a recording.
        """
        # Budget line at 16.6 ms, and the scale tops out at 33 ms — two frames.
        scale = SPARK_H / 33.0
        budget_y = y + SPARK_H - PERF.budget_ms * scale
        d.line((x, budget_y, x + SPARK_W, budget_y), fill=TEXT + (56,), width=1)

        n = PERF.history
        step = SPARK_W / n
        points = []
        for i in range(n):
            v = float(perf.history[(perf.history_head + i) % n])
            points.append((x + i * step, y + SPARK_H - min(SPARK_H, v * scale)))
        d.line(points, fill=(MAGENTA if perf.average_ms > PERF.budget_ms else GREEN) + (255,), width=2)

    def draw(self) -> int:
        if not self.visible:
            return 0
        sw, sh = self._screen
        # Pinned to the top left at a fixed pixel size, so it stays readable on a
        # 4K panel instead of shrinking into the corner.
        half_w = PANEL_W / sw
        half_h = PANEL_H / sh
        x = -1.0 + half_w + 24.0 / sw * 2
        y = 1.0 - half_h - 84.0 / sh * 2
        self._prog.use().vec4("uRect", x, y, half_w, half_h).tex("uLayer", 0, self._texture)
        self._vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        return 1
