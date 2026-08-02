"""
Demo mode: ``--demo`` on the command line.

Runs the entire app with no camera and no model — a synthetic video source and a
scripted pair of hands that charge, hold and push on a loop. Everything
downstream of the landmarker is the real thing: the same mirror, the same
filters, the same recognizers, the same renderer.

Two reasons it exists. It is how the app gets shown on a projector in a room
where the laptop's camera is pointed at a wall, and it is how the gesture path
gets verified without a person standing in front of a webcam.
"""

from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraInfo
from .clock import now_ms
from .config import LM
from .landmarks import RawPoint

MCP = (
    (0.38, -0.9),
    (0.05, -1.0),
    (-0.28, -0.95),
    (-0.55, -0.85),
)

DIR = (
    (0.05, -0.999),
    (0.0, -1.0),
    (-0.08, -0.997),
    (-0.15, -0.989),
)

FINGER_BASE = (LM.index_mcp, LM.middle_mcp, LM.ring_mcp, LM.pinky_mcp)

#: One loop: settle, charge, push, recover.
CYCLE_MS = 3400.0


@dataclass
class DemoHand:
    present: bool
    right: bool
    points: List[RawPoint] = field(default_factory=lambda: [RawPoint(0.0, 0.0)] * LM.count)


class DemoHands:
    def __init__(self) -> None:
        self.hands = [DemoHand(False, False), DemoHand(False, True)]
        self._start_ms = -1.0

    def update(self, t_ms: float) -> None:
        if self._start_ms < 0:
            self._start_ms = t_ms
        elapsed = t_ms - self._start_ms
        cycle = int(elapsed // CYCLE_MS)
        t = (elapsed % CYCLE_MS) / CYCLE_MS

        # Every third loop both hands go up, for a double-power cast.
        two_handed = cycle % 3 == 2

        self._pose(0, t, 0.34, two_handed)
        self._pose(1, t, 0.66, True)
        self.hands[0].present = two_handed

    def _pose(self, slot: int, t: float, cx: float, present: bool) -> None:
        hand = self.hands[slot]
        hand.present = present
        if not present:
            return

        # 0.00-0.12 arriving, 0.12-0.62 held, 0.62-0.70 push, 0.70-1.00 recover.
        scale = 0.11
        if 0.62 < t <= 0.7:
            scale = 0.11 + 0.085 * ((t - 0.62) / 0.08)
        elif t > 0.7:
            scale = 0.195 - 0.085 * min(1.0, (t - 0.7) / 0.18)

        # A little life, well under the steadiness threshold.
        drift = math.sin(t * 6.0 + slot) * 0.006
        hand.points = _write(cx + drift, 0.56 + drift * 0.5, scale, hand.right)


def _write(cx: float, cy: float, scale: float, right: bool) -> List[RawPoint]:
    """Writes an open palm facing the camera, in raw (un-mirrored) coordinates."""
    aspect = 4.0 / 3.0
    flip = 1.0 if right else -1.0
    out: List[Optional[RawPoint]] = [None] * LM.count

    def put(index: int, lx: float, ly: float) -> None:
        out[index] = RawPoint(1.0 - (cx + (flip * lx * scale) / aspect), cy + ly * scale)

    put(LM.wrist, 0.0, 0.0)
    put(1, 0.28, -0.16)
    put(LM.thumb_mcp, 0.52, -0.34)
    put(3, 0.68, -0.52)
    put(LM.thumb_tip, 0.8, -0.68)

    for f in range(4):
        mx, my = MCP[f]
        dx, dy = DIR[f]
        base = FINGER_BASE[f]
        put(base, mx, my)
        put(base + 1, mx + dx * 0.42, my + dy * 0.42)
        put(base + 2, mx + dx * 0.68, my + dy * 0.68)
        put(base + 3, mx + dx * 0.95, my + dy * 0.95)

    return [p for p in out if p is not None]


class DemoCamera:
    """
    A stand-in camera, with exactly the interface the real one has: ``read()``
    blocks until the next frame is due and then hands one over, so the detection
    clock is genuinely driven by video frames arriving at 30 Hz rather than by a
    loop spinning as fast as it can.

    The marker in the corner is deliberate. It is drawn at the top *left* of the
    raw frame, so if the mirror is right it appears at the top right on screen.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.info = CameraInfo(width, height, float(fps), "demo mode — no camera")
        self._period = 1.0 / fps
        self._next = time.perf_counter()
        self._frame = np.empty((height, width, 3), dtype=np.uint8)

        # The diagonal ramp the gradient runs along, built once. Painting a frame
        # is then three multiply-adds over it rather than a per-pixel loop.
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        self._ramp = ((xs + ys) * 0.5)[:, :, None]
        self._grid = self._build_grid(width, height)

    @staticmethod
    def _build_grid(width: int, height: int) -> np.ndarray:
        grid = np.zeros((height, width, 1), dtype=np.float32)
        grid[:, ::40] = 0.10
        grid[::40, :] = 0.10
        return grid

    def read(self) -> Optional[np.ndarray]:
        # Pace to the frame rate. A real camera makes the detection thread wait
        # here; so does this, and for the same reason.
        now = time.perf_counter()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self._period, time.perf_counter())

        t = now_ms() / 1000.0
        mid = colorsys.hls_to_rgb(((t * 12.0) % 360.0) / 360.0, 0.22, 0.30)
        # BGR, because that is what the rest of the pipeline expects from cv2.
        c0 = np.array((0x2E, 0x1C, 0x14), dtype=np.float32) / 255.0
        c1 = np.array((mid[2], mid[1], mid[0]), dtype=np.float32)
        c2 = np.array((0x1A, 0x0F, 0x0B), dtype=np.float32) / 255.0

        r = self._ramp
        lower = np.clip(r * 2.0, 0.0, 1.0)
        upper = np.clip(r * 2.0 - 1.0, 0.0, 1.0)
        img = c0 + (c1 - c0) * lower + (c2 - c1) * upper
        img += self._grid * 0.95  # the graticule, so motion is obvious
        np.multiply(np.clip(img, 0.0, 1.0), 255.0, out=img)
        self._frame[:] = img.astype(np.uint8)

        cv2.rectangle(self._frame, (16, 16), (72, 72), (0x8C, 0xD6, 0x3F), -1)
        cv2.putText(
            self._frame, "RAW", (22, 52), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0x1A, 0x0F, 0x0B), 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            self._frame, "demo mode - no camera", (88, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0xC0, 0xC4, 0xC2), 1, cv2.LINE_AA,
        )
        return self._frame

    @property
    def aspect(self) -> float:
        return self.info.width / max(1, self.info.height)

    def release(self) -> None:
        """Nothing to hand back."""
