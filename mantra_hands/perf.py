"""
Frame timing and the low-power governor.

The frame budget is 16.6 ms and it is measured, not assumed. Two numbers are
tracked separately because they fail differently:

    wall  — time between frames. What the user actually experiences.
    work  — time spent inside our own render callback. What we can fix.

A wall time over budget with low work means something else is eating the
machine: another app, thermal throttling, the encoder during a recording. A high
work time is ours.
"""

from __future__ import annotations

import os
import resource
from typing import Callable, Optional

import numpy as np

from .config import PERF
from .filters import RollingMean


class Perf:
    def __init__(self) -> None:
        #: Wall-clock delta of the last frame, ms.
        self.frame_ms = 16.6
        #: Time inside the render callback, ms.
        self.render_ms = 0.0
        #: Cost of the last detection, ms. Reported by the tracker, off our clock.
        self.detect_ms = 0.0
        self.detect_hz = 0.0

        self.low_power = False
        #: True when the governor turned it on rather than the user.
        self.auto_low_power = False
        self.on_low_power: Optional[Callable[[], None]] = None

        #: Ring of the last 120 frame times, for the sparkline.
        self.history = np.zeros(PERF.history, dtype=np.float32)
        self.history_head = 0

        self._wall = RollingMean(PERF.window)
        self._work = RollingMean(PERF.window)
        self._last_frame = -1.0
        self._over_budget_ms = 0.0
        self._frame_start = 0.0

    def begin_frame(self, now: float) -> float:
        dt = 16.6 if self._last_frame < 0 else min(200.0, now - self._last_frame)
        self._last_frame = now
        self.frame_ms = dt
        self._frame_start = now

        self._wall.push(dt)
        self.history[self.history_head] = dt
        self.history_head = (self.history_head + 1) % PERF.history
        return dt

    def end_frame(self, now: float) -> None:
        self.render_ms = now - self._frame_start
        self._work.push(self.render_ms)
        self._governor(self.frame_ms)

    def _governor(self, dt: float) -> None:
        """
        Two seconds of rolling average over 20 ms and the app drops to low power
        on its own — and says so, because a silent quality drop reads as the app
        getting worse for no reason.

        It latches. Climbing back out on its own would mean oscillating for
        anyone sitting right at the boundary, which looks far worse than staying
        down.
        """
        if self.low_power or not self._wall.full:
            return
        if self._wall.mean > PERF.degrade_ms:
            self._over_budget_ms += dt
            if self._over_budget_ms >= PERF.degrade_hold_ms:
                self.low_power = True
                self.auto_low_power = True
                if self.on_low_power:
                    self.on_low_power()
        else:
            self._over_budget_ms = 0.0

    def set_low_power(self, on: bool) -> None:
        self.low_power = on
        self.auto_low_power = False
        self._over_budget_ms = 0.0

    @property
    def fps(self) -> float:
        m = self._wall.mean
        return 1000.0 / m if m > 0 else 0.0

    @property
    def average_ms(self) -> float:
        return self._wall.mean

    @property
    def work_ms(self) -> float:
        return self._work.mean

    @property
    def heap_mb(self) -> float:
        """
        Resident set size, which is the closest thing to the browser build's
        ``performance.memory``. Reported in MB; -1 if the platform will not say.
        """
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (ValueError, OSError):
            return -1.0
        # Linux reports kilobytes, macOS and the BSDs report bytes.
        return rss / 1048576.0 if os.uname().sysname == "Darwin" else rss / 1024.0
