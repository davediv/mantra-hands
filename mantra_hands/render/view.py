"""
Screen shake and the chromatic aberration pulse.

Shake is a change to the view matrix and nothing else. It never re-renders,
never resizes the window, and never moves anything on the CPU — those all cost a
second pass, and this costs nine floats.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import FX


class View:
    def __init__(self) -> None:
        #: Column-major mat3, uploaded as uView.
        self.matrix = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)

        #: 0..1, decays after a pulse.
        self.shake = 0.0
        self.aberration = 0.0
        self.flash = 0.0
        self._t = 0.0
        self._amplitude = 0.0

    def pulse(self, strength: float) -> None:
        """Kicks the camera. ``strength`` is roughly 0..1.5 for a double-power cast."""
        self.shake = min(1.6, self.shake + strength)
        self.aberration = min(1.4, self.aberration + strength)
        self.flash = min(1.0, self.flash + strength * 0.45)
        self._amplitude = self.shake

    def update(self, dt_sec: float) -> None:
        self._t += dt_sec

        decay = math.exp(-dt_sec / (FX.shake_life / 3.0))
        self.shake *= decay
        self.aberration *= math.exp(-dt_sec / (FX.aberration_life / 3.0))
        self.flash *= math.exp(-dt_sec / 0.06)
        if self.shake < 1e-3:
            self.shake = 0.0
        if self.aberration < 1e-4:
            self.aberration = 0.0
        if self.flash < 1e-3:
            self.flash = 0.0

        # Two incommensurate sine pairs per axis: no repeating pattern, no rng,
        # no allocation, and it settles smoothly instead of stopping dead.
        s = self.shake
        k = s * s * 0.09
        tx = k * (math.sin(self._t * 61.0) * 0.6 + math.sin(self._t * 37.3) * 0.4)
        ty = k * (math.sin(self._t * 53.7) * 0.6 + math.sin(self._t * 29.1) * 0.4)
        rot = s * s * 0.022 * math.sin(self._t * 43.1)
        # A small punch-in keeps the shake from ever showing an edge and reads as
        # recoil rather than as a glitch.
        zoom = 1.0 + s * 0.035

        c = math.cos(rot) * zoom
        sn = math.sin(rot) * zoom
        m = self.matrix
        m[0] = c
        m[1] = sn
        m[2] = 0.0
        m[3] = -sn
        m[4] = c
        m[5] = 0.0
        m[6] = tx
        m[7] = ty
        m[8] = 1.0

    def reset(self) -> None:
        self.shake = 0.0
        self.aberration = 0.0
        self.flash = 0.0
        self._amplitude = 0.0
        self.update(0.0)

    @property
    def kick(self) -> float:
        return self._amplitude
