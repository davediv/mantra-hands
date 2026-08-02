"""Smoothing and the two ring buffers the recognizers run on."""

from __future__ import annotations

import math

import numpy as np

from .config import TRACKING


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroBank:
    """
    One Euro filter bank — one filter per landmark per axis.

    Stored as flat arrays rather than a list of filter objects so that smoothing
    84 channels costs nothing per frame and stays in cache. All state is
    allocated once in the constructor.

    Casiez, Roussel & Vogel, "1e Filter" (CHI 2012)::

        alpha  = 1 / (1 + tau/dt),  tau = 1 / (2*pi*cutoff)
        cutoff = min_cutoff + beta * |dx_hat|

    The point of the beta term: slow movement gets heavy smoothing (kills
    jitter), fast movement gets light smoothing (kills lag). A plain low-pass
    has to pick one and is wrong half the time.
    """

    __slots__ = ("channels", "_x_hat", "_dx_hat", "_primed", "_a", "_b")

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self._x_hat = np.zeros(channels, dtype=np.float32)
        self._dx_hat = np.zeros(channels, dtype=np.float32)
        self._primed = np.zeros(channels, dtype=np.uint8)
        # Scratch for filter_all, so the whole bank updates without allocating.
        self._a = np.zeros(channels, dtype=np.float32)
        self._b = np.zeros(channels, dtype=np.float32)

    def reset(self, start: int = 0, count: int = -1) -> None:
        """Drops history so a re-detected hand snaps to place instead of easing
        in from stale state."""
        end = self.channels if count < 0 else start + count
        self._primed[start:end] = 0

    def filter_all(self, values: np.ndarray, dt: float, out: np.ndarray) -> np.ndarray:
        """
        The whole bank in one pass — the vectorised equivalent of calling
        :meth:`filter` once per channel, and the only form used on the detection
        path.

        Every intermediate lands in a preallocated scratch array, so smoothing 42
        channels allocates nothing. The one exception is re-acquisition, which
        builds a boolean mask; that happens when a hand comes back into frame, a
        handful of times a session, never per frame.
        """
        if not self._primed.all():
            fresh = self._primed == 0
            self._x_hat[fresh] = values[fresh]
            self._dx_hat[fresh] = 0.0
            self._primed[fresh] = 1
            # Below, dx comes out as 0 for those channels and x_hat stays put at
            # the incoming value — which is exactly what the scalar path returns
            # for a channel it has just primed.

        a_d = _alpha(TRACKING.d_cutoff, dt)

        # dx_hat += a_d * ((value - x_hat)/dt - dx_hat)
        np.subtract(values, self._x_hat, out=self._a)
        self._a /= dt
        np.subtract(self._a, self._dx_hat, out=self._b)
        self._b *= a_d
        self._dx_hat += self._b

        # a_x = alpha(min_cutoff + beta*|dx_hat|, dt), written without the
        # divide-by-tau so it stays two multiplies and a divide.
        np.abs(self._dx_hat, out=self._a)
        self._a *= TRACKING.beta
        self._a += TRACKING.min_cutoff
        self._a *= float(2.0 * math.pi * dt)
        np.add(self._a, 1.0, out=self._b)
        np.divide(self._a, self._b, out=self._a)

        # x_hat += a_x * (value - x_hat)
        np.subtract(values, self._x_hat, out=self._b)
        self._b *= self._a
        self._x_hat += self._b

        out[:] = self._x_hat
        return out

    def filter(self, channel: int, value: float, dt: float) -> float:
        """:param dt: seconds since this channel's previous sample, clamped by the caller."""
        if not self._primed[channel]:
            self._primed[channel] = 1
            self._x_hat[channel] = value
            self._dx_hat[channel] = 0.0
            return value

        prev_x = float(self._x_hat[channel])
        dx = (value - prev_x) / dt

        a_d = _alpha(TRACKING.d_cutoff, dt)
        dx_hat = float(self._dx_hat[channel]) + a_d * (dx - float(self._dx_hat[channel]))
        self._dx_hat[channel] = dx_hat

        cutoff = TRACKING.min_cutoff + TRACKING.beta * abs(dx_hat)
        a_x = _alpha(cutoff, dt)
        x_hat = prev_x + a_x * (value - prev_x)
        self._x_hat[channel] = x_hat
        return float(self._x_hat[channel])


class RollingMean:
    """Fixed-length rolling mean over a preallocated ring."""

    __slots__ = ("size", "_buf", "_head", "_filled", "_sum")

    def __init__(self, size: int) -> None:
        self.size = size
        self._buf = np.zeros(size, dtype=np.float32)
        self._head = 0
        self._filled = 0
        self._sum = 0.0

    def push(self, v: float) -> float:
        if self._filled == self.size:
            self._sum -= float(self._buf[self._head])
        else:
            self._filled += 1
        self._buf[self._head] = v
        self._sum += v
        self._head = (self._head + 1) % self.size
        return self.mean

    @property
    def mean(self) -> float:
        return 0.0 if self._filled == 0 else self._sum / self._filled

    @property
    def count(self) -> int:
        return self._filled

    @property
    def full(self) -> bool:
        return self._filled == self.size

    def clear(self) -> None:
        self._head = 0
        self._filled = 0
        self._sum = 0.0


class TimedRing:
    """
    A time-stamped ring of scalars, used by the fire recognizer to ask "what was
    the hand's bounding-box area ~110 ms ago". Preallocated; push never grows.
    """

    __slots__ = ("size", "_time", "_value", "_head", "_filled")

    def __init__(self, size: int) -> None:
        self.size = size
        self._time = np.zeros(size, dtype=np.float64)
        self._value = np.zeros(size, dtype=np.float32)
        self._head = 0
        self._filled = 0

    def push(self, t: float, v: float) -> None:
        self._time[self._head] = t
        self._value[self._head] = v
        self._head = (self._head + 1) % self.size
        if self._filled < self.size:
            self._filled += 1

    def clear(self) -> None:
        self._head = 0
        self._filled = 0

    @property
    def count(self) -> int:
        return self._filled

    @property
    def oldest_time(self) -> float:
        """Oldest timestamp still in the ring, or -1 when empty."""
        if self._filled == 0:
            return -1.0
        idx = (self._head - self._filled + self.size) % self.size
        return float(self._time[idx])

    def sample_at(self, t: float) -> float:
        """
        The most recent sample at or before ``t``. Walks backwards from the head,
        which is at most ``size`` steps and in practice three or four.
        Returns -1 when the ring does not reach back that far.
        """
        if self._filled == 0:
            return -1.0
        for n in range(1, self._filled + 1):
            idx = (self._head - n + self.size) % self.size
            if self._time[idx] <= t:
                return float(self._value[idx])
        return -1.0
