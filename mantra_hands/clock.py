"""
One clock, everywhere.

The browser build times everything off ``performance.now()``: a monotonic
millisecond counter that no clock adjustment can walk backwards. This is that,
and every ``t_ms`` in the codebase comes from here.
"""

from __future__ import annotations

import time

_ORIGIN = time.perf_counter()


def now_ms() -> float:
    """Milliseconds since the process started, monotonic."""
    return (time.perf_counter() - _ORIGIN) * 1000.0
