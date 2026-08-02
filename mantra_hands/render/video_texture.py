"""
The camera frame as a GL texture.

The browser build handed the ``<video>`` element straight to ``texImage2D``. Here
the frame arrives as a numpy array that OpenCV decoded and the detection thread
already converted to RGB — the same conversion MediaPipe needs, done once and
used twice.

The texture's storage is allocated on the first upload and reused after that.
Reallocating every frame is worth roughly half the upload cost, and there is no
readback anywhere in the path.
"""

from __future__ import annotations

from typing import Optional

import moderngl
import numpy as np


class VideoTexture:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.width = 0
        self.height = 0
        # A 1x1 stand-in so the first frames — the ones drawn while the camera is
        # still warming up — have something to sample rather than a null binding.
        self.texture = ctx.texture((1, 1), 3, data=b"\x0b\x0f\x1a", dtype="f1")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False

    def upload(self, rgb: Optional[np.ndarray]) -> None:
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        if w == 0 or h == 0:
            return

        if w != self.width or h != self.height:
            self.texture.release()
            self.texture = self.ctx.texture((w, h), 3, dtype="f1")
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.texture.repeat_x = False
            self.texture.repeat_y = False
            self.width = w
            self.height = h

        self.texture.write(np.ascontiguousarray(rgb))

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height > 0 else 4.0 / 3.0
