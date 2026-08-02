"""
The webcam.

The resolution is a performance decision, not a laziness one: detection cost
scales with pixel count, and 640x480 is the cheapest input the landmarker still
does well on for a hand that fills a reasonable part of the frame. The displayed
video is a separate matter — it is drawn as a GL texture at whatever the camera
gives us and stretched to fit.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import VIDEO


@dataclass
class CameraInfo:
    width: int
    height: int
    fps: float
    label: str


class CameraError(RuntimeError):
    """Raised with copy that is safe to put in front of a user."""


def _backend() -> int:
    """
    AVFoundation on macOS, V4L2 on Linux, DirectShow on Windows.

    The default backend on macOS is AVFoundation anyway, but naming it skips a
    probe of every other backend first — which on a laptop with a virtual camera
    installed is the difference between opening in 300 ms and opening in three
    seconds.
    """
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    if system == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class Camera:
    """
    A camera that hands over one decoded frame per ``read()``.

    ``read()`` blocks until the driver has a new frame, which is what makes it a
    clock: the detection thread calls it in a loop and therefore runs at exactly
    camera rate, once per decoded frame and never twice on the same image. This
    is the Python equivalent of driving detection from
    ``requestVideoFrameCallback`` rather than from a timer.
    """

    def __init__(self, index: int = 0) -> None:
        self.info = CameraInfo(VIDEO.width, VIDEO.height, VIDEO.fps, "camera")
        cap = cv2.VideoCapture(index, _backend())
        if not cap.isOpened():
            # A second try on the default backend, for the odd driver that only
            # shows up there.
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise CameraError(
                "No camera found, or it is being held by something else — close "
                "any video call or other app using it and try again."
                + (
                    "\n\nOn macOS the terminal you launched from needs camera "
                    "permission: System Settings → Privacy & Security → Camera."
                    if sys.platform == "darwin"
                    else ""
                )
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO.height)
        cap.set(cv2.CAP_PROP_FPS, VIDEO.fps)
        # A one-frame queue. Anything deeper and the landmarker is looking at a
        # hand that has already moved on, which reads as lag no amount of
        # extrapolation can hide.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass  # Not every backend has it, and it is an optimisation.

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError(
                "The camera opened but never sent any video. Close anything else "
                "using it — another app, a video call — and try again."
            )

        self._cap = cap
        h, w = frame.shape[:2]
        self.info = CameraInfo(
            width=w,
            height=h,
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or float(VIDEO.fps),
            label=f"camera {index}",
        )
        self._first: Optional[np.ndarray] = frame

    def read(self) -> Optional[np.ndarray]:
        """One BGR frame, or None once the device has gone away."""
        if self._first is not None:
            frame = self._first
            self._first = None
            return frame
        ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def size(self) -> Tuple[int, int]:
        return self.info.width, self.info.height

    @property
    def aspect(self) -> float:
        return self.info.width / max(1, self.info.height)

    def release(self) -> None:
        """A webcam light left on after someone closes the app is not acceptable."""
        self._cap.release()
