"""
Hand tracking, decoupled from rendering.

THIS IS THE MOST IMPORTANT ARCHITECTURAL DECISION IN THE APP.

Detection runs on its own clock — a thread whose loop blocks on the camera — so
it fires exactly once per decoded camera frame, about 30 Hz, and never twice on
the same image. The render loop is the main thread, running at display rate. The
two never wait on each other: the render loop reads whatever the last detection
left behind and extrapolates forward from it (see :meth:`HandFrame.predict`).

The browser build gets this from ``requestVideoFrameCallback`` plus
``requestAnimationFrame``; here it is a camera-blocked thread plus a swap-blocked
main loop. If detection ran inline the frame rate would be pinned to the camera
and every inference hitch would be a dropped frame. This way the visuals stay at
display rate even though the tracking underneath them is at 30.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from .clock import now_ms
from .config import GESTURE, TRACKING, find_model
from .demo import DemoHands
from .landmarks import HandFrame

MAX_HANDS = 2


class Tracker:
    def __init__(self) -> None:
        self.hands: Sequence[HandFrame] = (HandFrame(), HandFrame())
        self.delegate = "GPU"
        #: Wall time of the most recent detect_for_video call, in ms.
        self.detect_ms = 0.0
        #: Completed detections per second.
        self.detect_hz = 0.0
        self.running = False

        #: Called after every detection, on the detection thread. Gestures live here.
        self.on_detect: Optional[Callable[[float], None]] = None

        #: Set in demo mode: scripted hands stand in for the landmarker.
        self.demo: Optional[DemoHands] = None

        self._landmarker = None
        self._camera = None
        self._thread: Optional[threading.Thread] = None
        self._last_timestamp = -1
        self._hz_count = 0
        self._hz_since = 0.0

        # The landmark arrays are written on the detection thread and read on the
        # render thread. The lock is held for the length of one ingest and one
        # predict — tens of microseconds — and never across anything that blocks.
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_dirty = False

        # MediaPipe labels handedness as though the input image were already
        # mirrored — a selfie feed. We hand it the raw camera frame, so the
        # reported label is the opposite of the anatomical hand.
        #
        # This flag exists because that convention has quietly changed between
        # MediaPipe releases, and it is the sign of the palm-facing test. If it
        # were wrong the app would never charge and there would be nothing on
        # screen to explain why, so _calibrate watches for exactly that failure
        # and flips it.
        self._handedness_inverted = True
        self._handedness_confirmed = False
        self._calib_wrong_ms = 0.0
        self._calib_right_ms = 0.0
        self._last_detect_ms = -1.0

    # ------------------------------------------------------------------ load ---

    def load(self, on_progress: Optional[Callable[[str], None]] = None) -> None:
        if self.demo is not None:
            self.delegate = "CPU"
            return

        if on_progress:
            on_progress("Loading hand model")

        # MediaPipe prints six lines to stderr on the way up — an XNNPACK notice,
        # two about tensor signatures, one about NORM_RECT. None are actionable
        # and none can be turned off: they come from absl's C++ logger, which
        # ignores GLOG_minloglevel and GLOG_stderrthreshold in this build (both
        # measured, neither works). Silencing them would mean redirecting fd 2
        # around the load, which would also swallow the errors that matter.
        #
        # Imported here rather than at module scope: mediapipe takes a couple of
        # seconds to import, and the demo path and the unit tests must not pay
        # for a dependency they never touch.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = str(find_model())

        def build(delegate) -> "vision.HandLandmarker":  # noqa: ANN001
            base = mp_python.BaseOptions(model_asset_path=model_path, delegate=delegate)
            options = vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=TRACKING.num_hands,
                min_hand_detection_confidence=TRACKING.min_hand_detection_confidence,
                min_hand_presence_confidence=TRACKING.min_hand_presence_confidence,
                min_tracking_confidence=TRACKING.min_tracking_confidence,
            )
            return vision.HandLandmarker.create_from_options(options)

        # THE DELEGATE IS CHOSEN BEFORE IT IS TRIED, NOT AFTER IT FAILS.
        #
        # The browser build asked for GPU and fell back to CPU in a catch, which
        # works because a WebGL delegate that cannot run the graph throws. The
        # Python runtime does not get that courtesy: on macOS the GPU delegate
        # constructs happily and then aborts the whole process inside
        # Image::ConvertToGpu on the first detect — a SIGABRT out of C++, with no
        # exception to catch and nothing left to fall back with.
        #
        # So the platform decides. GPU where MediaPipe actually supports it, CPU
        # everywhere else, and an override for anyone who wants to argue.
        want = os.environ.get("MANTRA_HANDS_DELEGATE", "").strip().lower()
        use_gpu = want == "gpu" or (want == "" and sys.platform.startswith("linux"))

        if use_gpu:
            try:
                self._landmarker = build(mp_python.BaseOptions.Delegate.GPU)
                self.delegate = "GPU"
                return
            except Exception:
                if on_progress:
                    on_progress("Loading hand model (CPU)")

        # CPU is slower but it is not a failure state: hand landmarking on 640x480
        # costs about 13 ms a frame on an M1, which fits inside the 33 ms the
        # camera gives us with room to spare.
        self._landmarker = build(mp_python.BaseOptions.Delegate.CPU)
        self.delegate = "CPU"

    def set_video_aspect(self, aspect: float) -> None:
        for hand in self.hands:
            hand.set_aspect(aspect)

    # ----------------------------------------------------------------- clock ---

    def start(self, camera) -> None:
        """Begins the detection clock. Safe to call once the camera has frames."""
        if self.running:
            return
        self._camera = camera
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="detect", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

    def _loop(self) -> None:
        while self.running:
            frame = self._camera.read()
            if frame is None:
                # The device went away mid-session. Stop cleanly rather than spin.
                self.running = False
                return
            # BGR from OpenCV, RGB for both MediaPipe and the GL upload. One
            # conversion serves both, and it happens here rather than on the
            # render thread.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
                self._frame_dirty = True
            try:
                self._detect(rgb)
            except Exception:
                # One dropped inference is not worth tearing down the loop over.
                continue

    def take_frame(self) -> Optional[np.ndarray]:
        """
        The newest camera frame, or None if the last one has already been taken.

        The render loop uploads what this returns. Half of all display frames
        have no new image, and those skip the upload entirely.
        """
        with self._lock:
            if not self._frame_dirty:
                return None
            self._frame_dirty = False
            return self._frame

    # ------------------------------------------------------------- detection ---

    def _detect(self, rgb: np.ndarray) -> None:
        if self.demo is not None:
            self._detect_demo()
            return
        if self._landmarker is None:
            return

        import mediapipe as mp

        now = now_ms()
        # detect_for_video rejects a timestamp that does not advance.
        timestamp = int(now)
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1
        self._last_timestamp = timestamp

        t0 = now_ms()
        result = None
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, timestamp)
        except Exception:
            result = None
        self.detect_ms = now_ms() - t0

        dt_ms = 0.0 if self._last_detect_ms < 0 else min(200.0, now - self._last_detect_ms)
        self._last_detect_ms = now

        taken = [False, False]
        if result is not None:
            landmarks = getattr(result, "hand_landmarks", None) or []
            handedness = getattr(result, "handedness", None) or []
            with self._lock:
                for i in range(min(len(landmarks), MAX_HANDS)):
                    points = landmarks[i]
                    if points is None or len(points) < 21:
                        continue
                    label = "Right" if i == 0 else "Left"
                    if i < len(handedness) and handedness[i]:
                        label = handedness[i][0].category_name
                    slot = self._slot_for(label, taken)
                    if slot < 0:
                        continue
                    taken[slot] = True
                    # Anatomical hand, after correcting for MediaPipe's mirror
                    # assumption.
                    reported_right = label == "Right"
                    anatomical_right = (
                        not reported_right if self._handedness_inverted else reported_right
                    )
                    self.hands[slot].ingest(points, label, anatomical_right, now)

        with self._lock:
            for s in range(MAX_HANDS):
                if not taken[s]:
                    self.hands[s].mark_missing()

        self._calibrate(dt_ms)
        self._count_rate(now)
        if self.on_detect:
            self.on_detect(now)

    def _detect_demo(self) -> None:
        """
        The demo path. Everything below the landmarker is identical: the same
        mirror, the same One Euro bank, the same derived geometry, the same
        on_detect callback on the same clock.
        """
        demo = self.demo
        if demo is None:
            return
        now = now_ms()
        demo.update(now)
        self.detect_ms = 0.0

        with self._lock:
            for s in range(MAX_HANDS):
                hand = demo.hands[s]
                if hand.present:
                    self.hands[s].ingest(
                        hand.points, "Right" if hand.right else "Left", hand.right, now
                    )
                else:
                    self.hands[s].mark_missing()

        self._count_rate(now)
        if self.on_detect:
            self.on_detect(now)

    @staticmethod
    def _slot_for(label: str, taken: List[bool]) -> int:
        """
        Handedness slotting. The label is stable across frames; the array index is
        not, and filter state has to follow the same physical hand or a two-handed
        cast turns into two hands swapping smoothing histories every few frames.
        """
        preferred = 0 if label == "Left" else 1
        if not taken[preferred]:
            return preferred
        other = 1 - preferred
        return -1 if taken[other] else other

    def _calibrate(self, dt_ms: float) -> None:
        """
        Watches for the palm-facing sign being backwards.

        If a hand is open — all four fingers extended — and the facing test reads
        strongly *negative* for over a second and a half without ever having read
        positive, then the handedness convention is inverted from what we assumed
        and the app would simply never charge. Flip it once and let the next
        detection prove it out. Confirming a single successful charge locks it.
        """
        if self._handedness_confirmed or dt_ms <= 0:
            return
        for hand in self.hands:
            if not hand.present:
                continue
            open_fingers = int(np.count_nonzero(hand.finger_ext > GESTURE.extend_on))
            if open_fingers < 4:
                continue
            if hand.facing > GESTURE.facing_on:
                self._calib_right_ms += dt_ms
            elif hand.facing < -GESTURE.facing_on:
                self._calib_wrong_ms += dt_ms
        if self._calib_wrong_ms > 1500 and self._calib_right_ms < 250:
            self._handedness_inverted = not self._handedness_inverted
            self._calib_wrong_ms = 0.0
            self._calib_right_ms = 0.0

    def confirm_handedness(self) -> None:
        """Called once a charge has actually started, which proves the sign is right."""
        self._handedness_confirmed = True

    def _count_rate(self, now: float) -> None:
        if self._hz_since == 0.0:
            self._hz_since = now
        self._hz_count += 1
        elapsed = now - self._hz_since
        if elapsed >= 500:
            self.detect_hz = (self._hz_count * 1000.0) / elapsed
            self._hz_count = 0
            self._hz_since = now

    # ---------------------------------------------------------------- render ---

    def predict(self, now_ms_: float) -> None:
        """Steps every tracked hand forward to the current display frame."""
        with self._lock:
            for hand in self.hands:
                hand.predict(now_ms_)

    @property
    def hands_visible(self) -> int:
        return sum(1 for hand in self.hands if hand.present)

    @property
    def handedness(self) -> str:
        return "inverted" if self._handedness_inverted else "direct"
