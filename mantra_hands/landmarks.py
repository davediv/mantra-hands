"""
THE MIRROR, the derived hand geometry, and one tracked hand.

The camera image is displayed flipped so the user sees themselves in a mirror,
which is the only orientation that feels natural to move in. The landmarker,
however, sees the raw un-flipped frame.

Every landmark x is flipped to (1 - x) the instant it arrives, before any other
code touches it. Get this wrong and nothing crashes — the app just feels subtly,
maddeningly reversed, and you will spend an evening looking in the wrong place
for it.

This is the only mirror in the codebase. Nothing downstream flips anything.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np

from .config import FINGERS, LM, TRACKING
from .filters import OneEuroBank


def mirror_x(x: float) -> float:
    return 1.0 - x


class RawPoint(NamedTuple):
    """One landmark as the detector hands it over. MediaPipe's own
    NormalizedLandmark duck-types as this."""

    x: float
    y: float


COORDS = LM.count * 2


def dist_norm(lm: np.ndarray, a: int, b: int, aspect: float) -> float:
    """
    Normalized landmark coordinates run 0..1 across the frame, so an x unit is
    640 px wide and a y unit is 480 px tall. Every distance must scale x by the
    aspect ratio or a hand held sideways would measure differently from the same
    hand held upright.
    """
    dx = (float(lm[a * 2]) - float(lm[b * 2])) * aspect
    dy = float(lm[a * 2 + 1]) - float(lm[b * 2 + 1])
    return math.sqrt(dx * dx + dy * dy)


def hand_size_of(lm: np.ndarray, aspect: float) -> float:
    """
    Hand size: wrist to middle-finger MCP. Every other distance in the app is
    divided by this, which is what makes gestures behave the same at 0.5 m and
    at 2 m.
    """
    return max(1e-4, dist_norm(lm, LM.wrist, LM.middle_mcp, aspect))


def palm_center_x(lm: np.ndarray) -> float:
    """Palm centre: the centroid of the wrist and the four MCP knuckles."""
    return (
        float(lm[LM.wrist * 2])
        + float(lm[LM.index_mcp * 2])
        + float(lm[LM.middle_mcp * 2])
        + float(lm[LM.ring_mcp * 2])
        + float(lm[LM.pinky_mcp * 2])
    ) / 5.0


def palm_center_y(lm: np.ndarray) -> float:
    return (
        float(lm[LM.wrist * 2 + 1])
        + float(lm[LM.index_mcp * 2 + 1])
        + float(lm[LM.middle_mcp * 2 + 1])
        + float(lm[LM.ring_mcp * 2 + 1])
        + float(lm[LM.pinky_mcp * 2 + 1])
    ) / 5.0


def bbox_area(lm: np.ndarray, aspect: float) -> float:
    """
    Hand bounding-box area, aspect corrected. This is the depth proxy the fire
    gesture runs on: it is the only signal that reliably grows as a hand comes at
    the camera. Landmark z is not usable for absolute depth.
    """
    xs = lm[0::2]
    ys = lm[1::2]
    return float((xs.max() - xs.min()) * aspect * (ys.max() - ys.min()))


def facing_of(lm: np.ndarray, aspect: float, hand_size: float, anatomical_right: bool) -> float:
    """
    How squarely the palm faces the camera, as a signed scalar. Positive means
    palm-at-camera; negative means we are looking at the back of the hand.

    The palm normal is the cross product of (lm5 - lm0) and (lm17 - lm0). Both
    vectors lie in the palm plane, so their cross product is perpendicular to it,
    and with only x and y available the cross product collapses to its z
    component: ax*by - ay*bx.

    Sign, derived once so nobody has to do it again. Work in mirrored display
    coordinates: +x is the user's own right, +y is down. A right hand with the
    palm toward the camera is anatomical position, so its thumb — and therefore
    lm5 — sits to the user's right of lm17::

        a = lm5  - lm0 ~ (+0.3, -0.7)
        b = lm17 - lm0 ~ (-0.3, -0.7)
        cross = (0.3)(-0.7) - (-0.7)(-0.3) = -0.42      negative

    The same hand turned around, or the other hand, flips that sign. So the raw
    cross product is negated for a right hand and left alone for a left hand.

    Divided by hand_size**2 to make it scale-free. A palm square to the camera
    reads about 0.5 and falls off as cos(tilt), crossing zero edge-on.
    """
    ox = float(lm[LM.wrist * 2])
    oy = float(lm[LM.wrist * 2 + 1])
    ax = (float(lm[LM.index_mcp * 2]) - ox) * aspect
    ay = float(lm[LM.index_mcp * 2 + 1]) - oy
    bx = (float(lm[LM.pinky_mcp * 2]) - ox) * aspect
    by = float(lm[LM.pinky_mcp * 2 + 1]) - oy
    cross = ax * by - ay * bx
    scaled = cross / (hand_size * hand_size)
    return -scaled if anatomical_right else scaled


def finger_extensions(lm: np.ndarray, aspect: float, out: np.ndarray) -> np.ndarray:
    """
    Per-finger extension ratio: distance(tip, wrist) / distance(pip, wrist).

    Above 1 the fingertip is further from the wrist than its own middle joint,
    which only happens when the finger is straight. Curl the finger and the tip
    swings back toward the palm and the ratio drops below 1.

    Ratio of two distances, so it is scale-free for free, and it needs no
    per-finger calibration — a pinky and an index read the same.
    """
    for f, triple in enumerate(FINGERS):
        pip = dist_norm(lm, triple[1], LM.wrist, aspect)
        tip = dist_norm(lm, triple[2], LM.wrist, aspect)
        out[f] = tip / pip if pip > 1e-5 else 0.0
    return out


class HandFrame:
    """
    One tracked hand: mirrored, smoothed landmarks plus the handful of derived
    quantities every gesture and effect needs.

    Everything is preallocated. ingest() and predict() do not allocate.
    """

    def __init__(self) -> None:
        self.present = False
        #: Detections missed in a row.
        self.missing = 999
        #: MediaPipe's raw handedness string, kept for the HUD.
        self.label = ""
        #: True when this is anatomically the user's right hand. Owned by the caller.
        self.anatomical_right = False

        #: Mirrored, filtered landmarks in video-normalized space: [x0,y0, x1,y1, ...].
        self.lm = np.zeros(COORDS, dtype=np.float32)
        #: Per-channel velocity, normalized units per second, smoothed.
        self.vel = np.zeros(COORDS, dtype=np.float32)
        #: What the renderer reads: lm stepped forward to the current display frame.
        self.view = np.zeros(COORDS, dtype=np.float32)

        self.hand_size = 0.1
        self.palm_x = 0.5
        self.palm_y = 0.5
        #: Palm centre for the current display frame, extrapolated alongside `view`.
        self.view_palm_x = 0.5
        self.view_palm_y = 0.5
        self.area = 0.0
        self.facing = 0.0
        self.finger_ext = np.zeros(4, dtype=np.float32)

        #: Monotonic clock, in ms, of the detection that produced `lm`.
        self.sample_ms = -1.0

        self._aspect = 4.0 / 3.0
        self._filters = OneEuroBank(COORDS)
        self._primed = False
        # Scratch for ingest(), so folding a detection in allocates nothing.
        self._raw = np.zeros(COORDS, dtype=np.float32)
        self._filtered = np.zeros(COORDS, dtype=np.float32)
        self._step = np.zeros(COORDS, dtype=np.float32)

    def set_aspect(self, aspect: float) -> None:
        self._aspect = aspect

    def ingest(
        self,
        points: Sequence[RawPoint],
        label: str,
        anatomical_right: bool,
        t_ms: float,
    ) -> None:
        """
        Folds one detection into this slot: mirror, smooth, differentiate, then
        recompute the derived geometry.

        :param anatomical_right: whether this is really the user's right hand,
            which only the caller can know — it depends on MediaPipe's handedness
            convention, not on the landmarks.
        """
        # Clamped so a stalled process cannot hand the filter a dt that makes it
        # either a passthrough or a brick wall.
        dt = 1.0 / 30.0 if self.sample_ms < 0 else min(0.2, max(1e-3, (t_ms - self.sample_ms) / 1000.0))
        inv_dt = 1.0 / dt

        raw = self._raw
        for i in range(LM.count):
            p = points[i]
            # MIRROR FIRST. Everything below this line, and everything downstream
            # of this class, sees only mirrored coordinates.
            raw[i * 2] = mirror_x(p.x)
            raw[i * 2 + 1] = p.y

        filtered = self._filters.filter_all(raw, dt, self._filtered)

        if self._primed:
            # Velocity is itself smoothed. Differentiating an already-smoothed
            # signal amplifies whatever noise survived, and extrapolating on a
            # noisy velocity looks worse than not extrapolating at all.
            step = self._step
            np.subtract(filtered, self.lm, out=step)
            step *= inv_dt
            step -= self.vel
            step *= 0.5
            self.vel += step

        self.lm[:] = filtered
        self.view[:] = filtered

        self.present = True
        self.missing = 0
        self._primed = True
        self.label = label
        self.anatomical_right = anatomical_right
        self.sample_ms = t_ms

        self.hand_size = hand_size_of(self.lm, self._aspect)
        self.palm_x = palm_center_x(self.lm)
        self.palm_y = palm_center_y(self.lm)
        self.view_palm_x = self.palm_x
        self.view_palm_y = self.palm_y
        self.area = bbox_area(self.lm, self._aspect)
        self.facing = facing_of(self.lm, self._aspect, self.hand_size, anatomical_right)
        finger_extensions(self.lm, self._aspect, self.finger_ext)

    def mark_missing(self) -> None:
        """Marks a detection cycle in which this slot saw nothing."""
        self.present = False
        self.missing += 1
        if self.missing == TRACKING.forget_frames:
            self.reset()

    def reset(self) -> None:
        self._filters.reset()
        self.vel.fill(0.0)
        self._primed = False
        self.sample_ms = -1.0
        self.facing = 0.0
        self.finger_ext.fill(0.0)

    def predict(self, now_ms: float) -> None:
        """
        Steps the landmarks forward to ``now_ms`` for rendering.

        Detection lands at 30 Hz; the display runs at 60. Without this, every
        other rendered frame repeats the previous one and the rune visibly
        stair-steps across the palm. With it, the visuals are smooth even though
        the tracking underneath is not.

        Two clamps keep a bad velocity from throwing landmarks across the screen:
        the time step is capped, and each channel's step is capped in absolute
        terms. Past a normal detection interval, confidence in the velocity falls
        off — a gap that long usually means the detection was dropped, not that
        the hand is still moving the way it was.
        """
        if not self._primed or self.sample_ms < 0:
            return
        gap_ms = now_ms - self.sample_ms
        if gap_ms <= 0:
            return

        step_sec = min(gap_ms, TRACKING.max_extrapolate_ms) / 1000.0
        over = max(0.0, gap_ms - TRACKING.extrapolate_hold_ms)
        span = max(1.0, TRACKING.max_extrapolate_ms - TRACKING.extrapolate_hold_ms)
        conf = 1.0 - TRACKING.extrapolate_falloff * min(1.0, over / span)
        k = step_sec * conf
        cap = TRACKING.max_extrapolate_step

        step = self._step
        np.multiply(self.vel, k, out=step)
        np.clip(step, -cap, cap, out=step)
        np.add(self.lm, step, out=self.view)

        self.view_palm_x = palm_center_x(self.view)
        self.view_palm_y = palm_center_y(self.view)
