"""
Every tunable in the app, in one place.

Nothing here is read inside the hot path more than once per frame, and none of
these objects is ever mutated at runtime — the low-power tier is a second field
the governor reads instead of a value it overwrites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Final, Tuple


@dataclass(frozen=True)
class VideoConfig:
    #: 640x480 is deliberate. Halving the input resolution roughly halves
    #: detection cost and the landmarker does not get meaningfully better above
    #: this for a hand that fills a decent part of the frame. Raise it only if
    #: tracking quality actually demands it.
    width: int = 640
    height: int = 480
    fps: int = 30


VIDEO: Final = VideoConfig()


@dataclass(frozen=True)
class LandmarkIndices:
    """0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky."""

    count: int = 21
    wrist: int = 0
    thumb_mcp: int = 2
    thumb_tip: int = 4
    index_mcp: int = 5
    index_pip: int = 6
    index_tip: int = 8
    middle_mcp: int = 9
    middle_pip: int = 10
    middle_tip: int = 12
    ring_mcp: int = 13
    ring_pip: int = 14
    ring_tip: int = 16
    pinky_mcp: int = 17
    pinky_pip: int = 18
    pinky_tip: int = 20


LM: Final = LandmarkIndices()

#: The four fingers the charge gesture checks, as (mcp, pip, tip) triples.
FINGERS: Final[Tuple[Tuple[int, int, int], ...]] = (
    (LM.index_mcp, LM.index_pip, LM.index_tip),
    (LM.middle_mcp, LM.middle_pip, LM.middle_tip),
    (LM.ring_mcp, LM.ring_pip, LM.ring_tip),
    (LM.pinky_mcp, LM.pinky_pip, LM.pinky_tip),
)


@dataclass(frozen=True)
class TrackingConfig:
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # One Euro filter. Raw MediaPipe landmarks jitter by a couple of pixels
    # every frame; at the scale the rune is drawn that reads as a buzzing halo.
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0

    #: Render-side extrapolation. Detection lands at ~30 Hz and the render loop
    #: runs at 60, so half of all frames have no fresh data. Those frames step
    #: the last known velocity forward instead of freezing, which is the
    #: difference between silk and a 30 Hz stutter.
    max_extrapolate_ms: float = 40.0
    #: Hard cap on how far a single extrapolation may move a landmark, in
    #: normalized units.
    max_extrapolate_step: float = 0.05
    #: Velocity is trusted at full weight for one detection interval. Past that
    #: the detection was probably dropped rather than the hand still moving, so
    #: confidence falls off and the hand coasts to a stop instead of flying off.
    extrapolate_hold_ms: float = 34.0
    extrapolate_falloff: float = 0.5

    #: Frames without a detection before a slot's filter state is dropped.
    forget_frames: int = 5


TRACKING: Final = TrackingConfig()


@dataclass(frozen=True)
class GestureConfig:
    """
    Gesture thresholds. Every one of these is a pair — enter and exit — because
    a bare threshold oscillates on the boundary and users read that as a broken
    app.

    All distances are normalized by hand size (wrist to middle-finger MCP) so a
    gesture behaves identically at 0.5 m and at 2 m from the camera.
    """

    #: Finger extension: distance(tip, wrist) / distance(pip, wrist). Above 1
    #: the tip is further out than its own middle joint, which only happens when
    #: the finger is straight. Scale-free and finger-agnostic, so index and
    #: pinky can share one threshold.
    extend_on: float = 1.15
    extend_off: float = 1.02
    #: How many of the four fingers must read extended.
    extend_count: int = 4

    #: Palm facing. |(lm5-lm0) x (lm17-lm0)| / hand_size^2, signed, corrected
    #: for handedness. A palm square to the camera reads about 0.7; the value
    #: falls off as cos(tilt) and crosses zero when the hand turns edge-on, so
    #: 0.38 is roughly a 57 degree tilt and 0.20 is roughly 74.
    facing_on: float = 0.38
    facing_off: float = 0.2

    #: Palm drift, in hand-widths per second. A charge has to be *held*.
    steady_on: float = 1.4
    steady_off: float = 2.2

    #: Minimum charge before a fire can register. Without this a hand simply
    #: entering the frame — bounding box going from nothing to everything —
    #: reads as a cast.
    charge_min_ms: float = 400.0
    #: Charge time at which the rune is visually maxed out.
    charge_full_ms: float = 1400.0
    #: Once armed, stay armed briefly after the charge conditions lapse. A fast
    #: push blurs the hand and the landmarker's confidence dips for a frame or
    #: two; without the grace window the user's best push is the one that fails.
    arm_grace_ms: float = 260.0

    #: Fire. Hand bounding-box area is the only depth proxy that can be trusted
    #: — landmark z is not usable for absolute depth. Compare the current area
    #: against the area one window ago and look at the relative growth rate.
    #:
    #: A real push from 0.7 m to 0.4 m in 150 ms grows the area about 3x, a rate
    #: near 14/s. Landmark jitter over the same window contributes under 0.4/s.
    area_window_ms: float = 110.0
    expand_rate_on: float = 2.0
    expand_rate_off: float = 0.8
    #: Absolute growth gate, so a small-but-fast wobble cannot clear the rate
    #: test.
    min_area_growth: float = 1.12

    #: A second, shorter window used only to judge how *hard* the push was.
    #:
    #: The trigger has to run on a long window to stay quiet, but a long window
    #: also averages the leading edge away — measured at the moment of firing, a
    #: gentle push and a violent one look nearly the same through it. The short
    #: window is essentially the instantaneous slope and separates them cleanly.
    punch_window_ms: float = 60.0
    #: Short-window rate that counts as a full-power cast.
    punch_full_rate: float = 8.0
    #: The cast fires on the leading edge, which is early — by design, because
    #: latency is what makes it feel like the hand caused it. Power keeps being
    #: revised upward for this long afterwards as the rest of the push arrives,
    #: so the burst swells to match a hard shove. It is a uniform, so revising
    #: it costs nothing.
    power_settle_ms: float = 200.0

    #: After a cast, the hand must settle before it can charge again.
    cooldown_ms: float = 620.0
    #: Both hands firing inside this window is one double-power cast.
    double_cast_window_ms: float = 240.0


GESTURE: Final = GestureConfig()


@dataclass(frozen=True)
class PerfConfig:
    #: Rolling window for the frame-time average, in frames.
    window: int = 60
    #: Rolling average frame time above this for degrade_hold_ms auto-enables
    #: low power.
    degrade_ms: float = 20.0
    degrade_hold_ms: float = 2000.0
    #: Frame budget line drawn in the HUD.
    budget_ms: float = 16.6
    #: Sparkline depth.
    history: int = 120


PERF: Final = PerfConfig()


@dataclass(frozen=True)
class FxConfig:
    """Effect sizing. The low-power tier halves particles and drops bloom resolution."""

    #: 128x128 = 16,384 particles, comfortably over the 12,000 the burst needs.
    #: The texture is square and power-of-two so the sim pass is a clean
    #: fullscreen triangle over it.
    particle_tex: int = 128
    particle_tex_low: int = 96  # 9,216 — still reads as a wall of sparks
    bloom_divisor: int = 4
    bloom_divisor_low: int = 8
    #: Burst lifetime in seconds. Particles fade out over this.
    burst_life: float = 1.5
    #: Screen shake decay, in seconds.
    shake_life: float = 0.42
    #: Chromatic aberration pulse decay, in seconds.
    aberration_life: float = 0.5
    #: How often bolts are regenerated, and the ceiling on their vertex buffer.
    bolt_regen_ms: float = 55.0
    bolt_max_verts: int = 4096
    #: Combo at which effects stop growing, so the screen stays watchable.
    combo_cap: int = 8


FX: Final = FxConfig()


@dataclass(frozen=True)
class RecordConfig:
    width: int = 1080
    height: int = 1920
    width_low: int = 720
    height_low: int = 1280
    fps: int = 30
    max_seconds: float = 30.0
    countdown_seconds: int = 3
    #: FourCC candidates, best first. avc1 gives an MP4 anyone can open; mp4v is
    #: the one OpenCV can always fall back to without a system H.264 encoder.
    fourcc_candidates: Tuple[Tuple[str, str], ...] = (
        ("avc1", ".mp4"),
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
    )


RECORD: Final = RecordConfig()


class Element(IntEnum):
    """Element index. Also the value of the uElement uniform in the shared shader."""

    FIRE = 0
    ICE = 1
    LIGHTNING = 2


#: Part of speech -> element. This mapping is the actual lesson, so it is loud.
POS_ELEMENT: Final[Dict[str, Element]] = {
    "verb": Element.FIRE,
    "noun": Element.ICE,
    "adjective": Element.LIGHTNING,
}

ELEMENT_LABEL: Final = ("FIRE", "ICE", "LIGHTNING")
ELEMENT_POS: Final = ("verb", "noun", "adjective")

#: Element palettes, linear-ish RGB with values above 1 where the bloom should
#: catch. Index by Element.
ELEMENT_COLOR: Final[Tuple[Tuple[float, float, float], ...]] = (
    (1.0, 0.28, 0.05),  # fire: orange-red
    (0.06, 0.72, 1.0),  # ice: cyan
    (1.0, 0.88, 0.42),  # lightning: yellow-white
)

#: The hot end of each palette. Deliberately not white for fire and ice.
#:
#: Thousands of additive sprites already climb toward white on their own, and
#: the filmic curve desaturates whatever is left. Starting from white means
#: every element ends up the same pale grey at the moment it is brightest, which
#: is exactly when it has to be identifiable from across a room.
ELEMENT_HOT: Final[Tuple[Tuple[float, float, float], ...]] = (
    (1.0, 0.74, 0.2),
    (0.42, 0.94, 1.0),
    (1.0, 1.0, 0.86),
)


@dataclass(frozen=True)
class Palette:
    base: str = "#0B0F1A"
    green: str = "#3FD68C"
    yellow: str = "#FFC531"
    magenta: str = "#FF3D7F"
    text: str = "#F2F4F0"


COLORS: Final = Palette()

#: Where the browser build used one localStorage key, this writes one JSON file.
#: Same contract: best effort, never worth losing a session over.
STORAGE_PATH: Final = Path(
    os.environ.get("MANTRA_HANDS_STATE", str(Path.home() / ".mantra-hands.json"))
)

#: The model is vendored into the package, never fetched at runtime. This runs on
#: camp wifi; a cold fetch of an 8 MB blob is the difference between the camera
#: coming up in 300 ms and the room giving up on it.
PACKAGE_ROOT: Final = Path(__file__).resolve().parent
REPO_ROOT: Final = PACKAGE_ROOT.parent
MODEL_CANDIDATES: Final = (
    PACKAGE_ROOT / "models" / "hand_landmarker.task",
    REPO_ROOT / "models" / "hand_landmarker.task",
)


def find_model() -> Path:
    """The vendored hand model, wherever this checkout keeps it."""
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "hand_landmarker.task was not found. Run `python3 -m mantra_hands.vendor` to "
        f"fetch it once — it lands in {MODEL_CANDIDATES[0]} — or drop the file there "
        "yourself from a machine that has it."
    )
