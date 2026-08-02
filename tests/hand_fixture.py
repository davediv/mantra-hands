"""
A synthetic hand, in RAW camera coordinates — i.e. not mirrored, exactly as the
landmarker would hand them over. Feeding these through HandFrame exercises the
real mirror and the real geometry rather than a stand-in.

Layout is in units of hand size (wrist to middle MCP), laid out with the hand
upright. Local +x points toward the thumb side of the hand.
"""

from __future__ import annotations

from typing import List, Optional

from mantra_hands.config import LM
from mantra_hands.landmarks import RawPoint

MCP = (
    (0.38, -0.9),  # index  (5)
    (0.05, -1.0),  # middle (9)
    (-0.28, -0.95),  # ring   (13)
    (-0.55, -0.85),  # pinky  (17)
)

#: Splay direction for each finger, roughly "up" with a little fan.
DIR = (
    (0.05, -0.999),
    (0.0, -1.0),
    (-0.08, -0.997),
    (-0.15, -0.989),
)

FINGER_BASE = (LM.index_mcp, LM.middle_mcp, LM.ring_mcp, LM.pinky_mcp)


def make_hand(
    cx: float = 0.5,
    cy: float = 0.5,
    scale: float = 0.12,
    right: bool = True,
    palm_to_camera: bool = True,
    cos_tilt: float = 1.0,
    curled: bool = False,
    aspect: float = 4.0 / 3.0,
) -> List[RawPoint]:
    """
    Builds one hand's 21 landmarks in raw (un-mirrored) normalized coordinates.

    :param cx: palm centre in display (mirrored) normalized coordinates.
    :param scale: hand size — wrist to middle MCP — in normalized y units.
    :param right: anatomically the right hand.
    :param palm_to_camera: palm toward the camera, versus the back of the hand.
    :param cos_tilt: rotation of the palm about the vertical axis, as
        cos(angle). 1 is square to the camera. Geometrically this is exactly an x
        compression, which is what makes it a faithful way to sweep the facing
        signal.
    :param curled: fingers folded into the palm.
    :param aspect: frame aspect ratio, so normalized x and y stay isotropic.
    """
    # Which way the thumb points on screen. A right hand with its palm at the
    # camera is anatomical position, so its thumb points to the user's own
    # right — display +x. Turning the hand over, or swapping hands, flips it.
    flip = 1.0 if right == palm_to_camera else -1.0

    out: List[Optional[RawPoint]] = [None] * LM.count

    def put(index: int, lx: float, ly: float) -> None:
        dx = (flip * lx * cos_tilt * scale) / aspect
        dy = ly * scale
        # Built in display space, then flipped back to raw. HandFrame will mirror
        # it again, and the two must cancel.
        out[index] = RawPoint(1.0 - (cx + dx), cy + dy)

    put(LM.wrist, 0.0, 0.0)

    # Thumb, fanned out to the thumb side.
    put(1, 0.28, -0.16)
    put(LM.thumb_mcp, 0.52, -0.34)
    put(3, 0.68, -0.52)
    put(LM.thumb_tip, 0.8, -0.68)

    for f in range(4):
        mx, my = MCP[f]
        dx, dy = DIR[f]
        base = FINGER_BASE[f]
        put(base, mx, my)
        if curled:
            # Folded: the PIP stays out but the tip swings back toward the wrist,
            # so distance(tip, wrist) drops below distance(pip, wrist).
            put(base + 1, mx + dx * 0.42, my + dy * 0.42)
            put(base + 2, mx * 0.95, my * 0.95)
            put(base + 3, mx * 0.85, my * 0.85)
        else:
            put(base + 1, mx + dx * 0.42, my + dy * 0.42)
            put(base + 2, mx + dx * 0.68, my + dy * 0.68)
            put(base + 3, mx + dx * 0.95, my + dy * 0.95)

    return [p for p in out if p is not None]
