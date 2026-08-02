"""
The mirror is the highest-consequence, lowest-visibility transform in the app.
Nothing throws when it is wrong; the whole thing just feels reversed.
"""

from __future__ import annotations

import unittest

from mantra_hands.config import LM
from mantra_hands.landmarks import HandFrame, facing_of, hand_size_of, mirror_x, palm_center_x

from .hand_fixture import make_hand

ASPECT = 4.0 / 3.0


class MirrorTransform(unittest.TestCase):
    def test_maps_x_to_one_minus_x_and_is_its_own_inverse(self) -> None:
        self.assertEqual(mirror_x(0.0), 1.0)
        self.assertEqual(mirror_x(1.0), 0.0)
        self.assertEqual(mirror_x(0.5), 0.5)
        self.assertAlmostEqual(mirror_x(0.25), 0.75, places=12)
        for x in (0.0, 0.13, 0.5, 0.87, 1.0):
            self.assertAlmostEqual(mirror_x(mirror_x(x)), x, places=12)

    def test_flips_every_landmark_x_and_leaves_y_untouched(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        raw = make_hand(cx=0.3, cy=0.6)
        frame.ingest(raw, "Left", True, 0.0)

        for i in range(LM.count):
            self.assertAlmostEqual(float(frame.lm[i * 2]), 1.0 - raw[i].x, places=5)
            self.assertAlmostEqual(float(frame.lm[i * 2 + 1]), raw[i].y, places=5)

    def test_puts_a_hand_on_the_raw_frame_right_onto_the_display_left(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        # Raw x near 0.85 — the right-hand side of the un-mirrored camera image.
        frame.ingest(make_hand(cx=0.15, cy=0.5), "Left", True, 0.0)
        self.assertLess(frame.palm_x, 0.3)
        self.assertAlmostEqual(palm_center_x(frame.lm), frame.palm_x, places=6)

    def test_mirrors_before_deriving_geometry_so_hand_size_is_unchanged(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        frame.ingest(make_hand(scale=0.12), "Left", True, 0.0)
        self.assertAlmostEqual(frame.hand_size, 0.12, places=2)
        self.assertAlmostEqual(hand_size_of(frame.lm, ASPECT), frame.hand_size, places=6)


def facing_for(right: bool, palm_to_camera: bool, cos_tilt: float = 1.0) -> float:
    frame = HandFrame()
    frame.set_aspect(ASPECT)
    frame.ingest(
        make_hand(right=right, palm_to_camera=palm_to_camera, cos_tilt=cos_tilt),
        "Left",
        right,
        0.0,
    )
    return frame.facing


class PalmFacing(unittest.TestCase):
    def test_is_positive_when_either_palm_faces_the_camera(self) -> None:
        self.assertGreater(facing_for(True, True), 0.5)
        self.assertGreater(facing_for(False, True), 0.5)

    def test_is_negative_when_either_hand_shows_its_back(self) -> None:
        self.assertLess(facing_for(True, False), -0.5)
        self.assertLess(facing_for(False, False), -0.5)

    def test_reads_the_same_for_both_hands_so_one_threshold_covers_both(self) -> None:
        self.assertAlmostEqual(facing_for(True, True), facing_for(False, True), places=5)

    def test_falls_off_as_the_cosine_of_the_palm_tilt(self) -> None:
        square = facing_for(True, True, 1.0)
        self.assertAlmostEqual(facing_for(True, True, 0.5), square * 0.5, places=2)
        self.assertAlmostEqual(facing_for(True, True, 0.0), 0.0, places=3)

    def test_is_scale_free(self) -> None:
        near = HandFrame()
        far = HandFrame()
        near.set_aspect(ASPECT)
        far.set_aspect(ASPECT)
        near.ingest(make_hand(scale=0.26), "Left", True, 0.0)
        far.ingest(make_hand(scale=0.07), "Left", True, 0.0)
        self.assertAlmostEqual(near.facing, far.facing, places=3)

    def test_agrees_with_the_standalone_helper(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        frame.ingest(make_hand(), "Left", True, 0.0)
        self.assertAlmostEqual(
            facing_of(frame.lm, ASPECT, frame.hand_size, True), frame.facing, places=6
        )


class Extrapolation(unittest.TestCase):
    def test_steps_landmarks_forward_along_the_measured_velocity(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        # Two samples 33 ms apart, hand travelling +0.03 in x per sample.
        frame.ingest(make_hand(cx=0.4), "Left", True, 0.0)
        frame.ingest(make_hand(cx=0.43), "Left", True, 33.0)
        frame.ingest(make_hand(cx=0.46), "Left", True, 66.0)

        at_sample = float(frame.view[LM.wrist * 2])
        frame.predict(66.0 + 16.0)
        ahead = float(frame.view[LM.wrist * 2])
        # The One Euro filter damps the measured velocity, so the step is smaller
        # than the raw motion — but it has to move, and it has to move forward.
        self.assertGreater(ahead, at_sample)
        self.assertLess(ahead - at_sample, 0.03)

    def test_never_moves_a_landmark_further_than_the_clamp_allows(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        frame.ingest(make_hand(cx=0.2), "Left", True, 0.0)
        frame.ingest(make_hand(cx=0.8), "Left", True, 16.0)
        base = float(frame.lm[LM.wrist * 2])
        # Ask for a step far past the extrapolation horizon.
        frame.predict(16.0 + 5000.0)
        self.assertLessEqual(abs(float(frame.view[LM.wrist * 2]) - base), 0.0501)

    def test_holds_still_for_a_hand_that_is_not_moving(self) -> None:
        frame = HandFrame()
        frame.set_aspect(ASPECT)
        for i in range(8):
            frame.ingest(make_hand(cx=0.5), "Left", True, i * 33.0)
        frame.predict(8 * 33.0 + 16.0)
        self.assertAlmostEqual(
            float(frame.view[LM.wrist * 2]), float(frame.lm[LM.wrist * 2]), places=4
        )


if __name__ == "__main__":
    unittest.main()
