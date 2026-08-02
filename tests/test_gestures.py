"""The charge/fire state machine, driven exactly the way the app drives it."""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from mantra_hands.config import GESTURE
from mantra_hands.gestures import CastRecognizer, Phase
from mantra_hands.landmarks import HandFrame

from .hand_fixture import make_hand

#: One detection tick. Detection runs at camera rate, not display rate.
TICK = 1000.0 / 30.0


class Rig:
    """
    Drives a recognizer the way the app does: synthetic landmarks in, one
    detection tick at a time, counting the fire pulses that come out.
    """

    def __init__(self) -> None:
        self.frame = HandFrame()
        self.rec = CastRecognizer()
        self.t = 0.0
        self.fires = 0
        self.last_power = 0.0
        self.frame.set_aspect(4.0 / 3.0)

    def hold(self, ms: float, opts: Optional[Dict[str, Any]]) -> "Rig":
        """Advances by ``ms``, holding the given pose. Pass None for "no hand"."""
        ticks = max(1, round(ms / TICK))
        for _ in range(ticks):
            self._tick(opts)
        return self

    def push(self, ms: float, frm: float, to: float, **opts: Any) -> "Rig":
        """Advances by ``ms`` while the hand grows from ``frm`` to ``to`` — a push."""
        ticks = max(1, round(ms / TICK))
        for i in range(1, ticks + 1):
            step = dict(opts)
            step["scale"] = frm + (to - frm) * (i / ticks)
            self._tick(step)
        return self

    def _tick(self, opts: Optional[Dict[str, Any]]) -> None:
        self.t += TICK
        if opts is not None:
            self.frame.ingest(make_hand(**opts), "Left", opts.get("right", True), self.t)
        else:
            self.frame.mark_missing()
        self.rec.update(self.frame, self.t)
        if self.rec.fired:
            self.fires += 1
            self.last_power = self.rec.power


class Charge(unittest.TestCase):
    def setUp(self) -> None:
        self.rig = Rig()

    def test_starts_when_an_open_palm_faces_the_camera(self) -> None:
        self.rig.hold(200, {"scale": 0.12})
        self.assertEqual(self.rig.rec.phase, Phase.CHARGING)
        self.assertEqual(self.rig.rec.extended_count, 4)

    def test_arms_only_after_the_minimum_hold(self) -> None:
        self.rig.hold(GESTURE.charge_min_ms - 100, {"scale": 0.12})
        self.assertEqual(self.rig.rec.phase, Phase.CHARGING)
        self.assertFalse(self.rig.rec.armed)

        self.rig.hold(200, {"scale": 0.12})
        self.assertEqual(self.rig.rec.phase, Phase.ARMED)
        self.assertGreaterEqual(self.rig.rec.charge_ms, GESTURE.charge_min_ms)

    def test_never_starts_with_the_fingers_curled(self) -> None:
        self.rig.hold(1500, {"scale": 0.12, "curled": True})
        self.assertEqual(self.rig.rec.phase, Phase.IDLE)
        self.assertEqual(self.rig.rec.extended_count, 0)

    def test_never_starts_on_the_back_of_the_hand(self) -> None:
        self.rig.hold(1500, {"scale": 0.12, "palm_to_camera": False})
        self.assertEqual(self.rig.rec.phase, Phase.IDLE)
        self.assertLess(self.rig.rec.facing, 0)

    def test_works_the_same_for_the_left_hand(self) -> None:
        self.rig.hold(700, {"scale": 0.12, "right": False})
        self.assertEqual(self.rig.rec.phase, Phase.ARMED)

    def test_behaves_identically_near_and_far_from_the_camera(self) -> None:
        near = Rig().hold(700, {"scale": 0.26})
        far = Rig().hold(700, {"scale": 0.06})
        self.assertEqual(near.rec.phase, Phase.ARMED)
        self.assertEqual(far.rec.phase, Phase.ARMED)

    def test_resets_when_the_hand_leaves_the_frame(self) -> None:
        self.rig.hold(700, {"scale": 0.12})
        self.assertEqual(self.rig.rec.phase, Phase.ARMED)
        self.rig.hold(300, None)
        self.assertEqual(self.rig.rec.phase, Phase.IDLE)
        self.assertEqual(self.rig.rec.charge_ms, 0)

    def test_holds_through_a_wobble_across_the_enter_threshold(self) -> None:
        """
        The reason every threshold in this app is a pair. The tilt here crosses
        back and forth over the enter threshold; a bare threshold would drop the
        charge on every other detection and the rune would strobe.
        """
        high = 0.62  # facing ~0.51, above facing_on
        low = 0.4  # facing ~0.33, below facing_on but above facing_off
        for i in range(27):
            self.rig.hold(TICK, {"scale": 0.12, "cos_tilt": high if i % 4 < 2 else low})
        self.assertEqual(self.rig.rec.phase, Phase.ARMED)
        # The low sample really is under the enter threshold — otherwise this
        # test proves nothing.
        probe = Rig().hold(TICK, {"scale": 0.12, "cos_tilt": low})
        self.assertLess(probe.rec.facing, GESTURE.facing_on)
        self.assertGreater(probe.rec.facing, GESTURE.facing_off)

    def test_drops_the_charge_when_the_palm_turns_past_the_exit_threshold(self) -> None:
        self.rig.hold(700, {"scale": 0.12})
        self.assertEqual(self.rig.rec.phase, Phase.ARMED)
        # Turning away takes a beat to register — the One Euro filter has to
        # catch up before facing crosses the exit threshold, and only then does
        # the arm grace start running down. Around 600 ms all in, which is
        # forgiving in the direction that matters: nothing can fire while the
        # palm is turned away.
        self.rig.hold(900, {"scale": 0.12, "cos_tilt": 0.1})
        self.assertEqual(self.rig.rec.phase, Phase.IDLE)


class Fire(unittest.TestCase):
    def test_fires_once_on_a_committed_push_after_a_full_charge(self) -> None:
        rig = Rig().hold(800, {"scale": 0.12}).push(170, 0.12, 0.21).hold(300, {"scale": 0.21})
        self.assertEqual(rig.fires, 1)
        self.assertGreater(rig.last_power, 0.35)
        self.assertLessEqual(rig.last_power, 1.0)

    def test_does_not_misfire_when_a_hand_simply_enters_the_frame(self) -> None:
        """
        The failure this app would be judged on. A hand appearing in frame takes
        the bounding box from nothing to everything, which is exactly what a push
        looks like to an area-growth test.
        """
        rig = (
            Rig()
            .hold(400, None)
            .push(170, 0.07, 0.22)  # sweeps in and toward the camera at once
            .hold(400, {"scale": 0.22})
        )
        self.assertEqual(rig.fires, 0)

    def test_does_not_misfire_when_a_hand_appears_already_large_and_close(self) -> None:
        rig = Rig().hold(400, None).hold(600, {"scale": 0.26})
        self.assertEqual(rig.fires, 0)

    def test_does_not_fire_before_the_minimum_charge_has_been_held(self) -> None:
        rig = (
            Rig()
            .hold(GESTURE.charge_min_ms - 150, {"scale": 0.12})
            .push(170, 0.12, 0.21)
            .hold(300, {"scale": 0.21})
        )
        self.assertEqual(rig.fires, 0)

    def test_ignores_a_slow_drift_toward_the_camera(self) -> None:
        rig = Rig().hold(800, {"scale": 0.12}).push(2500, 0.12, 0.2)
        self.assertEqual(rig.fires, 0)

    def test_does_not_fire_twice_from_one_long_shove(self) -> None:
        rig = Rig().hold(800, {"scale": 0.12}).push(500, 0.12, 0.3).hold(400, {"scale": 0.3})
        self.assertEqual(rig.fires, 1)

    def test_needs_a_fresh_charge_before_it_can_fire_again(self) -> None:
        rig = Rig().hold(800, {"scale": 0.12}).push(170, 0.12, 0.21)
        self.assertEqual(rig.fires, 1)
        # Retreat and immediately shove again without re-charging.
        rig.hold(200, {"scale": 0.12}).push(170, 0.12, 0.21)
        self.assertEqual(rig.fires, 1)
        # Charge properly and it works again.
        rig.hold(900, {"scale": 0.12}).push(170, 0.12, 0.21)
        self.assertEqual(rig.fires, 2)

    def test_scales_power_with_how_hard_the_push_was(self) -> None:
        """
        Power settles over the ~200 ms after the cast fires, because the cast
        fires on the leading edge of the push and most of the push has not
        happened yet at that moment.
        """
        soft = Rig().hold(800, {"scale": 0.12}).push(300, 0.12, 0.17).hold(250, {"scale": 0.17})
        hard = Rig().hold(800, {"scale": 0.12}).push(160, 0.12, 0.26).hold(250, {"scale": 0.26})
        self.assertEqual(soft.fires, 1)
        self.assertEqual(hard.fires, 1)
        self.assertLess(soft.rec.power, 0.5)
        self.assertGreater(hard.rec.power, 0.85)

    def test_fires_at_any_distance_from_the_camera(self) -> None:
        near = Rig().hold(800, {"scale": 0.2}).push(170, 0.2, 0.35)
        far = Rig().hold(800, {"scale": 0.06}).push(170, 0.06, 0.105)
        self.assertEqual(near.fires, 1)
        self.assertEqual(far.fires, 1)


if __name__ == "__main__":
    unittest.main()
