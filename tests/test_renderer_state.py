"""Pure state checks for render-pass scheduling decisions."""

from __future__ import annotations

import unittest

import numpy as np

from mantra_hands.render.renderer import MODE_BURST, MODE_CHARGE, Renderer


class ParticleSchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = Renderer.__new__(Renderer)
        self.renderer._u_mode = np.array([MODE_CHARGE, MODE_CHARGE], dtype=np.int32)
        self.renderer._u_rune_alpha = np.zeros(2, dtype=np.float32)

    def test_idle_particles_do_not_need_simulation(self) -> None:
        self.assertFalse(self.renderer._particles_active())

    def test_visible_charge_needs_simulation(self) -> None:
        self.renderer._u_rune_alpha[1] = 0.5
        self.assertTrue(self.renderer._particles_active())

    def test_burst_keeps_simulating_after_hand_leaves(self) -> None:
        self.renderer._u_mode[0] = MODE_BURST
        self.assertTrue(self.renderer._particles_active())


if __name__ == "__main__":
    unittest.main()
