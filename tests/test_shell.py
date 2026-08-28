"""Regression tests for latency-sensitive overlay state updates."""

from __future__ import annotations

import unittest

from PIL import Image

from mantra_hands.ui.shell import Shell


class ShellDirtyRegionTests(unittest.TestCase):
    def test_hint_change_uses_partial_texture_update(self) -> None:
        shell = Shell.__new__(Shell)
        shell._hint = "Open palm · say the word · push"
        shell._dirty = False
        shell._hint_dirty = False

        shell.set_hint("PUSH!")

        self.assertFalse(shell._dirty)
        self.assertTrue(shell._hint_dirty)

    def test_score_change_only_dirties_topbar(self) -> None:
        shell = Shell.__new__(Shell)
        shell._combo = shell._streak = shell._best = 0
        shell._dirty = False
        shell._topbar_dirty = False

        shell.set_counts(1, 2, 3)

        self.assertFalse(shell._dirty)
        self.assertTrue(shell._topbar_dirty)

    def test_heard_feedback_only_dirties_its_band(self) -> None:
        shell = Shell.__new__(Shell)
        shell._dirty = False
        shell._heard_dirty = False

        shell.show_heard("borrow", True, 100.0)

        self.assertFalse(shell._dirty)
        self.assertTrue(shell._heard_dirty)

    def test_unchanged_hint_does_not_request_an_update(self) -> None:
        shell = Shell.__new__(Shell)
        shell._hint = "PUSH!"
        shell._dirty = False
        shell._hint_dirty = False

        shell.set_hint("PUSH!")

        self.assertFalse(shell._dirty)
        self.assertFalse(shell._hint_dirty)

    def test_overlay_rows_are_grouped_without_spanning_large_empty_gaps(self) -> None:
        shell = Shell.__new__(Shell)
        shell._image = Image.new("RGBA", (20, 100), (0, 0, 0, 0))
        shell._image.paste((255, 255, 255, 255), (0, 5, 20, 12))
        shell._image.paste((255, 255, 255, 255), (0, 80, 20, 90))
        shell._h = 100
        shell._draw_regions = []

        shell._update_draw_regions()

        self.assertEqual(shell._draw_regions, [(5, 12), (80, 90)])

    def test_removed_center_component_drops_its_scissor_band(self) -> None:
        shell = Shell.__new__(Shell)
        shell._w = 1280
        shell._h = 720
        shell._empty = True
        shell._countdown = None
        shell._heard = ""
        shell._word = object()
        shell._toast = ""

        shell._refresh_component_regions()
        center = (shell._center_box()[1], shell._center_box()[3])
        self.assertIn(center, shell._draw_regions)

        shell._empty = False
        shell._refresh_component_regions()
        self.assertNotIn(center, shell._draw_regions)


if __name__ == "__main__":
    unittest.main()
