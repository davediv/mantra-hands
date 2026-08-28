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


if __name__ == "__main__":
    unittest.main()
