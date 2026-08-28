"""Regression tests for latency-sensitive overlay state updates."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
