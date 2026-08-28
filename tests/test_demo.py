"""Regression coverage for the allocation-free synthetic camera path."""

from __future__ import annotations

import colorsys
import unittest
from unittest.mock import patch

import numpy as np

from mantra_hands.demo import DemoCamera


class DemoCameraTests(unittest.TestCase):
    def test_gradient_matches_three_stop_reference_and_reuses_buffers(self) -> None:
        width, height = 100, 80
        camera = DemoCamera(width, height, fps=30)
        camera._period = 0.0
        work_id = id(camera._lut_work)

        with patch("mantra_hands.demo.now_ms", return_value=1234.0):
            frame = camera.read()
            again = camera.read()

        self.assertIs(frame, camera._frame)
        self.assertIs(again, camera._frame)
        self.assertEqual(id(camera._lut_work), work_id)

        # Away from the marker, labels, and grid, compare with the original
        # three-stop gradient equation.
        x, y = 75, 25
        ramp = ((x / (width - 1)) + (y / (height - 1))) * 0.5
        lower = min(1.0, ramp * 2.0)
        upper = max(0.0, ramp * 2.0 - 1.0)
        rgb = colorsys.hls_to_rgb(((1.234 * 12.0) % 360.0) / 360.0, 0.22, 0.30)
        c0 = np.array((0x2E, 0x1C, 0x14), dtype=np.float32) / 255.0
        c1 = np.array((rgb[2], rgb[1], rgb[0]), dtype=np.float32)
        c2 = np.array((0x1A, 0x0F, 0x0B), dtype=np.float32) / 255.0
        expected = ((c0 + (c1 - c0) * lower + (c2 - c1) * upper) * 255.0).astype(
            np.uint8
        )
        np.testing.assert_allclose(frame[y, x], expected, atol=1)


if __name__ == "__main__":
    unittest.main()
