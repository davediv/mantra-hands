"""Recording stays correct while codec work runs off the render thread."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from mantra_hands.ui.recorder import Recorder


class CollectingWriter:
    def __init__(self) -> None:
        self.frames = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class BlockingWriter(CollectingWriter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.unblock = threading.Event()

    def write(self, frame: np.ndarray) -> None:
        self.entered.set()
        self.unblock.wait(timeout=2.0)
        super().write(frame)


class RecorderEncodingTests(unittest.TestCase):
    def test_worker_flips_rgb_readback_into_top_down_bgr(self) -> None:
        recorder = Recorder()
        writer = CollectingWriter()
        recorder._writer = writer
        recorder._start_encoder(2, 2)

        # GL readback order: bottom row first, RGB channels.
        rgb_bottom_up = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[10, 20, 30], [40, 50, 60]],
            ],
            dtype=np.uint8,
        )
        self.assertTrue(recorder._submit_frame(rgb_bottom_up.tobytes()))
        recorder._stop_encoder()

        expected = np.array(
            [
                [[30, 20, 10], [60, 50, 40]],
                [[3, 2, 1], [6, 5, 4]],
            ],
            dtype=np.uint8,
        )
        self.assertEqual(len(writer.frames), 1)
        np.testing.assert_array_equal(writer.frames[0], expected)
        self.assertTrue(writer.released)

    def test_full_queue_drops_work_instead_of_blocking(self) -> None:
        recorder = Recorder()
        writer = BlockingWriter()
        recorder._writer = writer
        recorder._start_encoder(1, 1)

        self.assertTrue(recorder._submit_frame(bytes((1, 2, 3))))
        self.assertTrue(writer.entered.wait(timeout=1.0))
        self.assertTrue(recorder._submit_frame(bytes((4, 5, 6))))
        self.assertTrue(recorder._submit_frame(bytes((7, 8, 9))))
        self.assertFalse(recorder._submit_frame(bytes((10, 11, 12))))
        self.assertEqual(recorder._dropped_frames, 1)

        writer.unblock.set()
        recorder._stop_encoder()
        self.assertTrue(writer.released)


if __name__ == "__main__":
    unittest.main()
