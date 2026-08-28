"""Tests for GL helpers that do not require a window or driver."""

from __future__ import annotations

import unittest

import numpy as np

from mantra_hands.render.gl import Program


class RecordingUniform:
    def __init__(self) -> None:
        self.payloads = []

    def write(self, payload) -> None:  # noqa: ANN001
        self.payloads.append(payload)


class UniformUploadTests(unittest.TestCase):
    def test_array_uploads_pass_the_numpy_buffer_without_copying(self) -> None:
        uniform = RecordingUniform()
        program = Program.__new__(Program)
        program._locs = {"u": uniform}

        values = (
            np.arange(2, dtype=np.float32),
            np.arange(2, dtype=np.int32),
            np.arange(6, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )
        program.floatv("u", values[0])
        program.intv("u", values[1])
        program.vec3v("u", values[2])
        program.mat3("u", values[3])

        self.assertEqual(len(uniform.payloads), len(values))
        for payload, expected in zip(uniform.payloads, values):
            self.assertIs(payload, expected)


if __name__ == "__main__":
    unittest.main()
