"""Speech capability checks should keep the no-model startup path cheap."""

from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

from mantra_hands.audio.speech import speech_supported


class SpeechSupportTests(unittest.TestCase):
    def test_missing_model_does_not_import_native_audio_dependencies(self) -> None:
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
            if name in ("sounddevice", "vosk"):
                raise AssertionError(f"unexpected cold-start import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("mantra_hands.audio.speech._model_dir", return_value=None), patch(
            "builtins.__import__", side_effect=guarded_import
        ):
            self.assertFalse(speech_supported())


if __name__ == "__main__":
    unittest.main()
