"""
Word recognition, behind an interface with two implementations.

Everything here runs on its own thread. The render loop never polls it, never
awaits it, and never blocks on it — at cast time the app asks :meth:`check` what
has already been heard, which is a synchronous read of state that arrived
asynchronously minutes-of-frames ago in render terms.

The browser build's native scorer is the Web Speech API, which in Chrome ships
the audio to Google's servers. This one is Vosk: a small offline model that runs
on the machine. That is a straight upgrade for the room this app was written for
— a camp with forty students and one router — and it means the "native" path no
longer has a network failure mode to fall back from.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("mantra_hands.speech")

#: 16 kHz mono is what every small offline model wants, and it is a quarter of
#: the bandwidth of 48 kHz for no loss in word accuracy on speech.
SAMPLE_RATE = 16000
BLOCK = 4000


@dataclass
class SpeechResult:
    #: What we think was said. Empty when nothing was heard at all.
    transcript: str
    #: Did it match the target word closely enough to count.
    correct: bool
    #: Was anything heard, right or wrong. Silence and a wrong word read differently.
    heard: bool


# ---------------------------------------------------------------- matching ---

_STRIP = re.compile(r"[^a-z' ]+")


def _strip(s: str) -> str:
    return _STRIP.sub(" ", s.lower()).strip()


def distance(a: str, b: str) -> int:
    """Levenshtein distance, single row. Words are short; this is not a hot path."""
    n = len(b)
    if not a:
        return n
    if n == 0:
        return len(a)
    row = list(range(n + 1))
    for i in range(1, len(a) + 1):
        prev = row[0]
        row[0] = i
        for j in range(1, n + 1):
            tmp = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1))
            prev = tmp
    return row[n]


def matches(target: str, transcript: str) -> bool:
    """
    Does the transcript contain the target word.

    Deliberately generous. These are learners, many with a first language that
    does not mark final consonants, being judged by a recognizer trained mostly
    on American English — a strict match would punish an accent rather than a
    mistake, and this app never does that. One edit per four letters of slack,
    minimum one, plus a prefix check so "confidence" said with a soft ending
    still counts.
    """
    want = _strip(target)
    if not want:
        return False
    heard = _strip(transcript)
    if not heard:
        return False
    if heard == want or want in heard:
        return True

    slack = max(1, len(want) // 4)
    for token in heard.split():
        if not token:
            continue
        if distance(token, want) <= slack:
            return True
        # A recognizer that clips the ending still heard the word.
        if len(want) >= 6 and len(token) >= len(want) - 2 and want.startswith(token[:5]):
            if distance(token[: len(want)], want) <= slack + 1:
                return True
    return False


# ------------------------------------------------------------------ vosk ---


def _model_dir() -> Optional[Path]:
    """
    Where an offline model might live. Never downloads one — this app runs on
    camp wifi and a silent 40 MB fetch on first launch is exactly the failure it
    was written to avoid.
    """
    env = os.environ.get("MANTRA_HANDS_VOSK_MODEL")
    candidates = [Path(env)] if env else []
    candidates += [
        Path(__file__).resolve().parent.parent / "models" / "vosk",
        Path.home() / ".cache" / "vosk" / "model",
        Path.home() / "vosk-model",
    ]
    for path in candidates:
        if path.is_dir() and (path / "am").is_dir():
            return path
    return None


def speech_supported() -> bool:
    """A recognizer, a model and a microphone. Anything missing means the mock."""
    try:
        import sounddevice  # noqa: F401
        import vosk  # noqa: F401
    except Exception:
        return False
    return _model_dir() is not None


class SpeechScorer:
    """The interface the app talks to. Both implementations honour it exactly."""

    kind = "mock"

    def __init__(self) -> None:
        #: Fires when a transcript lands, for live on-screen feedback.
        self.on_heard: Optional[Callable[[str, bool], None]] = None
        #: Fires when the engine gives up, so the UI can say why.
        self.on_trouble: Optional[Callable[[str], None]] = None

    def begin(self) -> None:
        """Opens the session. Call once."""

    def listen_for(self, word: str) -> None:
        """Points the scorer at a new target and clears anything heard so far."""

    def check(self) -> SpeechResult:
        """Synchronous read of what has been heard since listen_for. Never blocks."""
        return SpeechResult("", False, False)

    def end(self) -> None:
        pass


class VoskScorer(SpeechScorer):
    """
    A persistent recognition session on a background thread.

    Started once and kept alive rather than started per charge: opening an audio
    device takes a couple of hundred milliseconds and a charge only lasts about
    that long. The recognizer's partial results are folded in as they arrive, so
    a word said early in the charge is already counted by the time the push
    lands.
    """

    kind = "web"  # the "native" scorer, whatever the platform's native happens to be

    def __init__(self) -> None:
        super().__init__()
        self._target = ""
        self._buffer = ""
        self._correct = False
        self._lock = threading.Lock()
        self._audio: "queue.Queue[bytes]" = queue.Queue(maxsize=32)
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._running = False

    def begin(self) -> None:
        try:
            import sounddevice as sd
            import vosk
        except Exception:
            self._trouble("no recognizer")
            return

        model_path = _model_dir()
        if model_path is None:
            self._trouble("no model")
            return

        try:
            vosk.SetLogLevel(-1)
            self._model = vosk.Model(str(model_path))
            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK,
                dtype="int16",
                channels=1,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as err:  # noqa: BLE001
            log.warning("microphone unavailable: %s", err)
            self._trouble("not-allowed")
            return

        self._running = True
        self._thread = threading.Thread(target=self._recognise, name="speech", daemon=True)
        self._thread.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        # Called on the audio thread. Never do work here: drop the block rather
        # than stall the device.
        try:
            self._audio.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _recognise(self) -> None:
        import vosk

        rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(False)
        while self._running:
            try:
                block = self._audio.get(timeout=0.25)
            except queue.Empty:
                continue
            import json

            if rec.AcceptWaveform(block):
                text = json.loads(rec.Result()).get("text", "")
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
            if text:
                self._fold(text)

    def _fold(self, text: str) -> None:
        with self._lock:
            self._buffer = f"{self._buffer} {text}"[-240:]
            if not self._correct and self._target and matches(self._target, self._buffer):
                self._correct = True
            correct = self._correct
        if self.on_heard:
            self.on_heard(text.strip(), correct)

    def _trouble(self, reason: str) -> None:
        if self.on_trouble:
            self.on_trouble(reason)

    def listen_for(self, word: str) -> None:
        with self._lock:
            self._target = word
            self._buffer = ""
            self._correct = False

    def check(self) -> SpeechResult:
        with self._lock:
            transcript = self._buffer.strip()
            return SpeechResult(transcript, self._correct, bool(transcript))

    def end(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=0.5)


# -------------------------------------------------------------------- mock ---


class MockScorer(SpeechScorer):
    """
    The offline scorer. The whole app is playable and demoable with this and no
    microphone, no model, and no permission prompt — which is how it will
    actually be shown in a room with forty students and one router.

    The miss pattern is deterministic rather than random, for two reasons: a
    three-cast recording always lands, and the fizzle is still guaranteed to show
    up early enough for anyone to see what a wrong word looks like.
    """

    kind = "mock"

    def __init__(self) -> None:
        super().__init__()
        #: Casts that miss: the 4th, then every 7th after it.
        self._attempt = 0
        self._transcript = ""
        self._correct = False
        self._timer: Optional[threading.Timer] = None

    def listen_for(self, word: str) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._transcript = ""
        self._correct = False
        self._attempt += 1

        will_miss = self._attempt >= 4 and (self._attempt - 4) % 7 == 0
        # Real recognizers land a result 400-900 ms after you speak. Anything
        # slower than the charge itself would make casting feel like a dice roll.
        delay = (420 + random.random() * 460) / 1000.0
        self._timer = threading.Timer(delay, self._land, args=(word, will_miss))
        self._timer.daemon = True
        self._timer.start()

    def _land(self, word: str, will_miss: bool) -> None:
        self._timer = None
        self._transcript = near_miss(word) if will_miss else word
        self._correct = not will_miss
        if self.on_heard:
            self.on_heard(self._transcript, self._correct)

    def check(self) -> SpeechResult:
        return SpeechResult(self._transcript, self._correct, bool(self._transcript))

    def end(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def near_miss(word: str) -> str:
    """A plausible mishearing: the kind of thing a recognizer actually returns."""
    w = word.lower()
    if len(w) <= 4:
        return f"{w}s"
    cut = max(3, len(w) - 3)
    return w[:cut]


def create_scorer(prefer_native: bool) -> SpeechScorer:
    return VoskScorer() if prefer_native and speech_supported() else MockScorer()
