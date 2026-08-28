"""
Clip capture.

The browser build recorded the canvas straight off its own compositor stream —
no second render pass, no readback, no extra copy per frame. Outside a browser
there is no such stream, so this reads the composited frame back from the
export-resolution target and feeds it to an encoder. That readback is the one
real cost the port adds, and it only runs while a clip is actually recording.

Everything else survives: before the countdown the render target switches to a
true 1080x1920 and the window letterboxes, so the framing during the countdown is
exactly the framing that gets exported. That resize is one reallocation, not a
render pass. Low power records 720x1280. The 2D chrome is not in the file, for
the same reason the DOM was never in the browser's captureStream.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import RECORD

ENCODE_QUEUE_FRAMES = 2


class Recorder:
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.state = "idle"
        self.supported = True
        self.output_dir = output_dir or Path.cwd()

        #: Asks the app to switch the render target to the export framing, and back.
        self.set_framing: Optional[Callable[[bool], None]] = None
        self.on_message: Optional[Callable[[str], None]] = None
        #: What the app should put on screen during the 3-2-1.
        self.countdown_text: Optional[str] = None

        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._started_at = 0.0
        self._countdown_at = 0.0
        self._last_write = 0.0
        self._frames = 0
        self._dropped_frames = 0

        # Conversion and codec writes can consume most of a 16.6 ms frame on a
        # slower machine. A bounded producer/consumer queue keeps that work off
        # the render thread without allowing recording memory to grow unchecked.
        self._encode_queue: "queue.Queue[bytes]" = queue.Queue(
            maxsize=ENCODE_QUEUE_FRAMES
        )
        self._encode_thread: Optional[threading.Thread] = None
        self._encoding = False
        self._encode_error: Optional[Exception] = None
        self._encode_size = (0, 0)

    # ---------------------------------------------------------------- control ---

    def toggle(self, now_ms: float) -> None:
        if self.state == "idle":
            self._begin(now_ms)
        elif self.state == "recording":
            self.stop()
        elif self.state == "countdown":
            self.cancel()

    def _begin(self, now_ms: float) -> None:
        """3-2-1, then roll."""
        self.state = "countdown"
        self._countdown_at = now_ms
        self.countdown_text = str(RECORD.countdown_seconds)
        if self.set_framing:
            self.set_framing(True)

    def cancel(self) -> None:
        self.countdown_text = None
        self._reset()

    def stop(self) -> None:
        if self.state != "recording":
            return
        self._finish()

    @property
    def framing(self) -> bool:
        """True while the export framing is in force — the app pauses resizing then."""
        return self.state in ("countdown", "recording")

    # ------------------------------------------------------------------ frame ---

    def update(self, now_ms: float, renderer) -> None:  # noqa: ANN001
        """Called once per rendered frame, after the composite."""
        if self.state == "countdown":
            elapsed = (now_ms - self._countdown_at) / 1000.0
            left = RECORD.countdown_seconds - int(elapsed)
            if left <= 0:
                self.countdown_text = None
                self._roll(now_ms, renderer)
            else:
                self.countdown_text = str(left)
            return

        if self.state != "recording" or self._writer is None:
            return
        if self._encode_error is not None:
            self.stop()
            return

        # The window runs at display rate; the file is 30 fps. Writing only when
        # a slot is due keeps the clip's duration honest instead of stretching it.
        if now_ms - self._last_write < 1000.0 / RECORD.fps:
            if now_ms - self._started_at >= RECORD.max_seconds * 1000.0:
                self.stop()
            return
        self._last_write = now_ms

        data = renderer.read_present()
        if data is not None:
            self._submit_frame(data)

        if now_ms - self._started_at >= RECORD.max_seconds * 1000.0:
            self.stop()

    def _roll(self, now_ms: float, renderer) -> None:  # noqa: ANN001
        width, height = renderer.buffer_w, renderer.buffer_h
        for fourcc, ext in RECORD.fourcc_candidates:
            path = self._next_path(ext)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*fourcc), RECORD.fps, (width, height)
            )
            if writer.isOpened():
                self._writer = writer
                self._path = path
                break
            writer.release()
            path.unlink(missing_ok=True)

        if self._writer is None:
            self._message("No video encoder available — could not start recording.")
            self._reset()
            return

        self._started_at = now_ms
        self._last_write = 0.0
        self._frames = 0
        self._dropped_frames = 0
        self._start_encoder(width, height)
        self.state = "recording"

    def _finish(self) -> None:
        self._stop_encoder()

        path = self._path
        frames = self._frames
        error = self._encode_error
        self._reset()

        if error is not None:
            if path is not None and frames == 0:
                path.unlink(missing_ok=True)
            self._message("Recording stopped because the video encoder failed.")
            return
        if path is None or frames == 0:
            if path is not None:
                path.unlink(missing_ok=True)
            self._message("Nothing got recorded. Coba lagi?")
            return
        self._message(f"Saved {path.name} — {frames / RECORD.fps:.0f}s")

    def _reset(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.state = "idle"
        self.countdown_text = None
        self._path = None
        if self.set_framing:
            self.set_framing(False)

    def close(self) -> None:
        """Finishes an active recording before the GL context is destroyed."""
        if self.state == "recording":
            self._finish()
        elif self.state == "countdown":
            self.cancel()

    # --------------------------------------------------------------- encoding ---

    def _start_encoder(self, width: int, height: int) -> None:
        writer = self._writer
        if writer is None:
            return
        self._encode_queue = queue.Queue(maxsize=ENCODE_QUEUE_FRAMES)
        self._encode_size = (width, height)
        self._encode_error = None
        self._encoding = True
        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            args=(writer,),
            name="record-encode",
            daemon=True,
        )
        self._encode_thread.start()

    def _submit_frame(self, data: bytes) -> bool:
        """Queues one readback without ever waiting on the codec."""
        try:
            self._encode_queue.put_nowait(data)
            return True
        except queue.Full:
            self._dropped_frames += 1
            return False

    def _encode_loop(self, writer) -> None:  # noqa: ANN001
        width, height = self._encode_size
        try:
            while self._encoding or not self._encode_queue.empty():
                try:
                    data = self._encode_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                # GL hands back bottom-up RGB; the encoder wants top-down BGR.
                writer.write(cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR))
                self._frames += 1
        except Exception as err:  # noqa: BLE001
            self._encode_error = err
            self._encoding = False
        finally:
            writer.release()

    def _stop_encoder(self) -> None:
        self._encoding = False
        thread = self._encode_thread
        self._encode_thread = None
        if thread is not None and thread is not threading.current_thread():
            # At most ENCODE_QUEUE_FRAMES are pending, so normal shutdown is a
            # small bounded flush. Waiting here keeps the exported file valid.
            thread.join()
        # The worker owns and releases the writer once it starts.
        self._writer = None

    def _next_path(self, ext: str) -> Path:
        n = 1
        while True:
            path = self.output_dir / f"mantra-hands-{n:03d}{ext}"
            if not path.exists():
                return path
            n += 1

    def _message(self, text: str) -> None:
        if self.on_message:
            self.on_message(text)
