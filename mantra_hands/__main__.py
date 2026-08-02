"""
Mantra Hands.

Charge a spell with an open palm, say the English word, push at the camera.
The part of speech picks the element — that mapping is the lesson.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence, Tuple


def _size(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return max(320, int(w)), max(240, int(h))
    except Exception as err:  # noqa: BLE001
        raise argparse.ArgumentTypeError("size must look like 1280x720") from err


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mantra-hands",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "keys\n"
            "  d      performance readout      m   mirror the camera\n"
            "  s      settings                 p   low power mode\n"
            "  r      record a clip            [ ] effect intensity\n"
            "  q      quit\n"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="scripted hands and a synthetic camera — no webcam, no model",
    )
    parser.add_argument("--deck", metavar="PATH", help="a word list to use instead of the built-in one")
    parser.add_argument("--camera", type=int, default=0, metavar="N", help="camera index (default 0)")
    parser.add_argument("--size", type=_size, default=(1280, 720), help="window size, e.g. 1280x720")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="allow the microphone. Needs an offline Vosk model; without one the app "
        "scores you offline, which is fully playable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log deck and speech warnings")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    # Imported after argument parsing so that --help does not pay for GL, OpenCV
    # and 300 MB of MediaPipe.
    from .app import App

    try:
        app = App(
            demo=args.demo,
            deck_path=args.deck,
            camera_index=args.camera,
            fullscreen=args.fullscreen,
            width=args.size[0],
            height=args.size[1],
        )
    except Exception as err:  # noqa: BLE001
        # Nothing is on screen yet, so this is the one failure that has to go to
        # the terminal.
        print(f"Mantra Hands could not start: {err}", file=sys.stderr)
        return 1

    if args.mic:
        # The microphone is opt-in and stays opt-in: this flag sets the stored
        # preference, and the explainer still asks before anything opens a device.
        app.progress.settings.mic = True

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
