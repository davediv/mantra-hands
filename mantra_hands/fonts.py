"""
The type, resolved from whatever the machine already has.

The browser build used a system font stack rather than a webfont, because a
40 KB font request that times out on camp wifi means no word on screen at all.
The same reasoning survives the port for a different reason: bundling a display
face would put a licence and a megabyte into a package whose whole point is that
it runs from a USB stick. On Android the CSS stack resolved to Roboto Condensed;
here macOS lands on DIN Condensed, Linux on DejaVu Sans Condensed, Windows on
Arial Narrow.

Every lookup is cached, because PIL parses the file on each ``truetype`` call and
the HUD asks for the same face eight times a second.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

from PIL import ImageFont

#: Condensed and heavy, for the huge in-canvas word. Index is for .ttc bundles.
DISPLAY_STACK: Sequence[Tuple[str, int]] = (
    ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf", 0),
    ("/System/Library/Fonts/Avenir Next Condensed.ttc", 2),
    ("/System/Library/Fonts/Supplemental/Impact.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("C:/Windows/Fonts/ARIALNB.TTF", 0),
    ("C:/Windows/Fonts/impact.ttf", 0),
)

UI_STACK: Sequence[Tuple[str, int]] = (
    ("/System/Library/Fonts/SFNS.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ("C:/Windows/Fonts/segoeui.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
)

UI_BOLD_STACK: Sequence[Tuple[str, int]] = (
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ("C:/Windows/Fonts/segoeuib.ttf", 0),
)

MONO_STACK: Sequence[Tuple[str, int]] = (
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/Monaco.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ("C:/Windows/Fonts/consola.ttf", 0),
)

STACKS = {
    "display": DISPLAY_STACK,
    "ui": UI_STACK,
    "ui-bold": UI_BOLD_STACK,
    "mono": MONO_STACK,
}


@lru_cache(maxsize=64)
def load(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """The first face in ``kind``'s stack that this machine actually has."""
    for path, index in STACKS[kind]:
        if not Path(path).is_file():
            continue
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    # PIL's built-in bitmap face. Ugly at 180 px, but the app still runs and the
    # word is still legible, which is the only thing that matters here.
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def display_name() -> str:
    """Which face the display stack resolved to, for the HUD and the README."""
    for path, _ in DISPLAY_STACK:
        if Path(path).is_file():
            return Path(path).stem
    return "PIL default"
