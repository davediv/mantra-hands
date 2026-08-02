"""
The word list, and the sequencing that makes the lesson land.

The bundled deck is packaged with the app, so it always has something to teach
with no network at all.

A tutor who cannot edit the package can point ``--deck their-list.json`` at their
own file. It is opt-in rather than a probe on every start, so nobody ever sees a
warning about a file they were never expected to have.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import ELEMENT_POS, POS_ELEMENT, Element

log = logging.getLogger("mantra_hands.deck")

BUNDLED = Path(__file__).resolve().parent / "deck.json"


@dataclass(frozen=True)
class WordEntry:
    word: str
    pos: str
    cefr: str
    gloss: str
    element: Element


def load_deck(override: Optional[str] = None) -> "Deck":
    file: Dict[str, Any] = json.loads(BUNDLED.read_text(encoding="utf-8"))
    source = "built in"

    if override:
        name = Path(override)
        try:
            candidate = json.loads(name.read_text(encoding="utf-8"))
            if len(parse(candidate)) >= 3:
                file = candidate
                source = name.name
            else:
                log.warning(
                    "%s has fewer than three usable words; keeping the built-in list", name
                )
        except Exception as err:  # noqa: BLE001 — a bad path must not stop a class
            log.warning("could not load %s (%s); using the built-in list", name, err)

    return Deck(parse(file), file.get("name", "deck"), source)


def parse(file: Dict[str, Any]) -> List[WordEntry]:
    """
    Validates and normalizes entries. A tutor editing JSON by hand will make
    typos, and one bad row must not take the app down — bad rows are dropped with
    a warning naming the row.
    """
    rows = file.get("words")
    if not isinstance(rows, list):
        rows = []
    out: List[WordEntry] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            log.warning('row %d: not an object, skipped', i)
            continue
        word = row.get("word").strip() if isinstance(row.get("word"), str) else ""
        pos = row.get("pos").strip().lower() if isinstance(row.get("pos"), str) else ""
        if not word:
            log.warning('row %d: missing "word", skipped', i)
            continue
        element = POS_ELEMENT.get(pos)
        if element is None:
            log.warning(
                '"%s": pos "%s" is not one of %s, skipped', word, pos, ", ".join(ELEMENT_POS)
            )
            continue
        out.append(
            WordEntry(
                word=word,
                pos=pos,
                cefr=row.get("cefr") if isinstance(row.get("cefr"), str) else "",
                gloss=row.get("gloss") if isinstance(row.get("gloss"), str) else "",
                element=element,
            )
        )
    return out


class Deck:
    """
    Session sequencing.

    The part-of-speech to element mapping is the lesson, so the order guarantees
    it gets taught: no two consecutive words share a part of speech while any
    alternative remains. A student sees fire, then ice, then lightning inside the
    first three casts and the rule lands without anybody explaining it.
    """

    def __init__(self, words: Sequence[WordEntry], name: str, source: str) -> None:
        if not words:
            raise ValueError("the word deck is empty")
        self.words = list(words)
        self.name = name
        self.source = source
        self._queues: List[List[WordEntry]] = [[], [], []]
        self._last_element = -1
        self._index = 0
        self._refill()

    def _refill(self) -> None:
        for q in self._queues:
            q.clear()
        for w in self.words:
            self._queues[int(w.element)].append(w)
        for q in self._queues:
            random.shuffle(q)

    def next(self) -> WordEntry:
        """Pulls the next word. Never repeats until the whole deck has been used."""
        if all(not q for q in self._queues):
            self._refill()
            self._last_element = -1
        # Prefer the biggest queue that is not the element we just used, so the
        # rotation holds without starving a large group at the end of a pass.
        best = -1
        best_len = -1
        for e in range(3):
            length = len(self._queues[e])
            if length == 0 or e == self._last_element:
                continue
            if length > best_len:
                best_len = length
                best = e
        if best < 0:
            for e in range(3):
                if self._queues[e]:
                    best = e
        entry = self._queues[best].pop()
        self._last_element = int(entry.element)
        self._index += 1
        return entry

    @property
    def size(self) -> int:
        return len(self.words)

    @property
    def served(self) -> int:
        return self._index

    def count_of(self, element: Element) -> int:
        """How many of each part of speech, for the settings readout."""
        return sum(1 for w in self.words if w.element == element)
