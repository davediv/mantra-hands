"""
Streak, personal best, words completed, and the settings — one JSON file,
written on change rather than on a timer.

Storage can fail: a read-only home directory, a locked-down lab machine, a full
disk. None of that is worth losing a session over, so every access is guarded and
the app simply runs without persistence if it has to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..config import STORAGE_PATH


@dataclass
class Settings:
    mirror: bool = True
    mic: bool = False
    low_power: bool = False
    #: Effect intensity as a percentage, 40-150.
    intensity: int = 100


class Progress:
    def __init__(self) -> None:
        self.streak = 0
        self.best = 0
        self.completed = 0
        self.settings = Settings()

        try:
            saved = json.loads(STORAGE_PATH.read_text(encoding="utf-8"))
            self.streak = _num(saved.get("streak"), 0)
            self.best = _num(saved.get("best"), 0)
            self.completed = _num(saved.get("completed"), 0)
            for key, value in (saved.get("settings") or {}).items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            self.settings.intensity = max(40, min(150, _num(self.settings.intensity, 100)))
        except Exception:
            # Unreadable or unavailable. Defaults stand.
            pass

    def correct(self) -> bool:
        """:returns: True when this cast set a new personal best."""
        self.streak += 1
        self.completed += 1
        record = self.streak > self.best
        if record:
            self.best = self.streak
        self.save()
        return record

    def wrong(self) -> None:
        self.streak = 0
        self.save()

    def save(self) -> None:
        try:
            STORAGE_PATH.write_text(
                json.dumps(
                    {
                        "streak": self.streak,
                        "best": self.best,
                        "completed": self.completed,
                        "settings": asdict(self.settings),
                    }
                ),
                encoding="utf-8",
            )
        except Exception:
            # Nothing to be done, and nothing worth interrupting the session for.
            pass


def _num(v: object, fallback: int) -> int:
    return int(v) if isinstance(v, (int, float)) and v == v else fallback
