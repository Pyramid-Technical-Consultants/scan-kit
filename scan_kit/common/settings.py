"""Global view settings, persisted as ``settings.json`` in the data directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

_FILENAME = "settings.json"


@dataclass
class ViewSettings:
    """Lightweight bag of global settings passed from the TUI to view subprocesses."""

    auto_calibrate: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> ViewSettings:
        raw = json.loads(s)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, base_dir: str | Path) -> None:
        path = Path(base_dir) / _FILENAME
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, base_dir: str | Path) -> ViewSettings:
        path = Path(base_dir) / _FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})
