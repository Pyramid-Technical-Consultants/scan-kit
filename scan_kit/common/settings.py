"""Global view settings, persisted as ``settings.json`` in the data directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

_FILENAME = "settings.json"

CALIBRATION_MODES = ("off", "per_session", "constrained")


@dataclass
class ViewSettings:
    """Lightweight bag of global settings passed from the TUI to view subprocesses."""

    bg_subtract: bool = False
    calibration_mode: str = "off"
    cal_factors: dict[str, float] | None = None
    selected_sessions: list[str] = field(default_factory=list)

    @property
    def auto_calibrate(self) -> bool:
        return self.calibration_mode != "off"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> ViewSettings:
        raw = json.loads(s)
        return cls(**cls._clean(raw))

    def save(self, base_dir: str | Path) -> None:
        path = Path(base_dir) / _FILENAME
        d = asdict(self)
        d.pop("cal_factors", None)  # runtime-only, not persisted
        path.write_text(
            json.dumps(d, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, base_dir: str | Path) -> ViewSettings:
        path = Path(base_dir) / _FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return cls()
        return cls(**cls._clean(raw))

    @classmethod
    def _clean(cls, raw: dict) -> dict:
        """Normalise a raw JSON dict into valid constructor kwargs."""
        known = set(cls.__dataclass_fields__)
        out = {k: v for k, v in raw.items() if k in known}
        # Backward compat: old bool auto_calibrate -> calibration_mode
        if "auto_calibrate" in raw and "calibration_mode" not in raw:
            out["calibration_mode"] = "per_session" if raw["auto_calibrate"] else "off"
            out.pop("auto_calibrate", None)
        if out.get("calibration_mode") not in CALIBRATION_MODES:
            out["calibration_mode"] = "off"
        sel = out.get("selected_sessions")
        if isinstance(sel, list):
            out["selected_sessions"] = [str(s) for s in sel]
        else:
            out.pop("selected_sessions", None)
        return out
