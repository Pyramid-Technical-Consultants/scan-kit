"""Application-level settings persisted outside the session data directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_SETTINGS_DIR = Path.home() / ".scan-kit"
_FILENAME = "app_settings.json"


@dataclass
class AppSettings:
    """User preferences for scan-kit workflows."""

    config_dir: str | None = None
    last_opened_xml: str | None = None
    hide_unused_map2map_xml: bool = False
    last_main_tab: str | None = None
    last_report_dir: str | None = None

    @classmethod
    def settings_path(cls) -> Path:
        return _SETTINGS_DIR / _FILENAME

    def save(self) -> None:
        _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.settings_path().write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> AppSettings:
        path = cls.settings_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        return cls(
            config_dir=_optional_str(raw.get("config_dir")),
            last_opened_xml=_optional_str(raw.get("last_opened_xml")),
            hide_unused_map2map_xml=bool(raw.get("hide_unused_map2map_xml", False)),
            last_main_tab=_optional_str(raw.get("last_main_tab")),
            last_report_dir=_optional_str(raw.get("last_report_dir")),
        )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
