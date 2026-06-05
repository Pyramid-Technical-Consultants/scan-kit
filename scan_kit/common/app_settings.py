"""Application-level settings persisted outside the session data directory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
    window_width: int | None = None
    window_height: int | None = None
    window_x: int | None = None
    window_y: int | None = None
    last_report_dir: str | None = None
    last_plan_synthesis_save_dir: str | None = None
    last_report_author: str | None = None
    last_report_organization: str | None = None
    last_report_views: list[str] = field(default_factory=list)

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
            window_width=_optional_int(raw.get("window_width")),
            window_height=_optional_int(raw.get("window_height")),
            window_x=_optional_int(raw.get("window_x")),
            window_y=_optional_int(raw.get("window_y")),
            last_report_dir=_optional_str(raw.get("last_report_dir")),
            last_plan_synthesis_save_dir=_optional_str(
                raw.get("last_plan_synthesis_save_dir")
            ),
            last_report_author=_optional_str(raw.get("last_report_author")),
            last_report_organization=_optional_str(raw.get("last_report_organization")),
            last_report_views=_optional_str_list(raw.get("last_report_views")),
        )


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out
