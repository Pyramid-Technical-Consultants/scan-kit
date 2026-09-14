"""Application-level settings persisted in the machine-local SQLite store."""

from __future__ import annotations

from dataclasses import dataclass
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
    last_plan_synthesis_save_dir: str | None = None
    last_rci_host: str | None = None
    last_plan_runner_file_dir: str | None = None

    @classmethod
    def settings_path(cls) -> Path:
        """Legacy JSON path under ``~/.scan-kit`` (imported once, then unused)."""
        return _SETTINGS_DIR / _FILENAME

    @classmethod
    def from_mapping(cls, raw: dict) -> AppSettings:
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
            last_plan_synthesis_save_dir=_optional_str(
                raw.get("last_plan_synthesis_save_dir")
            ),
            last_rci_host=_optional_str(raw.get("last_rci_host")),
            last_plan_runner_file_dir=_optional_str(
                raw.get("last_plan_runner_file_dir")
            ),
        )

    def save(self) -> None:
        from .user_store import save_app_settings

        save_app_settings(self)

    @classmethod
    def load(cls) -> AppSettings:
        from .user_store import load_app_settings

        return load_app_settings()


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
