"""Resolve map2map paths inside a configuration folder."""

from __future__ import annotations

from pathlib import Path

from scan_kit.common.session_source import resolve_session_source

_DEVICES_REL_CANDIDATES = (
    "map2map/devices.xml",
    "devices.xml",
    "config/map2map/devices.xml",
)


def resolve_devices_xml_path(config_root: Path) -> Path | None:
    """Return ``devices.xml`` under *config_root*, if present."""
    root = config_root.resolve()
    for rel in _DEVICES_REL_CANDIDATES:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    for path in sorted(root.rglob("devices.xml")):
        if "map2map" in path.parts:
            return path
    return None


def resolve_session_config_dir(session_id: str, base_dir: str | Path) -> Path | None:
    """Return the on-disk configuration folder for a session, if available."""
    source = resolve_session_source(session_id, base_dir)
    if source is None or source.kind != "directory":
        return None
    for candidate in (source.path / "config", source.path):
        if candidate.is_dir() and resolve_devices_xml_path(candidate) is not None:
            return candidate.resolve()
    return None
