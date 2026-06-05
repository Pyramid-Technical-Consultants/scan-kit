"""Resolve map2map paths inside a configuration folder."""

from __future__ import annotations

from pathlib import Path

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
