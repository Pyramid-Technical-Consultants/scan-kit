"""Map2map XML fields that are never consumed by the Pyramid map2map library.

Derived from ``scan_dose/map2map`` (``scan_magnet.cpp``, ``ion_chamber.cpp``, etc.).
Only list items with no read path in that codebase — safe to hide in the config editor.

See scan_magnet ``sc_gain_conversion``: ``m2`` is loaded but never referenced after parse.
``e0``/``e1``/``e2``/``c1``/``c3``/``d2``/``d4`` never appear in any ``<xmlattr>`` getter.
``b0``/``b1`` on ``units="volts"`` rows are stored but cross-coupling reads the
``kilogauss`` row only (``engine.cpp`` + ``get_cross_correction_factor``).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# gain_conversion @attributes — present in room XML, never parsed by map2map
GAIN_CONVERSION_DEAD_ATTRS = frozenset(
    {"e0", "e1", "e2", "c1", "c3", "d2", "d4", "m2"}
)

# Parsed on volts rows but never read (kG row supplies cross-coupling factors)
GAIN_CONVERSION_VOLTS_ONLY_DEAD_ATTRS = frozenset({"b0", "b1"})

# ion_chamber child elements — only zero_offset_at_iso_mm is read
ION_CHAMBER_DEAD_CHILD_TAGS = frozenset({"zero_offset_mm"})


def is_map2map_config_path(path: str | None) -> bool:
    if not path:
        return False
    try:
        return "map2map" in Path(path).resolve().parts
    except (OSError, ValueError):
        normalized = path.replace("\\", "/").lower()
        return "/map2map/" in normalized or normalized.endswith("/map2map")


def should_hide_map2map_attribute(
    element: ET.Element,
    attr: str,
    *,
    ancestor_tags: tuple[str, ...] = (),
) -> bool:
    """True when *attr* on *element* is known dead in map2map."""
    if element.tag != "gain_conversion":
        return False
    if attr in GAIN_CONVERSION_DEAD_ATTRS:
        return True
    if attr in GAIN_CONVERSION_VOLTS_ONLY_DEAD_ATTRS:
        return element.get("units", "").lower() == "volts"
    return False


def should_hide_map2map_child(
    parent: ET.Element,
    child_tag: str,
    *,
    ancestor_tags: tuple[str, ...] = (),
) -> bool:
    """True when a child element under *parent* is never read by map2map."""
    if parent.tag == "ion_chamber" and child_tag in ION_CHAMBER_DEAD_CHILD_TAGS:
        return True
    return False


def filter_attribute_names(
    element: ET.Element,
    names: list[str],
    *,
    ancestor_tags: tuple[str, ...] = (),
) -> list[str]:
    return [
        name
        for name in names
        if not should_hide_map2map_attribute(element, name, ancestor_tags=ancestor_tags)
    ]


def filter_child_tags(
    parent: ET.Element,
    tags: list[str],
    *,
    ancestor_tags: tuple[str, ...] = (),
) -> list[str]:
    return [
        tag
        for tag in tags
        if not should_hide_map2map_child(parent, tag, ancestor_tags=ancestor_tags)
    ]
