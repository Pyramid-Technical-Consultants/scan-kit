"""What the Pyramid map2map library actually does with each XML field.

One catalog, derived from ``scan_dose/map2map`` (``ion_chamber.cpp``,
``scan_magnet.cpp``, ``engine.cpp``, ``devices.cpp``).  It drives two things in the
config editor: which fields to hide as dead, and the tooltip explaining every field
whose XML name is misleading about its runtime effect.

The headline case is magnification.  ``engine::load_configuration`` finishes with
``set_mag_factors(m_geom.virtual_sad)``, which overwrites the per-chamber
``source_to_axis_distance_mm`` seed for every ion chamber.  Runtime magnification is
always ``source_to_isocenter_distance / source_to_device_distance_mm``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Read and used in conversion math.
ROLE_EFFECTIVE = "effective"
# Read into a member, then overwritten before anything reads it.
ROLE_OVERWRITTEN = "overwritten"
# Parsed and range-checked at load, then never read again.
ROLE_VALIDATED_ONLY = "validated_only"
# No read path at all, or parsed only to be logged.
ROLE_UNUSED = "unused"
# Effective, but on the scan-magnet side rather than the ion-chamber side.
ROLE_MAGNET_EFFECTIVE = "magnet_effective"

_ROLE_PREFIX = {
    ROLE_EFFECTIVE: "Used by map2map",
    ROLE_MAGNET_EFFECTIVE: "Used by map2map (scan magnet)",
    ROLE_OVERWRITTEN: "Parsed, then overwritten — no runtime effect",
    ROLE_VALIDATED_ONLY: "Range-checked at load, then never read",
    ROLE_UNUSED: "Never read by map2map",
}


@dataclass(frozen=True)
class Map2MapField:
    """One map2map XML field, the element it lives on, and its runtime role."""

    name: str
    scope: str
    role: str
    note: str
    cpp_site: str
    units_filter: str | None = None
    """When set, the role only applies to rows whose ``units`` attribute matches."""

    @property
    def is_dead(self) -> bool:
        return self.role == ROLE_UNUSED

    def tooltip(self) -> str:
        return f"{_ROLE_PREFIX[self.role]}: {self.note} ({self.cpp_site})"


MAP2MAP_FIELDS: tuple[Map2MapField, ...] = (
    # --- ion chamber geometry -------------------------------------------------
    Map2MapField(
        "source_to_device_distance_mm",
        "ion_chamber",
        ROLE_EFFECTIVE,
        "denominator of the strip-to-isocenter magnification factor, and the only "
        "per-chamber scale knob",
        "ion_chamber.cpp set_mag_factor",
    ),
    Map2MapField(
        "source_to_axis_distance_mm",
        "ion_chamber",
        ROLE_OVERWRITTEN,
        "seeds the magnification factor at parse time, then set_mag_factors replaces "
        "it with source_to_isocenter_distance from scan_dose_system.xml",
        "engine::load_configuration, last statement",
    ),
    Map2MapField(
        "zero_offset_at_iso_mm",
        "ion_chamber",
        ROLE_EFFECTIVE,
        "added to converted positions at isocenter; not applied to spot sizes",
        "ion_chamber.hpp convert_strips",
    ),
    Map2MapField(
        "zero_offset_mm",
        "ion_chamber",
        ROLE_UNUSED,
        "only zero_offset_at_iso_mm is loaded",
        "ion_chamber.cpp set_data",
    ),
    # --- system geometry ------------------------------------------------------
    Map2MapField(
        "source_to_isocenter_distance",
        "geometry",
        ROLE_EFFECTIVE,
        "numerator of the magnification factor for every ion chamber, overriding each "
        "chamber's own source_to_axis_distance_mm",
        "engine.cpp load_system, devices.cpp set_mag_factors",
    ),
    Map2MapField(
        "source_to_x_axis_distance",
        "geometry",
        ROLE_VALIDATED_ONLY,
        "stored as m_geom.sad_x and bounds-checked, but no code reads it",
        "engine.cpp load_system",
    ),
    Map2MapField(
        "source_to_y_axis_distance",
        "geometry",
        ROLE_VALIDATED_ONLY,
        "stored as m_geom.sad_y and bounds-checked, but no code reads it",
        "engine.cpp load_system",
    ),
    # --- scan magnet ----------------------------------------------------------
    Map2MapField(
        "magnet_axis_to_iso_distance_mm",
        "scan_magnet",
        ROLE_MAGNET_EFFECTIVE,
        "the magnet's own SAD, used for the tangent term in dose conversions; "
        "unaffected by source_to_isocenter_distance",
        "scan_magnet.cpp set_data, engine.cpp get_sad",
    ),
    # --- gain conversion attributes -------------------------------------------
    *(
        Map2MapField(
            name,
            "gain_conversion",
            ROLE_UNUSED,
            "no xmlattr getter anywhere in scan_dose",
            "scan_magnet.cpp sc_gain_conversion",
        )
        for name in ("e0", "e1", "e2", "c1", "c3", "d2", "d4")
    ),
    Map2MapField(
        "m2",
        "gain_conversion",
        ROLE_UNUSED,
        "parsed but only ever written to the debug log",
        "scan_magnet.cpp sc_gain_conversion",
    ),
    *(
        Map2MapField(
            name,
            "gain_conversion",
            ROLE_UNUSED,
            "stored on volts rows, but cross-coupling reads the kilogauss row",
            "engine.cpp get_cross_correction_factor",
            units_filter="volts",
        )
        for name in ("b0", "b1")
    ),
)

_BY_SCOPE_NAME: dict[tuple[str, str], Map2MapField] = {
    (f.scope, f.name): f for f in MAP2MAP_FIELDS
}

# The form widget renders a scalar element without knowing its parent, so allow a
# scope-free lookup for any name the catalog defines exactly once.
_NAME_COUNTS = Counter(f.name for f in MAP2MAP_FIELDS)
_BY_UNIQUE_NAME: dict[str, Map2MapField] = {
    f.name: f for f in MAP2MAP_FIELDS if _NAME_COUNTS[f.name] == 1
}

GAIN_CONVERSION_DEAD_ATTRS = frozenset(
    f.name
    for f in MAP2MAP_FIELDS
    if f.scope == "gain_conversion" and f.is_dead and f.units_filter is None
)


def map2map_field(name: str, *, scope: str | None = None) -> Map2MapField | None:
    """Look up a field by name, preferring the entry for *scope* when given."""
    if scope is not None:
        scoped = _BY_SCOPE_NAME.get((scope, name))
        if scoped is not None:
            return scoped
    return _BY_UNIQUE_NAME.get(name)


def map2map_field_tooltip(name: str, *, scope: str | None = None) -> str:
    """Tooltip for a field, or an empty string when there is nothing worth saying."""
    entry = map2map_field(name, scope=scope)
    return entry.tooltip() if entry is not None else ""


def is_map2map_config_path(path: str | None) -> bool:
    if not path:
        return False
    try:
        return "map2map" in Path(path).resolve().parts
    except (OSError, ValueError):
        normalized = path.replace("\\", "/").lower()
        return "/map2map/" in normalized or normalized.endswith("/map2map")


def should_hide_map2map_attribute(element: ET.Element, attr: str) -> bool:
    """True when *attr* on *element* is known dead in map2map."""
    entry = _BY_SCOPE_NAME.get((element.tag, attr))
    if entry is None or not entry.is_dead:
        return False
    if entry.units_filter is not None:
        return element.get("units", "").lower() == entry.units_filter
    return True


def should_hide_map2map_child(parent: ET.Element, child_tag: str) -> bool:
    """True when a child element under *parent* is never read by map2map."""
    entry = _BY_SCOPE_NAME.get((parent.tag, child_tag))
    return entry is not None and entry.is_dead


def filter_attribute_names(element: ET.Element, names: list[str]) -> list[str]:
    return [name for name in names if not should_hide_map2map_attribute(element, name)]
