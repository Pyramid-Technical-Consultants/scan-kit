"""Tests for the map2map isocenter geometry model in ``common/devices_xml``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.devices_xml import (
    INVALID_POSITION_MM,
    load_map2map_geometry,
    parse_devices_xml,
    parse_system_geometry,
)

_DEVICES = """<?xml version="1.0"?>
<devices>
  <ion_chamber>
    <device name="IC_1_X"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>-4.0</zero_offset_at_iso_mm>
    <source_to_device_distance_mm>1250</source_to_device_distance_mm>
    <source_to_axis_distance_mm>999</source_to_axis_distance_mm>
  </ion_chamber>
  <ion_chamber>
    <device name="IC_1_Y"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
    <zero_offset_at_iso_mm>0.5</zero_offset_at_iso_mm>
    <reverse_strips>1</reverse_strips>
    <source_to_device_distance_mm>1250</source_to_device_distance_mm>
    <source_to_axis_distance_mm>999</source_to_axis_distance_mm>
  </ion_chamber>
  <ion_chamber>
    <device name="IC_NO_SDD"/>
    <strip_count>128</strip_count>
    <strip_to_mm>2</strip_to_mm>
  </ion_chamber>
  <scan_magnet>
    <device name="SM_X"/>
    <major_axis>X</major_axis>
    <magnet_axis_to_iso_distance_mm>1800</magnet_axis_to_iso_distance_mm>
  </scan_magnet>
</devices>
"""

_SYSTEM = """<?xml version="1.0"?>
<MapToMap>
  <geometry>
    <source_to_isocenter_distance>2500</source_to_isocenter_distance>
    <source_to_x_axis_distance>1900</source_to_x_axis_distance>
    <source_to_y_axis_distance>1520</source_to_y_axis_distance>
  </geometry>
</MapToMap>
"""


def _write_config(tmp_path: Path, *, devices: str = _DEVICES, system: str = _SYSTEM) -> Path:
    config_dir = tmp_path / "config" / "map2map"
    config_dir.mkdir(parents=True)
    (config_dir / "devices.xml").write_text(devices, encoding="utf-8")
    (config_dir / "scan_dose_system.xml").write_text(system, encoding="utf-8")
    return config_dir


def test_mag_factor_uses_virtual_sad_not_per_ic_sad(tmp_path: Path) -> None:
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    # 2500 / 1250, never 999 / 1250 — set_mag_factors overwrites the per-IC seed.
    assert geometry.mag_factor("IC_1_X") == 2.0
    assert geometry.chambers["IC_1_X"].xml_sad_mm == 999.0


def test_strip_to_iso_centers_and_offsets(tmp_path: Path) -> None:
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    # center strip 63.5, 4 mm per strip at iso (2 mm x mag 2), offset -4 mm.
    iso = geometry.strip_to_iso("IC_1_X", [63.5, 64.5, 0.0])
    assert np.allclose(iso, [-4.0, 0.0, -4.0 - 63.5 * 4.0])


def test_reverse_strips_flips_the_slope(tmp_path: Path) -> None:
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    assert geometry.iso_mm_per_strip("IC_1_Y") == -4.0
    assert np.allclose(geometry.strip_to_iso("IC_1_Y", [63.5, 64.5]), [0.5, -3.5])


def test_invalid_raw_strips_become_the_sentinel(tmp_path: Path) -> None:
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    iso = geometry.strip_to_iso("IC_1_X", [-1.0, -10000.0, -0.5])
    assert iso[0] == INVALID_POSITION_MM
    assert iso[1] == INVALID_POSITION_MM
    assert iso[2] != INVALID_POSITION_MM


def test_sigma_to_iso_scales_without_offset_or_flip(tmp_path: Path) -> None:
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    # Spot size gets the same magnification but no zero offset and no sign flip.
    assert np.allclose(geometry.sigma_to_iso("IC_1_X", [1.0, 2.5]), [4.0, 10.0])
    assert np.allclose(geometry.sigma_to_iso("IC_1_Y", [1.0]), [4.0])


def test_chamber_without_required_distance_is_skipped(tmp_path: Path) -> None:
    config = parse_devices_xml(_DEVICES)
    assert "IC_NO_SDD" not in config.chambers
    geometry = load_map2map_geometry(_write_config(tmp_path))
    assert geometry is not None
    assert geometry.strip_to_iso("IC_NO_SDD", [64.0]) is None


def test_magnet_sad_is_parsed_and_untouched_by_virtual_sad() -> None:
    config = parse_devices_xml(_DEVICES)
    magnet = config.magnets["SM_X"]
    assert magnet.axis == "x"
    assert magnet.axis_to_iso_mm == 1800.0


def test_system_geometry_carries_validated_only_fields() -> None:
    system = parse_system_geometry(_SYSTEM)
    assert system is not None
    assert system.virtual_sad_mm == 2500.0
    assert system.sad_x_mm == 1900.0
    assert system.sad_y_mm == 1520.0


def test_out_of_range_virtual_sad_is_rejected() -> None:
    for value in ("0", "-1", "7000"):
        text = _SYSTEM.replace(
            "<source_to_isocenter_distance>2500<",
            f"<source_to_isocenter_distance>{value}<",
        )
        assert parse_system_geometry(text) is None


def test_malformed_config_returns_none(tmp_path: Path) -> None:
    # Two test_data sessions ship devices.xml with junk after the root element.
    broken = _write_config(tmp_path / "junk", devices=_DEVICES + "</devices>")
    assert load_map2map_geometry(broken) is None
    empty = _write_config(tmp_path / "empty", system="")
    assert load_map2map_geometry(empty) is None
    assert load_map2map_geometry(tmp_path / "missing") is None
