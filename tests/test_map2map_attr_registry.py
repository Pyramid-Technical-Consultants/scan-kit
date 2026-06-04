"""Tests for map2map dead-field registry used by the config editor."""

import xml.etree.ElementTree as ET

from scan_kit.workflows.config_tuning.map2map_attr_registry import (
    GAIN_CONVERSION_DEAD_ATTRS,
    filter_attribute_names,
    is_map2map_config_path,
    should_hide_map2map_attribute,
    should_hide_map2map_child,
)


def test_dead_gain_conversion_attrs_never_parsed_in_map2map() -> None:
    el = ET.Element(
        "gain_conversion",
        {
            "units": "kilogauss",
            "e0": "1",
            "e1": "2",
            "e2": "3",
            "c1": "4",
            "c3": "5",
            "d2": "6",
            "d4": "7",
            "m2": "8",
        },
    )
    for attr in GAIN_CONVERSION_DEAD_ATTRS:
        assert should_hide_map2map_attribute(el, attr)


def test_b0_b1_hidden_only_on_volts_row() -> None:
    volts = ET.Element("gain_conversion", {"units": "volts", "b0": "1", "b1": "2", "K1": "1"})
    kgauss = ET.Element("gain_conversion", {"units": "kilogauss", "b0": "1", "b1": "2", "K1": "1"})
    assert should_hide_map2map_attribute(volts, "b0")
    assert should_hide_map2map_attribute(volts, "b1")
    assert not should_hide_map2map_attribute(kgauss, "b0")
    assert not should_hide_map2map_attribute(kgauss, "K1")


def test_zero_offset_mm_child_hidden_under_ion_chamber() -> None:
    ic = ET.Element("ion_chamber")
    assert should_hide_map2map_child(ic, "zero_offset_mm")
    assert not should_hide_map2map_child(ic, "zero_offset_at_iso_mm")


def test_filter_preserves_live_attrs() -> None:
    el = ET.Element(
        "gain_conversion",
        {"units": "volts", "K0": "0", "K1": "1", "m1": "2", "Vb": "3", "e0": "4"},
    )
    names = filter_attribute_names(el, list(el.attrib))
    assert "K1" in names
    assert "Vb" in names
    assert "e0" not in names


def test_is_map2map_config_path() -> None:
    assert is_map2map_config_path("C:/sess/config/map2map/devices.xml")
    assert not is_map2map_config_path("C:/sess/config/map2map/../Input.xml")
