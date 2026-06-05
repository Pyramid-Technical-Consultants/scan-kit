"""Tests for sigma auto-tuning from session measurements."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scan_kit.common.session_sigma import load_measured_sigma_spots, measured_sigma_by_energy
from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_devices_xml_path
from scan_kit.workflows.config_tuning.auto_tuning.sigma_tune import (
    apply_measured_sigmas_to_tree,
    tune_sigmas_from_session,
)

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"


def test_load_measured_sigma_spots_fixture() -> None:
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None
    assert "IC_1_X" in spots
    energies, sigmas = spots["IC_1_X"]
    assert len(energies) == len(sigmas)
    assert len(energies) > 100
    assert sigmas.min() > 0


def test_measured_sigma_by_energy_fixture() -> None:
    by_energy = measured_sigma_by_energy(_SESSION, _TEST_DATA)
    assert by_energy is not None
    assert len(by_energy["IC_1_X"]) >= 10
    assert max(by_energy["IC_1_X"]) > 100.0


def test_resolve_devices_xml_under_config_root() -> None:
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    path = resolve_devices_xml_path(config_root)
    assert path is not None
    assert path.name == "devices.xml"
    assert "map2map" in path.parts


def test_apply_measured_sigmas_updates_k0() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    text = devices_path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None

    chamber = next(
        c
        for c in root.iter("ion_chamber")
        if c.find("device") is not None and c.find("device").get("name") == "IC_1_X"
    )
    before = [el.get("K0") for el in chamber.findall("beam_sigma_conversions")]

    result = apply_measured_sigmas_to_tree(root, spots)
    assert result.bands_updated > 50
    after = [el.get("K0") for el in chamber.findall("beam_sigma_conversions")]
    assert before != after


def test_tune_sigmas_from_session_integration() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    result = tune_sigmas_from_session(root, _SESSION, str(_TEST_DATA))
    assert result.ok
    assert result.bands_updated > 0
