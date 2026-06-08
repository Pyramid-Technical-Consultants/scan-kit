"""Tests for sigma auto-tuning from session measurements."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scan_kit.common.session_sigma import (
    load_measured_sigma_spots,
    load_measured_sigma_spots_for_sessions,
    measured_sigma_by_energy,
    merge_measured_sigma_spots,
)
from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_devices_xml_path
from scan_kit.workflows.config_tuning.auto_tuning.sigma_preview_table import (
    max_preview_extreme_pct_deviation,
)
from scan_kit.workflows.config_tuning.auto_tuning.sigma_tune import (
    apply_measured_sigmas_to_tree,
    band_furthest_extreme_pct_deviation,
    band_sigma_variance,
    collect_sigma_band_updates,
    compute_band_sigma,
    compute_sigma_tune_preview,
    tune_sigmas_from_session,
)

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"


def test_load_measured_sigma_spots_fixture() -> None:
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None
    assert "IC_1_X" in spots.by_device
    energies, sigmas = spots.by_device["IC_1_X"]
    assert len(energies) == len(sigmas)
    assert len(energies) > 100
    assert sigmas.min() > 0
    assert spots.weights is not None
    assert len(spots.weights) == len(energies)


def test_measured_sigma_by_energy_fixture() -> None:
    by_energy = measured_sigma_by_energy(_SESSION, _TEST_DATA)
    assert by_energy is not None
    assert len(by_energy["IC_1_X"]) >= 10
    assert max(by_energy["IC_1_X"]) > 100.0


def test_band_furthest_extreme_pct_deviation_uses_furthest_min_or_max() -> None:
    pct, observed, kind = band_furthest_extreme_pct_deviation(
        np.array([2.0, 4.0, 10.0]),
        5.0,
    )
    assert kind == "max"
    assert observed == pytest.approx(10.0)
    assert pct == pytest.approx(100.0)

    pct_min, observed_min, kind_min = band_furthest_extreme_pct_deviation(
        np.array([1.0, 4.0, 10.0]),
        8.0,
    )
    assert kind_min == "min"
    assert observed_min == pytest.approx(1.0)
    assert pct_min == pytest.approx(87.5)


def test_band_sigma_variance_uses_sample_variance() -> None:
    assert band_sigma_variance(np.array([2.0])) == 0.0
    assert band_sigma_variance(np.array([2.0, 4.0, 6.0])) == pytest.approx(4.0)


def test_compute_band_sigma_modes() -> None:
    sigmas = np.array([2.0, 4.0, 10.0])
    weights = np.array([1.0, 1.0, 8.0])
    assert compute_band_sigma(sigmas, weights, "median") == pytest.approx(4.0)
    assert compute_band_sigma(sigmas, weights, "min_max_midpoint") == pytest.approx(6.0)
    assert compute_band_sigma(sigmas, weights, "weighted_average") == pytest.approx(8.6)


def test_resolve_devices_xml_under_config_root() -> None:
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    path = resolve_devices_xml_path(config_root)
    assert path is not None
    assert path.name == "devices.xml"
    assert "map2map" in path.parts


def test_resolve_session_config_dir_uses_session_config_folder() -> None:
    from scan_kit.workflows.config_tuning.auto_tuning.paths import (
        resolve_session_config_dir,
    )

    config_dir = resolve_session_config_dir(_SESSION, _TEST_DATA)
    assert config_dir is not None
    assert config_dir.name == "config"
    assert resolve_devices_xml_path(config_dir) is not None


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


def test_optimize_modes_can_differ_for_same_band() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None

    median_updates, _ = collect_sigma_band_updates(root, spots, optimize_mode="median")
    weighted_updates, _ = collect_sigma_band_updates(
        root, spots, optimize_mode="weighted_average"
    )
    midpoint_updates, _ = collect_sigma_band_updates(
        root, spots, optimize_mode="min_max_midpoint"
    )
    assert len(median_updates) > 0
    assert {u.new_k0 for u in weighted_updates} != {u.new_k0 for u in midpoint_updates}


def test_compute_sigma_tune_preview_matches_apply_count() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    rows, warnings = compute_sigma_tune_preview(root, [_SESSION], str(_TEST_DATA))
    assert not warnings or rows
    assert len(rows) > 50
    assert rows[0].device in {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}
    assert rows[0].n_spots > 0
    assert rows[0].sigma_variance >= 0.0
    assert rows[0].extreme_pct_deviation >= 0.0
    assert rows[0].extreme_kind in {"min", "max"}
    max_pct = max_preview_extreme_pct_deviation(rows)
    assert max_pct is not None
    assert max_pct >= rows[0].extreme_pct_deviation


def test_merge_measured_sigma_spots_combines_sessions() -> None:
    single = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert single is not None
    single_n = len(single.by_device["IC_1_X"][0])

    merged, warnings = load_measured_sigma_spots_for_sessions(
        [_SESSION, _SESSION],
        _TEST_DATA,
    )
    assert merged is not None
    assert not warnings
    assert len(merged.by_device["IC_1_X"][0]) == single_n * 2
    assert len(merged.weights) == single_n * 2


def test_tune_sigmas_from_session_integration() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    result = tune_sigmas_from_session(root, _SESSION, str(_TEST_DATA))
    assert result.ok
    assert result.bands_updated > 0
