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
from scan_kit.common.session_sigma import MeasuredSigmaSpots
from scan_kit.workflows.config_tuning.auto_tuning.sigma_tune import (
    apply_measured_sigmas_to_tree,
    band_max_tolerance_excursion_pct,
    band_sigma_variance,
    collect_sigma_band_updates,
    compute_band_sigma,
    compute_band_sigma_k0,
    compute_sigma_tune_preview,
    format_sigma_k0,
    sigma_tolerance_lower_mm,
    sigma_tolerance_upper_mm,
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


def test_band_max_tolerance_excursion_pct_detects_out_of_band_spots() -> None:
    k0 = 5.0
    tol = 20.0
    pct_in_band, observed_in_band, kind_in_band = band_max_tolerance_excursion_pct(
        np.array([4.0, 5.0, 6.0]),
        k0,
        tol,
    )
    assert pct_in_band == 0.0
    assert kind_in_band == ""
    assert not np.isfinite(observed_in_band)

    pct, observed, kind = band_max_tolerance_excursion_pct(
        np.array([3.0, 5.0, 6.0]),
        k0,
        tol,
    )
    assert kind == "below"
    assert observed == pytest.approx(3.0)
    assert pct == pytest.approx(20.0)

    pct_above, observed_above, kind_above = band_max_tolerance_excursion_pct(
        np.array([4.0, 6.5, 6.0]),
        k0,
        tol,
    )
    assert kind_above == "above"
    assert observed_above == pytest.approx(6.5)
    assert pct_above == pytest.approx(10.0)


def test_band_oob_differs_from_legacy_center_deviation() -> None:
    sigmas = np.array([4.0, 5.0, 6.0])
    k0 = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=1.0)
    legacy = float(np.max(np.abs(sigmas - k0)) / abs(k0) * 100.0)
    oob_pct, _, _ = band_max_tolerance_excursion_pct(sigmas, k0, 20.0)
    assert legacy == pytest.approx(20.792, abs=0.01)
    assert oob_pct == pytest.approx(0.792, abs=0.01)


def test_collect_sigma_band_updates_oob_not_flat_tolerance_pct() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None

    updates, _ = collect_sigma_band_updates(root, spots, tolerance_percent=20.0)
    oob_values = [u.extreme_pct_deviation for u in updates]
    assert oob_values
    assert max(oob_values) < 5.0
    assert max(oob_values) == pytest.approx(0.792, abs=0.01)

    wide_updates, _ = collect_sigma_band_updates(
        root,
        spots,
        tolerance_percent=20.0,
        lower_headroom_percent=0.0,
    )
    wide_oob = max(u.extreme_pct_deviation for u in wide_updates)
    assert wide_oob == 0.0


def test_band_sigma_variance_uses_sample_variance() -> None:
    assert band_sigma_variance(np.array([2.0])) == 0.0
    assert band_sigma_variance(np.array([2.0, 4.0, 6.0])) == pytest.approx(4.0)


def test_compute_band_sigma_modes() -> None:
    sigmas = np.array([2.0, 4.0, 10.0])
    weights = np.array([1.0, 1.0, 8.0])
    assert compute_band_sigma(sigmas, weights, "median") == pytest.approx(4.0)
    assert compute_band_sigma(sigmas, weights, "min_max_midpoint") == pytest.approx(6.0)
    assert compute_band_sigma(sigmas, weights, "weighted_average") == pytest.approx(8.6)


def test_compute_band_sigma_k0_pins_minimum_with_headroom_to_lower_edge() -> None:
    k0 = compute_band_sigma_k0(
        np.array([4.0, 5.0, 6.0]),
        tolerance_percent=20.0,
        lower_headroom_percent=0.0,
    )
    assert k0 == pytest.approx(5.0)
    assert sigma_tolerance_lower_mm(k0, 20.0) == pytest.approx(4.0)


def test_compute_band_sigma_k0_lower_headroom_adds_margin() -> None:
    sigmas = np.array([4.0, 5.0])
    bare = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=0.0)
    with_headroom = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=1.0)
    assert with_headroom > bare
    assert sigma_tolerance_lower_mm(with_headroom, 20.0) > float(np.min(sigmas))


def test_compute_band_sigma_k0_covers_max_when_spread_fits() -> None:
    sigmas = np.array([4.0, 5.0])
    k0 = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=0.0)
    assert sigma_tolerance_lower_mm(k0, 20.0) <= float(np.min(sigmas))
    assert sigma_tolerance_upper_mm(k0, 20.0) >= float(np.max(sigmas))


def test_compute_band_sigma_k0_prioritizes_min_when_spread_too_wide() -> None:
    sigmas = np.array([4.0, 7.0])
    k0 = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=0.0)
    assert sigma_tolerance_lower_mm(k0, 20.0) == pytest.approx(4.0)
    assert sigma_tolerance_upper_mm(k0, 20.0) < float(np.max(sigmas))


def test_collect_sigma_band_updates_respects_tolerance() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    spots = load_measured_sigma_spots(_SESSION, _TEST_DATA)
    assert spots is not None

    updates, _ = collect_sigma_band_updates(
        root,
        spots,
        tolerance_percent=20.0,
        lower_headroom_percent=1.0,
    )
    assert updates
    for update in updates:
        lower = sigma_tolerance_lower_mm(update.new_k0, 20.0)
        upper = sigma_tolerance_upper_mm(update.new_k0, 20.0)
        energies, sigmas = spots.by_device[update.device]
        mask = (energies >= update.min_energy) & (energies <= update.max_energy)
        band = sigmas[mask]
        min_with_headroom = float(np.min(band)) * 1.01
        assert lower >= min_with_headroom - 1e-9
        assert lower > float(np.min(band))
        if float(np.max(band)) <= upper:
            assert upper >= float(np.max(band))


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


def test_tolerance_percent_changes_k0_assignment() -> None:
    sigmas = np.array([4.0, 5.0])
    loose = compute_band_sigma_k0(sigmas, tolerance_percent=20.0, lower_headroom_percent=0.0)
    tight = compute_band_sigma_k0(sigmas, tolerance_percent=10.0, lower_headroom_percent=0.0)
    assert loose > tight


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
    assert rows[0].extreme_kind in {"below", "above", ""}
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


def test_collect_sigma_band_updates_skips_non_constant_k_bands() -> None:
    xml = """
    <devices>
      <ion_chamber>
        <device name="IC_1_X"/>
        <beam_sigma_conversions in_units="MEV" out_units="mm"
            min_energy="100" max_energy="100" K0="5.0" K1="0.5" K2="0" K3="0"/>
        <beam_sigma_conversions in_units="MEV" out_units="mm"
            min_energy="200" max_energy="200" K0="5.0" K1="0" K2="0" K3="0"/>
      </ion_chamber>
    </devices>
    """
    root = ET.fromstring(xml)
    measured = MeasuredSigmaSpots(
        by_device={"IC_1_X": (np.array([100.0, 200.0]), np.array([4.0, 5.0]))},
    )
    updates, _ = collect_sigma_band_updates(root, measured, lower_headroom_percent=0.0)
    assert len(updates) == 1
    assert updates[0].min_energy == pytest.approx(200.0)
    assert updates[0].element.get("K1") == "0"


def _find_band_element(
    root: ET.Element,
    device: str,
    min_energy: float,
    max_energy: float,
) -> ET.Element | None:
    for chamber in root.iter("ion_chamber"):
        device_el = chamber.find("device")
        if device_el is None or device_el.get("name") != device:
            continue
        for el in chamber.findall("beam_sigma_conversions"):
            try:
                min_e = float(el.get("min_energy", "nan"))
                max_e = float(el.get("max_energy", "nan"))
            except (TypeError, ValueError):
                continue
            if min_e == min_energy and max_e == max_energy:
                return el
    return None


def test_sigma_preview_k0_values_match_apply() -> None:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    preview_rows, _ = compute_sigma_tune_preview(root, [_SESSION], str(_TEST_DATA))
    assert preview_rows

    apply_root = ET.fromstring(devices_path.read_text(encoding="utf-8"))
    result = tune_sigmas_from_session(apply_root, _SESSION, str(_TEST_DATA))
    assert result.ok

    for row in preview_rows[:10]:
        element = _find_band_element(
            apply_root,
            row.device,
            row.min_energy,
            row.max_energy,
        )
        assert element is not None
        assert element.get("K0") == format_sigma_k0(row.new_k0)
