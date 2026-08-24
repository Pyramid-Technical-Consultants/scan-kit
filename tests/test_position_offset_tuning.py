"""Tests for position-offset auto-tuning from session measurements."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scan_kit.common.session_position import (
    load_measured_position_errors_for_sessions,
    load_measured_position_errors_spot,
    load_measured_position_errors_timeslice,
)
from scan_kit.workflows.config_tuning.auto_tuning.position_offset_preview_table import (
    max_preview_residual_mm,
)
from scan_kit.workflows.config_tuning.auto_tuning.position_offset_tune import (
    apply_position_offsets_to_tree,
    band_error_variance,
    band_max_abs_error_mm,
    band_max_residual_mm,
    collect_offset_updates,
    compute_global_offset_correction,
    compute_position_offset_tune_preview,
    read_zero_offsets_from_tree,
    tune_position_offsets_from_sessions,
)

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"


def _load_devices_root() -> ET.Element:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    return ET.fromstring(devices_path.read_text(encoding="utf-8"))


def test_load_measured_position_errors_spot_fixture() -> None:
    measured = load_measured_position_errors_spot(_SESSION, _TEST_DATA)
    assert measured is not None
    assert "IC_1_X" in measured.by_device
    energies, errors = measured.by_device["IC_1_X"]
    assert len(energies) == len(errors)
    assert len(energies) > 100
    assert np.isfinite(errors).any()
    assert measured.weights is not None
    assert len(measured.weights) == len(energies)


def test_read_zero_offsets_from_tree_fixture() -> None:
    root = _load_devices_root()
    offsets = read_zero_offsets_from_tree(root)
    assert set(offsets) >= {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}
    _, ic1_x = offsets["IC_1_X"]
    assert ic1_x == pytest.approx(-4.0)


def test_global_correction_reduces_median_residual() -> None:
    measured = load_measured_position_errors_spot(_SESSION, _TEST_DATA)
    assert measured is not None
    _, errors = measured.by_device["IC_1_X"]
    finite = errors[np.isfinite(errors)]
    correction = compute_global_offset_correction(finite, measured.weights, "median")
    residuals = finite - correction
    assert abs(float(np.median(residuals))) < abs(float(np.median(finite)))


def test_collect_offset_updates_finds_four_devices() -> None:
    root = _load_devices_root()
    measured = load_measured_position_errors_spot(_SESSION, _TEST_DATA)
    assert measured is not None
    updates, warnings = collect_offset_updates(root, measured)
    assert len(updates) == 4
    assert not warnings or updates
    devices = {update.device for update in updates}
    assert devices == {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}


def test_apply_position_offsets_updates_xml_text() -> None:
    root = _load_devices_root()
    measured = load_measured_position_errors_spot(_SESSION, _TEST_DATA)
    assert measured is not None

    chamber = next(
        c
        for c in root.iter("ion_chamber")
        if c.find("device") is not None and c.find("device").get("name") == "IC_1_X"
    )
    before = chamber.find("zero_offset_at_iso_mm").text

    result = apply_position_offsets_to_tree(root, measured)
    assert result.offsets_updated == 4
    after = chamber.find("zero_offset_at_iso_mm").text
    assert before != after


def test_compute_position_offset_tune_preview_matches_apply_count() -> None:
    root = _load_devices_root()
    rows, warnings = compute_position_offset_tune_preview(
        root,
        [_SESSION],
        str(_TEST_DATA),
    )
    assert not warnings or rows
    assert len(rows) > 50
    assert rows[0].device in {"IC_1_X", "IC_1_Y", "IC_2_X", "IC_2_Y"}
    assert rows[0].n_samples > 0
    assert rows[0].error_variance >= 0.0
    assert rows[0].residual_variance >= 0.0
    assert rows[0].max_abs_error_mm >= 0.0
    assert rows[0].max_residual_mm >= 0.0
    max_residual = max_preview_residual_mm(rows)
    assert max_residual is not None
    assert max_residual >= rows[0].max_residual_mm


def test_band_error_variance_and_residual_helpers() -> None:
    assert band_error_variance(np.array([0.1])) == 0.0
    assert band_error_variance(np.array([0.0, 2.0, 4.0])) == pytest.approx(4.0)
    assert band_max_abs_error_mm(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert band_max_residual_mm(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)


def test_merge_measured_position_errors_combines_sessions() -> None:
    single = load_measured_position_errors_spot(_SESSION, _TEST_DATA)
    assert single is not None
    single_n = len(single.by_device["IC_1_X"][0])

    merged, warnings = load_measured_position_errors_for_sessions(
        [_SESSION, _SESSION],
        _TEST_DATA,
    )
    assert merged is not None
    assert not warnings
    assert len(merged.by_device["IC_1_X"][0]) == single_n * 2
    assert len(merged.weights) == single_n * 2


def test_tune_position_offsets_from_sessions_integration() -> None:
    root = _load_devices_root()
    result = tune_position_offsets_from_sessions(root, [_SESSION], str(_TEST_DATA))
    assert result.ok
    assert result.offsets_updated == 4


def test_load_measured_position_errors_timeslice_fixture() -> None:
    measured = load_measured_position_errors_timeslice(_SESSION, _TEST_DATA)
    if measured is None:
        pytest.skip("Fixture session has no timeslice position error data.")
    assert "IC_1_X" in measured.by_device
    energies, errors = measured.by_device["IC_1_X"]
    assert len(energies) == len(errors)
    assert len(energies) > 0
