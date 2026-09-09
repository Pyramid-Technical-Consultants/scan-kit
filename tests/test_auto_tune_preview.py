"""Regression tests for auto-tune preview tables and panel wiring."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget

from scan_kit.workflows.config_tuning.auto_tuning.position_offset_preview_table import (
    clear_position_offset_preview_table,
    fill_position_offset_preview_table,
)
from scan_kit.workflows.config_tuning.auto_tuning.position_offset_tune import (
    compute_position_offset_tune_preview,
)
from scan_kit.workflows.config_tuning.auto_tuning.registry import get_auto_tune_workflow
from scan_kit.workflows.config_tuning.auto_tuning.sigma_preview_table import (
    _EXTREME_PCT_COLUMN,
    _TABLE_COLUMNS as SIGMA_TABLE_COLUMNS,
    clear_sigma_preview_table,
    fill_sigma_preview_table,
    max_preview_extreme_pct_deviation,
)
from scan_kit.workflows.config_tuning.auto_tuning.paths import resolve_devices_xml_path
from scan_kit.workflows.config_tuning.auto_tuning.sigma_tune import (
    SigmaTunePreviewRow,
    band_max_tolerance_excursion_pct,
    compute_band_sigma_k0,
    compute_sigma_tune_preview,
)
from scan_kit.workflows.config_tuning.auto_tune_panel import AutoTuneDetailWidget
from scan_kit.workflows.config_tuning.panel import ConfigTuningPanel

_ROOT = Path(__file__).resolve().parent.parent
_TEST_DATA = _ROOT / "test_data"
_SESSION = "1943968267"
_RIGHT_ALIGN = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def _load_devices_root() -> ET.Element:
    devices_path = _TEST_DATA / _SESSION / _SESSION / "config" / "map2map" / "devices.xml"
    return ET.fromstring(devices_path.read_text(encoding="utf-8"))


def _legacy_center_pct_deviation(sigmas: np.ndarray, k0: float) -> float:
    """Old preview metric: distance from K0 center, not from the tolerance band."""
    if sigmas.size == 0 or not np.isfinite(k0) or abs(k0) < 1e-12:
        return float("nan")
    return float(np.max(np.abs(sigmas - k0)) / abs(k0) * 100.0)


def _sigma_preview_rows() -> list:
    root = _load_devices_root()
    rows, _ = compute_sigma_tune_preview(root, [_SESSION], str(_TEST_DATA))
    assert rows
    return rows


def _position_offset_preview_rows() -> list:
    root = _load_devices_root()
    rows, _ = compute_position_offset_tune_preview(root, [_SESSION], str(_TEST_DATA))
    assert rows
    return rows


def test_tolerance_band_oob_is_not_center_deviation_regression() -> None:
    """Regression: center deviation looked like flat ~tolerance% for every band."""
    sigmas = np.array([4.0, 5.0, 6.0])
    k0 = compute_band_sigma_k0(
        sigmas,
        tolerance_percent=20.0,
        lower_headroom_percent=1.0,
    )
    legacy = _legacy_center_pct_deviation(sigmas, k0)
    oob_pct, _, _ = band_max_tolerance_excursion_pct(sigmas, k0, 20.0)

    assert legacy == pytest.approx(20.792, abs=0.01)
    assert oob_pct == pytest.approx(0.792, abs=0.01)
    assert oob_pct < legacy / 2.0


def test_compute_sigma_tune_preview_max_oob_stays_below_tolerance() -> None:
    rows = _sigma_preview_rows()
    max_oob = max_preview_extreme_pct_deviation(rows)
    assert max_oob is not None
    assert max_oob < 5.0
    assert max_oob != pytest.approx(20.0, abs=1.0)


def test_sigma_preview_table_header_matches_cell_alignment(qapp) -> None:
    table = QTableWidget()
    clear_sigma_preview_table(table)
    assert table.horizontalHeader().defaultAlignment() == _RIGHT_ALIGN

    fill_sigma_preview_table(table, _sigma_preview_rows())
    assert table.rowCount() > 0
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item is not None:
                assert item.textAlignment() == _RIGHT_ALIGN


def test_sigma_preview_table_oob_column_not_flat_tolerance(qapp) -> None:
    table = QTableWidget()
    fill_sigma_preview_table(table, _sigma_preview_rows())

    oob_col = SIGMA_TABLE_COLUMNS.index(_EXTREME_PCT_COLUMN)
    values: list[float] = []
    for row in range(table.rowCount()):
        item = table.item(row, oob_col)
        assert item is not None
        text = item.text().strip()
        if text and text != "—":
            values.append(float(text.rstrip("%")))

    assert values
    assert max(values) < 5.0
    assert not all(abs(value - 20.0) < 1.0 for value in values)


def test_position_offset_preview_table_header_matches_cell_alignment(qapp) -> None:
    table = QTableWidget()
    clear_position_offset_preview_table(table)
    assert table.horizontalHeader().defaultAlignment() == _RIGHT_ALIGN

    fill_position_offset_preview_table(table, _position_offset_preview_rows())
    assert table.rowCount() > 0
    for row in range(table.rowCount()):
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item is not None:
                assert item.textAlignment() == _RIGHT_ALIGN


def test_position_offset_preview_table_shows_offset_changes(qapp) -> None:
    table = QTableWidget()
    fill_position_offset_preview_table(table, _position_offset_preview_rows())

    ic_texts: list[str] = []
    for row in range(table.rowCount()):
        for col in range(1, 5):
            item = table.item(row, col)
            if item is not None:
                ic_texts.append(item.text())

    assert ic_texts
    assert any("→" in text for text in ic_texts)


def test_config_tuning_panel_position_offset_preview_returns_changes(qapp) -> None:
    panel = ConfigTuningPanel()
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    assert panel.open_config_root(config_root)

    workflow = get_auto_tune_workflow("position_offset_tuning")
    assert workflow is not None

    result = panel._on_auto_tune_preview(
        workflow,
        {
            "session_ids": [_SESSION],
            "data_dir": str(_TEST_DATA),
            "optimize_method": "median",
        },
    )
    assert result is not None
    rows, warnings = result
    assert not warnings or rows
    assert len(rows) > 0
    assert any(abs(row.delta_offset) > 1e-9 for row in rows)


def test_config_tuning_panel_sigma_preview_returns_oob_rows(qapp) -> None:
    panel = ConfigTuningPanel()
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    assert panel.open_config_root(config_root)

    workflow = get_auto_tune_workflow("sigma_tuning")
    assert workflow is not None

    result = panel._on_auto_tune_preview(
        workflow,
        {
            "session_ids": [_SESSION],
            "data_dir": str(_TEST_DATA),
            "sigma_tolerance_percent": 20.0,
            "sigma_lower_headroom_percent": 1.0,
        },
    )
    assert result is not None
    rows, warnings = result
    assert not warnings or rows
    assert len(rows) > 0
    max_oob = max_preview_extreme_pct_deviation(rows)
    assert max_oob is not None
    assert max_oob < 5.0


def test_sigma_preview_table_zero_delta_shows_single_value(qapp) -> None:
    row = SigmaTunePreviewRow(
        device="IC_1_X",
        min_energy=100.0,
        max_energy=100.0,
        old_k0=5.0,
        new_k0=5.0,
        n_spots=10,
        sigma_variance=0.1,
        extreme_pct_deviation=0.0,
        extreme_observed_mm=float("nan"),
        extreme_kind="",
    )
    table = QTableWidget()
    fill_sigma_preview_table(table, [row])
    ic_item = table.item(0, 1)
    assert ic_item is not None
    assert "→" not in ic_item.text()
    assert ic_item.text() == "5.000"


def test_auto_tune_detail_read_params_sigma_spin_values(qapp) -> None:
    workflow = get_auto_tune_workflow("sigma_tuning")
    assert workflow is not None
    detail = AutoTuneDetailWidget()
    detail.set_workflow(workflow)
    detail._sigma_tolerance_spin.setValue(12.5)
    detail._sigma_lower_headroom_spin.setValue(2.0)
    params = detail.read_params()
    assert params["sigma_tolerance_percent"] == 12.5
    assert params["sigma_lower_headroom_percent"] == 2.0


def test_auto_tune_detail_read_params_position_optimize_method(qapp) -> None:
    workflow = get_auto_tune_workflow("position_offset_tuning")
    assert workflow is not None
    detail = AutoTuneDetailWidget()
    detail.set_workflow(workflow)
    detail._method_weighted.setChecked(True)
    params = detail.read_params()
    assert params["optimize_method"] == "weighted_average"
    assert params["data_source"] == "spot"


def test_config_tuning_panel_sigma_apply_marks_devices_dirty(qapp) -> None:
    panel = ConfigTuningPanel()
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    assert panel.open_config_root(config_root)

    workflow = get_auto_tune_workflow("sigma_tuning")
    assert workflow is not None
    result = panel._on_auto_tune_apply(
        workflow,
        {
            "session_ids": [_SESSION],
            "data_dir": str(_TEST_DATA),
            "sigma_tolerance_percent": 20.0,
            "sigma_lower_headroom_percent": 1.0,
        },
    )
    assert result is not None
    assert result.success

    devices_path = resolve_devices_xml_path(config_root)
    assert devices_path is not None
    document = panel._open_documents.get(devices_path.resolve())
    assert document is not None
    assert document.dirty


def test_config_tuning_panel_position_offset_apply_marks_devices_dirty(qapp) -> None:
    panel = ConfigTuningPanel()
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    assert panel.open_config_root(config_root)

    workflow = get_auto_tune_workflow("position_offset_tuning")
    assert workflow is not None
    result = panel._on_auto_tune_apply(
        workflow,
        {
            "session_ids": [_SESSION],
            "data_dir": str(_TEST_DATA),
            "optimize_method": "median",
        },
    )
    assert result is not None
    assert result.success

    devices_path = resolve_devices_xml_path(config_root)
    assert devices_path is not None
    document = panel._open_documents.get(devices_path.resolve())
    assert document is not None
    assert document.dirty


def test_config_tuning_panel_apply_failure_does_not_mark_dirty(qapp) -> None:
    panel = ConfigTuningPanel()
    config_root = _TEST_DATA / _SESSION / _SESSION / "config"
    assert panel.open_config_root(config_root)

    workflow = get_auto_tune_workflow("sigma_tuning")
    assert workflow is not None
    result = panel._on_auto_tune_apply(
        workflow,
        {"session_ids": ["missing-session"], "data_dir": str(_TEST_DATA)},
    )
    assert result is not None
    assert not result.success

    devices_path = resolve_devices_xml_path(config_root)
    assert devices_path is not None
    document = panel._open_documents.get(devices_path.resolve())
    assert document is not None
    assert not document.dirty
