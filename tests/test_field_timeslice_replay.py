"""Tests for magnetic-field timeslice column resolution and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scan_kit.common.schema import C_MAG_FIELD_X, C_MAG_FIELD_Y, resolve_concept_column
from scan_kit.views.field_timeslice_replay import _load_session_timeline

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_field_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "1091134775" / "1091134775" / "layer-71" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_MAG_FIELD_X) == "r_tx2_probe_x"
    assert resolve_concept_column(cols, C_MAG_FIELD_Y) == "r_tx2_probe_y"


def test_g2_field_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "883144654" / "layer-9" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_MAG_FIELD_X) == "field_c_x"
    assert resolve_concept_column(cols, C_MAG_FIELD_Y) == "field_c_y"


def test_load_g3_session_timeline() -> None:
    data = _load_session_timeline("1091134775", str(TEST_DATA))
    assert data is not None
    assert data["n_samples"] > 0
    assert len(data["bx"]) == data["n_samples"]
    assert np.nanmax(np.abs(data["bx"])) > 0


def test_load_g2_session_timeline() -> None:
    data = _load_session_timeline("590658542", str(TEST_DATA))
    assert data is not None
    assert np.nanmax(np.abs(data["bx"])) > 0
