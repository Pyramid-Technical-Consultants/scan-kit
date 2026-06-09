"""Tests for selective timeslice CSV loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scan_kit.common.schema import C_LAYER_ID, canonicalize_dataframe_columns
from scan_kit.common.session_source import (
    _read_csv_robust,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from scan_kit.common.timeslice_position_error import TIMESLICE_POSITION_ERROR_COLS

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_read_csv_robust_usecols_matches_full_subset() -> None:
    path = (
        TEST_DATA / "1091134775" / "1091134775" / "layer-71" / "run-0"
        / "timeslice_data_device_units.csv"
    )
    if not path.is_file():
        return

    usecols = [
        C_LAYER_ID,
        "rci_in_trigger",
        "r_ic1_x_position",
        "ic1_position_x_target",
    ]
    full = canonicalize_dataframe_columns(pd.read_csv(path))
    subset = _read_csv_robust(path, usecols=usecols)

    for col in usecols:
        if col not in full.columns:
            continue
        pd.testing.assert_series_equal(
            subset[col],
            full[col],
            check_names=False,
        )


def test_cached_raw_usecols_loads_all_layers() -> None:
    sid = "1091134775"
    src = resolve_session_source(sid, str(TEST_DATA))
    if src is None:
        return

    frames = load_session_timeslice_device_units(src, usecols=TIMESLICE_POSITION_ERROR_COLS)
    assert len(frames) == 76
    for df in frames:
        assert "rci_in_trigger" in df.columns or "r_beamOk" in df.columns
        assert C_LAYER_ID in df.columns
        assert "_layer_idx" in df.columns
