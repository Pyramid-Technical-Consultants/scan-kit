"""Tests for G3 timeslice position column handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scan_kit.common.g3_timeslice_position import (
    g3_position_error_arrays,
    resolve_g3_fit_ok_column,
    resolve_g3_position_target_columns,
    resolve_g3_quality_columns,
    valid_g3_fit_values,
)
from scan_kit.common.schema import (
    C_IC1_X_POS,
    POSITION_KEY_G3,
    canonical_column_aliases,
    canonicalize_dataframe_columns,
    concept_column_candidates,
    resolve_concept_column,
)
from scan_kit.common.schema import C_IC1_X_POS_RAW, POSITION_KEY_G3_RAW

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_raw_concept_candidates_exclude_non_raw_key_variants() -> None:
    raw = concept_column_candidates(C_IC1_X_POS_RAW, position_key=POSITION_KEY_G3_RAW)
    assert "r_ic1_x_spot_position_raw" in raw
    assert "r_ic1_x_spot_position" not in raw


def test_canonical_aliases_do_not_swallow_processed_spot_position() -> None:
    aliases = canonical_column_aliases()
    raw_alts = aliases.get("r_ic1_x_spot_position_raw", ())
    assert "r_ic1_x_spot_position" not in raw_alts


def test_valid_g3_fit_values_rejects_negative() -> None:
    vals = np.array([-1.0, 0.0, 1.5, np.nan])
    clean = valid_g3_fit_values(vals)
    assert np.isnan(clean[0])
    assert clean[1] == 0.0
    assert clean[2] == 1.5


def test_resolve_g3_fit_ok_prefers_spot_position_ok_alias() -> None:
    columns = (
        "r_ic1_x_confidence",
        "r_ic1_x_spot_error_code",
        "r_ic1_x_position_ok",
        "r_ic1_x_spot_position_ok",
    )
    assert resolve_g3_fit_ok_column(columns, "ic1_x") == "r_ic1_x_spot_position_ok"
    quality = resolve_g3_quality_columns(columns)
    assert quality.ic1_x_fit_ok == "r_ic1_x_spot_position_ok"


def test_g3_position_minus_target_from_timeslice() -> None:
    from scan_kit.common.session_source import load_session_timeslice_device_units, resolve_session_source

    sid = "1091134775"
    base = str(TEST_DATA)
    src = resolve_session_source(sid, base)
    if src is None:
        return

    frames = load_session_timeslice_device_units(src)
    df = frames[71]
    cols = resolve_g3_position_target_columns(df.columns)
    assert cols is not None

    sno = 7101
    on = (df["rci_in_trigger"] == 1) & (df["spot_no"].astype(int) == sno)
    idx = on.values.nonzero()[0]
    start, end = int(idx[0]), int(idx[-1]) + 1

    errors = g3_position_error_arrays(df, start, end, cols)
    assert errors is not None
    ic1_x_err, _, _, _ = errors

    pos = valid_g3_fit_values(df[cols.ic1_x].values[start:end])
    tgt = valid_g3_fit_values(df[cols.ic1_x_target].values[start:end])
    expected = pos - tgt
    np.testing.assert_allclose(ic1_x_err, expected, equal_nan=True)


def test_g3_timeslice_resolves_position_target_columns() -> None:
    path = (
        TEST_DATA / "1091134775" / "1091134775" / "layer-71" / "run-0"
        / "timeslice_data_device_units.csv"
    )
    if not path.is_file():
        return

    df = canonicalize_dataframe_columns(pd.read_csv(path, nrows=32))
    cols = resolve_g3_position_target_columns(df.columns)
    assert cols is not None
    assert cols.ic2_x == "r_ic2_x_position"
    assert cols.ic2_x_target == "ic2_position_x_target"
    assert resolve_concept_column(df.columns, C_IC1_X_POS, position_key=POSITION_KEY_G3)
