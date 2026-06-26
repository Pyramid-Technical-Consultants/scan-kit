"""Tests for correcting-coil amplifier column resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scan_kit.common.schema import (
    C_AMPLIFIER_CMD_X,
    C_AMPLIFIER_CMD_Y,
    C_AMPLIFIER_READBACK_X,
    C_AMPLIFIER_READBACK_Y,
    TIMESLICE_AMPLIFIER_FIELD_COLS,
    resolve_concept_column,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G2_LAYER = TEST_DATA / "590658542" / "layer-9" / "run-0"


def test_g3_amplifier_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "1943968267" / "1943968267" / "layer-0" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_X) == "amplifier_x_target"
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_Y) == "amplifier_y_target"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_X) == "amplifier_x_readback"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_Y) == "amplifier_y_readback"


def test_g2_amplifier_column_aliases() -> None:
    cols = pd.read_csv(
        G2_LAYER / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_X) == "c_x"
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_Y) == "c_y"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_X) == "r_xI"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_Y) == "r_yI"


def test_g2_amplifier_columns_also_on_883144654() -> None:
    cols = pd.read_csv(
        TEST_DATA / "883144654" / "layer-9" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_X) == "c_x"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_X) == "r_xI"


def test_g2_amplifier_readback_is_volts_in_device_units_only() -> None:
    """G2 r_xI/r_yI must be read from timeslice_data_device_units.csv."""
    raw = pd.read_csv(G2_LAYER / "timeslice_data.csv", usecols=["r_xI", "r_yI"])
    dev = pd.read_csv(G2_LAYER / "timeslice_data_device_units.csv", usecols=["r_xI", "r_yI"])
    # Raw register values span tens of amps; device_units are volt-scale (~±2).
    assert raw["r_xI"].abs().max() > 10
    assert dev["r_xI"].abs().max() < 5
    assert not raw["r_xI"].equals(dev["r_xI"])


def test_timeslice_amplifier_field_cols_cover_g2_and_g3_device_units() -> None:
    g2_cols = pd.read_csv(G2_LAYER / "timeslice_data_device_units.csv", nrows=0).columns
    g3_cols = pd.read_csv(
        TEST_DATA / "1943968267" / "1943968267" / "layer-0" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    for cols in (g2_cols, g3_cols):
        assert resolve_concept_column(cols, C_AMPLIFIER_CMD_X) is not None
        assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_X) is not None
    assert "c_x" in TIMESLICE_AMPLIFIER_FIELD_COLS
    assert "amplifier_x_target" in TIMESLICE_AMPLIFIER_FIELD_COLS
    assert "r_xI" in TIMESLICE_AMPLIFIER_FIELD_COLS
    assert "amplifier_x_readback" in TIMESLICE_AMPLIFIER_FIELD_COLS
    assert "r_xV" in TIMESLICE_AMPLIFIER_FIELD_COLS
    assert "r_xB" in TIMESLICE_AMPLIFIER_FIELD_COLS


def test_older_g3_tx2_column_aliases() -> None:
    cols = pd.read_csv(
        TEST_DATA / "1262268206" / "1262268206" / "layer-0" / "run-0"
        / "timeslice_data_device_units.csv",
        nrows=0,
    ).columns
    assert resolve_concept_column(cols, C_AMPLIFIER_CMD_X) == "c_x"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_X) == "r_xV"
    assert resolve_concept_column(cols, C_AMPLIFIER_READBACK_Y) == "r_yV"
    from scan_kit.common.schema import C_MAG_FIELD_X, C_MAG_FIELD_Y

    assert resolve_concept_column(cols, C_MAG_FIELD_X) == "r_xB"
    assert resolve_concept_column(cols, C_MAG_FIELD_Y) == "r_yB"
