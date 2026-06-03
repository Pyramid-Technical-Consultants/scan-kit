"""Tests for plan synthesis template generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scan_kit.workflows.plan_synthesis.input_map import INPUT_MAP_COLUMNS, write_input_map_csv
from scan_kit.workflows.plan_synthesis.layouts.rectangular_field import rectangular_grid_positions
from scan_kit.workflows.plan_synthesis.registry import get_template


@pytest.fixture
def zero_field():
    template = get_template("zero_field")
    assert template is not None
    return template


@pytest.fixture
def rectangular_field():
    template = get_template("rectangular_field")
    assert template is not None
    return template


def test_zero_field_generates_origin_spots(zero_field) -> None:
    params = {
        "selected_energies": [250.0, 247.5, 245.0],
        "charge_req_mu": 0.02,
        "spots_per_layer": 10,
    }
    assert zero_field.validate(params) == []
    df = zero_field.generate(params)
    assert len(df) == 30
    assert list(df.columns) == list(INPUT_MAP_COLUMNS)
    assert (df["X_POSITION"] == 0.0).all()
    assert (df["Y_POSITION"] == 0.0).all()
    assert (df["CURRENT"] == 0.0).all()
    assert (df["VELOCITY"] == 0.0).all()
    assert (df["beam_off"] == 1.0).all()
    assert (df["map_checksum"] == 0).all()
    assert (df[""] == "").all()
    assert df["spot_no"].tolist() == list(range(30))
    assert df["layer_id"].nunique() == 3
    assert df["ENERGY"].tolist()[0] == 250.0
    assert df["ENERGY"].tolist()[-1] == 245.0


def test_format_plan_summary_notes_preview_cap() -> None:
    from scan_kit.workflows.plan_synthesis.preview import format_plan_summary

    df = pd.DataFrame({"ENERGY": [1.0] * 5, "CHARGE_REQ": [0.01] * 5})
    assert "preview shows" not in format_plan_summary(df, preview_row_cap=10)
    assert "preview shows first 3 rows" in format_plan_summary(df, preview_row_cap=3)


def test_zero_field_validation_rejects_empty_energies(zero_field) -> None:
    params = zero_field.default_params()
    params["selected_energies"] = []
    errors = zero_field.validate(params)
    assert any("energy" in e.lower() for e in errors)


def test_rectangular_field_grid_positions() -> None:
    positions = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=3,
    )
    assert len(positions) == 9
    xs = sorted({p[0] for p in positions})
    ys = sorted({p[1] for p in positions})
    assert xs == pytest.approx([-10.0, 0.0, 10.0])
    assert ys == pytest.approx([-10.0, 0.0, 10.0])


def test_rectangular_field_generates_grid(rectangular_field) -> None:
    params = {
        "selected_energies": [200.0],
        "charge_req_mu": 0.05,
        "center_x_mm": 0.0,
        "center_y_mm": 0.0,
        "field_width_mm": 20.0,
        "field_height_mm": 20.0,
        "spots_x": 3,
        "spots_y": 3,
    }
    assert rectangular_field.validate(params) == []
    df = rectangular_field.generate(params)
    assert len(df) == 9
    assert df["ENERGY"].nunique() == 1
    assert len(df[["X_POSITION", "Y_POSITION"]].drop_duplicates()) == 9


def test_rectangular_field_validation_rejects_zero_grid(rectangular_field) -> None:
    params = rectangular_field.default_params()
    params["selected_energies"] = [200.0]
    params["spots_x"] = 0
    errors = rectangular_field.validate(params)
    assert errors


def test_write_input_map_csv_round_trip(tmp_path: Path, zero_field) -> None:
    params = {
        "selected_energies": [250.0, 247.5],
        "charge_req_mu": 0.02,
        "spots_per_layer": 2,
    }
    df = zero_field.generate(params)
    out = tmp_path / "input_map.csv"
    write_input_map_csv(df, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 4
    assert list(loaded.columns[: len(INPUT_MAP_COLUMNS) - 1]) == list(INPUT_MAP_COLUMNS[:-1])
