"""Tests for plan synthesis template generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scan_kit.workflows.plan_synthesis.input_map import (
    DEFAULT_CURRENT_A,
    INPUT_MAP_COLUMNS,
    INPUT_MAP_EXPORT_COLUMNS,
    order_input_map_by_energy,
    write_input_map_csv,
)
from scan_kit.workflows.plan_synthesis.energies import (
    STANDARD_ENERGIES_MEV,
    TEN_MEV_STEP_ENERGIES_MEV,
    WHOLE_MEV_STEP_ENERGIES_MEV,
)
from scan_kit.workflows.plan_synthesis.layouts.rectangular_field import (
    FAST_AXIS_X,
    FAST_AXIS_Y,
    LAYER_TRANSITION_CONTINUE,
    LAYER_TRANSITION_RESET,
    START_CORNER_BOTTOM_LEFT,
    START_CORNER_BOTTOM_RIGHT,
    START_CORNER_TOP_LEFT,
    START_CORNER_TOP_RIGHT,
    positions_for_layer,
    rectangular_grid_positions,
)
from scan_kit.workflows.plan_synthesis.registry import get_template
from scan_kit.workflows.plan_synthesis.spot_weight import (
    SPOT_WEIGHT_METHOD_EVEN_TOTAL,
    SPOT_WEIGHT_METHOD_FIXED,
    SPOT_WEIGHT_METHOD_LAYER_EVEN,
    SPOT_WEIGHT_METHOD_RANDOM,
    SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
    compute_spot_weights,
    validate_spot_weight_params,
)
from scan_kit.workflows.plan_synthesis.generators.base import SpotRow


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


def _fixed_weight_params(**overrides) -> dict:
    params = {
        "spot_weight_method": SPOT_WEIGHT_METHOD_FIXED,
        "spot_weight_mu": 0.02,
        "spot_weight_total_mu": 1.0,
        "spot_weight_variance_pct": 10.0,
        "spot_weight_min_mu": 0.002,
        "spot_weight_max_mu": 0.1,
        "spot_weight_layer_shuffle": False,
    }
    params.update(overrides)
    return params


def test_ten_mev_step_energies_are_catalog_subset() -> None:
    assert TEN_MEV_STEP_ENERGIES_MEV[0] == 70.0
    assert TEN_MEV_STEP_ENERGIES_MEV[-1] == 250.0
    assert all(energy % 10 == 0 for energy in TEN_MEV_STEP_ENERGIES_MEV)
    assert set(TEN_MEV_STEP_ENERGIES_MEV).issubset(set(STANDARD_ENERGIES_MEV))


def test_whole_mev_step_energies_are_catalog_subset() -> None:
    assert WHOLE_MEV_STEP_ENERGIES_MEV[0] == 70.0
    assert WHOLE_MEV_STEP_ENERGIES_MEV[-1] == 250.0
    assert all(energy % 1 == 0 for energy in WHOLE_MEV_STEP_ENERGIES_MEV)
    assert set(WHOLE_MEV_STEP_ENERGIES_MEV).issubset(set(STANDARD_ENERGIES_MEV))
    assert 102.5 not in WHOLE_MEV_STEP_ENERGIES_MEV
    assert 105.0 in WHOLE_MEV_STEP_ENERGIES_MEV
    assert set(TEN_MEV_STEP_ENERGIES_MEV).issubset(set(WHOLE_MEV_STEP_ENERGIES_MEV))


def test_zero_field_generates_origin_spots(zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 247.5, 245.0],
        spots_per_layer=10,
    )
    assert zero_field.validate(params) == []
    df = zero_field.generate(params)
    assert len(df) == 30
    assert list(df.columns) == list(INPUT_MAP_COLUMNS)
    assert (df["X_POSITION"] == 0.0).all()
    assert (df["Y_POSITION"] == 0.0).all()
    assert (df["CHARGE_REQ"] == 0.02).all()
    assert (df["CURRENT"] == DEFAULT_CURRENT_A).all()
    assert (df["BEAM_SIZE"] == 3.61).all()
    assert (df["VELOCITY"] == 0.0).all()
    assert df["spot_no"].tolist() == list(range(30))
    assert df["layer_id"].nunique() == 3
    assert df["ENERGY"].tolist()[0] == 250.0
    assert df["ENERGY"].tolist()[-1] == 245.0


def test_format_plan_summary_notes_preview_cap() -> None:
    from scan_kit.workflows.plan_synthesis.preview import format_plan_summary

    df = pd.DataFrame({"ENERGY": [1.0] * 5, "CHARGE_REQ": [0.01] * 5})
    assert "preview shows" not in format_plan_summary(df, preview_row_cap=10)
    assert "preview shows first 3 rows" in format_plan_summary(df, preview_row_cap=3)


def test_format_plan_summary_total_mu_uses_three_sig_figs() -> None:
    from scan_kit.workflows.plan_synthesis.preview import format_plan_summary

    df = pd.DataFrame({"ENERGY": [1.0] * 3, "CHARGE_REQ": [0.12345] * 3})
    assert "0.37 MU total" in format_plan_summary(df)

    df_large = pd.DataFrame({"ENERGY": [1.0] * 2, "CHARGE_REQ": [1234.567] * 2})
    assert "2.47e+03 MU total" in format_plan_summary(df_large)


def test_format_plan_summary_includes_estimated_delivery_time() -> None:
    import pytest

    from scan_kit.workflows.plan_synthesis.preview import (
        estimate_delivery_seconds,
        format_delivery_duration,
        format_plan_summary,
    )

    assert estimate_delivery_seconds(0.8) == pytest.approx(2.0)
    assert format_delivery_duration(2.0) == "2.0 s"
    assert format_delivery_duration(125.0) == "2 min 5 s"
    assert format_delivery_duration(3660.0) == "1 h 1 min"

    df = pd.DataFrame({"ENERGY": [200.0, 200.0], "CHARGE_REQ": [0.4, 0.4]})
    summary = format_plan_summary(df)
    assert "0.8 MU total" in summary
    assert "est. 2.0 s delivery" in summary


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


def test_rectangular_field_serpentine_fast_x() -> None:
    positions = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=3,
        fast_axis=FAST_AXIS_X,
    )
    assert positions == pytest.approx(
        [
            (-10.0, 10.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, 0.0),
            (0.0, 0.0),
            (-10.0, 0.0),
            (-10.0, -10.0),
            (0.0, -10.0),
            (10.0, -10.0),
        ]
    )


def test_rectangular_field_serpentine_start_corners_fast_x() -> None:
    kwargs = dict(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=2,
        fast_axis=FAST_AXIS_X,
    )
    assert rectangular_grid_positions(
        **kwargs, start_corner=START_CORNER_BOTTOM_RIGHT
    ) == pytest.approx(
        [
            (10.0, -10.0),
            (0.0, -10.0),
            (-10.0, -10.0),
            (-10.0, 10.0),
            (0.0, 10.0),
            (10.0, 10.0),
        ]
    )
    assert rectangular_grid_positions(
        **kwargs, start_corner=START_CORNER_TOP_LEFT
    ) == pytest.approx(
        [
            (-10.0, 10.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, -10.0),
            (0.0, -10.0),
            (-10.0, -10.0),
        ]
    )
    assert rectangular_grid_positions(
        **kwargs, start_corner=START_CORNER_TOP_RIGHT
    ) == pytest.approx(
        [
            (10.0, 10.0),
            (0.0, 10.0),
            (-10.0, 10.0),
            (-10.0, -10.0),
            (0.0, -10.0),
            (10.0, -10.0),
        ]
    )


def test_rectangular_field_serpentine_start_corners_fast_y() -> None:
    kwargs = dict(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=2,
        spots_y=3,
        fast_axis=FAST_AXIS_Y,
    )
    assert rectangular_grid_positions(
        **kwargs, start_corner=START_CORNER_TOP_LEFT
    ) == pytest.approx(
        [
            (-10.0, 10.0),
            (-10.0, 0.0),
            (-10.0, -10.0),
            (10.0, -10.0),
            (10.0, 0.0),
            (10.0, 10.0),
        ]
    )
    assert rectangular_grid_positions(
        **kwargs, start_corner=START_CORNER_BOTTOM_RIGHT
    ) == pytest.approx(
        [
            (10.0, -10.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (-10.0, 10.0),
            (-10.0, 0.0),
            (-10.0, -10.0),
        ]
    )


def test_rectangular_field_serpentine_fast_y() -> None:
    positions = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=3,
        fast_axis=FAST_AXIS_Y,
    )
    assert positions == pytest.approx(
        [
            (-10.0, 10.0),
            (-10.0, 0.0),
            (-10.0, -10.0),
            (0.0, -10.0),
            (0.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (10.0, 0.0),
            (10.0, -10.0),
        ]
    )


def _manhattan_path_length(positions: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(positions, positions[1:]):
        total += abs(x1 - x0) + abs(y1 - y0)
    return total


def test_rectangular_field_serpentine_reduces_travel() -> None:
    kwargs = dict(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=100.0,
        field_height_mm=100.0,
        spots_x=11,
        spots_y=11,
    )
    serpentine = rectangular_grid_positions(**kwargs, fast_axis=FAST_AXIS_X)
    row_major = []
    xs = sorted({p[0] for p in serpentine})
    ys = sorted({p[1] for p in serpentine})
    for y in ys:
        for x in xs:
            row_major.append((x, y))
    assert _manhattan_path_length(serpentine) < _manhattan_path_length(row_major)


def test_rectangular_field_generates_grid(rectangular_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[200.0],
        spot_weight_mu=0.05,
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=3,
    )
    assert rectangular_field.validate(params) == []
    df = rectangular_field.generate(params)
    assert len(df) == 9
    assert df["ENERGY"].nunique() == 1
    assert len(df[["X_POSITION", "Y_POSITION"]].drop_duplicates()) == 9
    assert (df["CHARGE_REQ"] == 0.05).all()


def test_rectangular_field_validation_rejects_zero_grid(rectangular_field) -> None:
    params = rectangular_field.default_params()
    params["selected_energies"] = [200.0]
    params["spots_x"] = 0
    errors = rectangular_field.validate(params)
    assert errors


def test_zero_field_layers_ordered_high_to_low(zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[70.0, 250.0, 100.0, 200.0],
        spots_per_layer=2,
    )
    df = zero_field.generate(params)
    assert df["ENERGY"].tolist() == [
        250.0,
        250.0,
        200.0,
        200.0,
        100.0,
        100.0,
        70.0,
        70.0,
    ]
    assert df["spot_no"].tolist() == list(range(len(df)))


def test_positions_for_layer_continue_alternates_direction() -> None:
    base = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert positions_for_layer(
        base, 0, layer_transition=LAYER_TRANSITION_CONTINUE
    ) == base
    assert positions_for_layer(
        base, 1, layer_transition=LAYER_TRANSITION_CONTINUE
    ) == list(reversed(base))
    assert positions_for_layer(
        base, 2, layer_transition=LAYER_TRANSITION_CONTINUE
    ) == base
    assert positions_for_layer(
        base, 1, layer_transition=LAYER_TRANSITION_RESET
    ) == base


def test_rectangular_field_layer_transition_continue(rectangular_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 200.0],
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=2,
        spots_y=2,
        layer_transition=LAYER_TRANSITION_CONTINUE,
    )
    df = rectangular_field.generate(params)
    base = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=2,
        spots_y=2,
        fast_axis=FAST_AXIS_X,
    )
    high_layer = df[df["ENERGY"] == 250.0]
    low_layer = df[df["ENERGY"] == 200.0]
    high_positions = [
        (float(x), float(y))
        for x, y in high_layer[["X_POSITION", "Y_POSITION"]].itertuples(index=False)
    ]
    low_positions = [
        (float(x), float(y))
        for x, y in low_layer[["X_POSITION", "Y_POSITION"]].itertuples(index=False)
    ]
    assert high_positions == pytest.approx(base)
    assert low_positions == pytest.approx(list(reversed(base)))
    assert high_positions[-1] == low_positions[0]


def test_rectangular_field_preserves_grid_within_layer(rectangular_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[200.0, 250.0],
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=2,
        spots_y=2,
    )
    df = rectangular_field.generate(params)
    high_layer = df[df["ENERGY"] == 250.0]
    low_layer = df[df["ENERGY"] == 200.0]
    assert len(high_layer) == 4
    assert len(low_layer) == 4
    assert high_layer.index.tolist()[0] < low_layer.index.tolist()[0]
    expected_positions = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=2,
        spots_y=2,
        fast_axis=FAST_AXIS_X,
    )
    actual_positions = [
        (float(x), float(y))
        for x, y in high_layer[["X_POSITION", "Y_POSITION"]].itertuples(index=False)
    ]
    assert actual_positions == pytest.approx(expected_positions)


def test_write_input_map_csv_reorders_layers(tmp_path: Path, zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 70.0],
        spots_per_layer=1,
    )
    df = zero_field.generate(params)
    shuffled = pd.concat([df[df["ENERGY"] == 70.0], df[df["ENERGY"] == 250.0]])
    shuffled = shuffled.copy()
    shuffled["spot_no"] = [9, 0]

    out = tmp_path / "input_map.csv"
    write_input_map_csv(shuffled, out)
    loaded = pd.read_csv(out)

    assert list(loaded.columns) == list(INPUT_MAP_EXPORT_COLUMNS)
    assert loaded["ENERGY(MeV)"].tolist() == [250.0, 70.0]
    assert loaded["#NO"].tolist() == [1, 2]


def test_order_input_map_by_energy_is_stable() -> None:
    df = pd.DataFrame(
        {
            "ENERGY": [100.0, 100.0, 200.0, 200.0],
            "CURRENT": [0.0] * 4,
            "BEAM_SIZE": [3.61] * 4,
            "X_POSITION": [1.0, 2.0, 3.0, 4.0],
            "Y_POSITION": [0.0] * 4,
            "CHARGE_REQ": [0.01] * 4,
            "VELOCITY": [0.0] * 4,
            "spot_no": [0, 1, 2, 3],
            "layer_id": [1, 1, 2, 2],
        }
    )
    ordered = order_input_map_by_energy(df)
    assert ordered["ENERGY"].tolist() == [200.0, 200.0, 100.0, 100.0]
    assert ordered["X_POSITION"].tolist() == pytest.approx([3.0, 4.0, 1.0, 2.0])
    assert ordered["spot_no"].tolist() == [0, 1, 2, 3]


def test_write_input_map_csv_header_order(tmp_path: Path, zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0],
        spots_per_layer=1,
    )
    df = zero_field.generate(params)
    out = tmp_path / "input_map.csv"
    write_input_map_csv(df, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(INPUT_MAP_EXPORT_COLUMNS)


def test_write_input_map_csv_round_trip(tmp_path: Path, zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 247.5],
        spots_per_layer=2,
    )
    df = zero_field.generate(params)
    out = tmp_path / "input_map.csv"
    write_input_map_csv(df, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 4
    assert list(loaded.columns) == list(INPUT_MAP_EXPORT_COLUMNS)
    assert loaded["#NO"].tolist() == [1, 2, 3, 4]


def test_spot_weight_fixed() -> None:
    rows = [
        SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0),
        SpotRow(energy=200.0, layer_id=1, x_position=1.0, y_position=0.0),
    ]
    weights = compute_spot_weights(
        rows,
        _fixed_weight_params(spot_weight_mu=0.03),
    )
    assert weights == [0.03, 0.03]


def test_spot_weight_random_range() -> None:
    rows = [SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0)] * 20
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_RANDOM,
        spot_weight_min_mu=0.01,
        spot_weight_max_mu=0.02,
    )
    weights = compute_spot_weights(rows, params)
    assert len(weights) == 20
    assert all(0.01 <= weight <= 0.02 for weight in weights)


def test_spot_weight_layer_even_range() -> None:
    rows = [
        SpotRow(energy=250.0, layer_id=10, x_position=0.0, y_position=0.0),
        SpotRow(energy=250.0, layer_id=10, x_position=1.0, y_position=0.0),
        SpotRow(energy=250.0, layer_id=10, x_position=2.0, y_position=0.0),
        SpotRow(energy=247.5, layer_id=11, x_position=0.0, y_position=0.0),
        SpotRow(energy=247.5, layer_id=11, x_position=1.0, y_position=0.0),
    ]
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_LAYER_EVEN,
        spot_weight_min_mu=0.01,
        spot_weight_max_mu=0.03,
        spot_weight_layer_shuffle=False,
    )
    weights = compute_spot_weights(rows, params)
    assert weights[:3] == pytest.approx([0.01, 0.02, 0.03])
    assert weights[3:] == pytest.approx([0.01, 0.03])


def test_spot_weight_layer_even_shuffle_per_layer() -> None:
    import random

    rows = [
        SpotRow(energy=250.0, layer_id=10, x_position=0.0, y_position=0.0),
        SpotRow(energy=250.0, layer_id=10, x_position=1.0, y_position=0.0),
        SpotRow(energy=250.0, layer_id=10, x_position=2.0, y_position=0.0),
        SpotRow(energy=247.5, layer_id=11, x_position=0.0, y_position=0.0),
        SpotRow(energy=247.5, layer_id=11, x_position=1.0, y_position=0.0),
        SpotRow(energy=247.5, layer_id=11, x_position=2.0, y_position=0.0),
    ]
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_LAYER_EVEN,
        spot_weight_min_mu=0.01,
        spot_weight_max_mu=0.03,
        spot_weight_layer_shuffle=True,
    )

    random.seed(7)
    weights = compute_spot_weights(rows, params)
    assert sorted(weights[:3]) == pytest.approx([0.01, 0.02, 0.03])
    assert sorted(weights[3:]) == pytest.approx([0.01, 0.02, 0.03])
    assert weights[:3] != pytest.approx([0.01, 0.02, 0.03])
    assert weights[3:] != pytest.approx([0.01, 0.02, 0.03])

    random.seed(7)
    again = compute_spot_weights(rows, params)
    assert again == weights


def test_spot_weight_even_total() -> None:
    rows = [SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0)] * 4
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_EVEN_TOTAL,
        spot_weight_total_mu=1.0,
    )
    weights = compute_spot_weights(rows, params)
    assert len(weights) == 4
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(weights) == pytest.approx(1.0)


def test_spot_weight_even_total_absorbs_rounding_remainder() -> None:
    rows = [SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0)] * 3
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_EVEN_TOTAL,
        spot_weight_total_mu=1.0,
    )
    weights = compute_spot_weights(rows, params)
    assert sum(weights) == pytest.approx(1.0)
    assert all(weight > 0 for weight in weights)


def test_zero_field_even_total_integration(zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 247.5],
        spots_per_layer=2,
        spot_weight_method=SPOT_WEIGHT_METHOD_EVEN_TOTAL,
        spot_weight_total_mu=2.0,
    )
    assert zero_field.validate(params) == []
    df = zero_field.generate(params)
    assert len(df) == 4
    assert df["CHARGE_REQ"].sum() == pytest.approx(2.0)


def test_spot_weight_random_total_with_zero_variance_matches_even_total() -> None:
    rows = [SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0)] * 4
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
        spot_weight_total_mu=1.0,
        spot_weight_variance_pct=0.0,
    )
    weights = compute_spot_weights(rows, params)
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(weights) == pytest.approx(1.0)


def test_spot_weight_random_total_respects_target_and_variance() -> None:
    rows = [SpotRow(energy=200.0, layer_id=1, x_position=0.0, y_position=0.0)] * 50
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
        spot_weight_total_mu=5.0,
        spot_weight_variance_pct=20.0,
    )
    weights = compute_spot_weights(rows, params)
    assert len(weights) == 50
    assert sum(weights) == pytest.approx(5.0)
    assert all(weight > 0 for weight in weights)
    assert len(set(weights)) > 1
    mean = 5.0 / 50
    assert min(weights) < mean
    assert max(weights) > mean


def test_spot_weight_validation_rejects_excessive_variance() -> None:
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_RANDOM_TOTAL,
        spot_weight_variance_pct=150.0,
    )
    errors = validate_spot_weight_params(params)
    assert any("Spot Variance" in error for error in errors)


def test_spot_weight_validation_rejects_inverted_range() -> None:
    params = _fixed_weight_params(
        spot_weight_method=SPOT_WEIGHT_METHOD_RANDOM,
        spot_weight_min_mu=0.05,
        spot_weight_max_mu=0.01,
    )
    errors = validate_spot_weight_params(params)
    assert any("Maximum Weight (MU)" in error for error in errors)


def test_both_templates_expose_spot_weight_method(zero_field, rectangular_field) -> None:
    for template in (zero_field, rectangular_field):
        keys = {spec.key for spec in template.param_specs()}
        assert "spot_weight_method" in keys
        assert "spot_weight_mu" in keys
        assert "spot_weight_total_mu" in keys
        assert "spot_weight_variance_pct" in keys
        assert "spot_weight_min_mu" in keys
        assert "spot_weight_max_mu" in keys
        assert "spot_weight_layer_shuffle" in keys


def test_rectangular_field_layer_transition_button_group(rectangular_field) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QRadioButton

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    form = ParamFormWidget(rectangular_field.param_specs(), rectangular_field.default_params())
    app.processEvents()

    transition_buttons = {
        button.text(): button
        for button in form.findChildren(QRadioButton)
        if button.text() in {"Reset to Corner", "Continue from End"}
    }
    assert set(transition_buttons) == {"Reset to Corner", "Continue from End"}
    assert transition_buttons["Reset to Corner"].isChecked()

    transition_buttons["Continue from End"].click()
    app.processEvents()
    assert form.read_params()["layer_transition"] == LAYER_TRANSITION_CONTINUE


def test_rectangular_field_start_corner_button_group(rectangular_field) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QRadioButton

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    form = ParamFormWidget(rectangular_field.param_specs(), rectangular_field.default_params())
    app.processEvents()

    corner_buttons = {
        button.text(): button
        for button in form.findChildren(QRadioButton)
        if button.text() in {"Top Left", "Top Right", "Bottom Left", "Bottom Right"}
    }
    assert set(corner_buttons) == {
        "Top Left",
        "Top Right",
        "Bottom Left",
        "Bottom Right",
    }
    assert corner_buttons["Top Left"].isChecked()

    corner_buttons["Top Right"].click()
    app.processEvents()
    assert form.read_params()["start_corner"] == START_CORNER_TOP_RIGHT


def test_rectangular_field_fast_axis_button_group(rectangular_field) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QRadioButton

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    params = rectangular_field.default_params()
    params["fast_axis"] = FAST_AXIS_X
    form = ParamFormWidget(rectangular_field.param_specs(), params)
    app.processEvents()

    axis_buttons = {
        button.text(): button
        for button in form.findChildren(QRadioButton)
        if button.text() in {"X", "Y"}
    }
    assert set(axis_buttons) == {"X", "Y"}
    assert axis_buttons["X"].isChecked()
    assert not axis_buttons["Y"].isChecked()

    axis_buttons["Y"].click()
    app.processEvents()
    assert form.read_params()["fast_axis"] == FAST_AXIS_Y


def test_rectangular_field_geometry_quick_set_buttons(rectangular_field) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QPushButton

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    params = rectangular_field.default_params()
    params["field_width_mm"] = 50.0
    params["field_height_mm"] = 60.0
    params["spots_x"] = 5
    params["spots_y"] = 6
    params["center_x_mm"] = 10.0
    params["center_y_mm"] = -5.0
    form = ParamFormWidget(rectangular_field.param_specs(), params)
    app.processEvents()

    button_labels = {button.text() for button in form.findChildren(QPushButton)}
    assert {"100", "200", "250", "300"}.issubset(button_labels)
    assert {"3", "7", "11", "33"}.issubset(button_labels)
    assert "0,0" in button_labels

    for button in form.findChildren(QPushButton):
        if button.text() == "250":
            button.click()
            break
    app.processEvents()
    updated = form.read_params()
    assert updated["field_width_mm"] == 250.0
    assert updated["field_height_mm"] == 250.0

    for button in form.findChildren(QPushButton):
        if button.text() == "11":
            button.click()
            break
    app.processEvents()
    updated = form.read_params()
    assert updated["spots_x"] == 11
    assert updated["spots_y"] == 11

    for button in form.findChildren(QPushButton):
        if button.text() == "0,0":
            button.click()
            break
    app.processEvents()
    updated = form.read_params()
    assert updated["center_x_mm"] == 0.0
    assert updated["center_y_mm"] == 0.0


def test_plan_synthesis_panel_rebuilds_form_on_template_switch() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QGroupBox

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget
    from scan_kit.workflows.plan_synthesis_panel import PlanSynthesisPanel

    app = QApplication.instance() or QApplication(sys.argv)
    panel = PlanSynthesisPanel()
    panel.show()
    app.processEvents()

    first_form = panel._param_scroll.widget()
    assert isinstance(first_form, ParamFormWidget)
    assert panel._param_form is first_form
    assert first_form.isVisible()
    assert first_form.height() > 0
    assert first_form.findChildren(QGroupBox)

    panel._template_list.setCurrentRow(1)
    app.processEvents()

    second_form = panel._param_scroll.widget()
    assert isinstance(second_form, ParamFormWidget)
    assert second_form is not first_form
    assert panel._param_form is second_form
    assert second_form.isVisible()
    assert second_form.height() > 0
    assert second_form.findChildren(QGroupBox)

    panel._template_list.setCurrentRow(0)
    app.processEvents()

    third_form = panel._param_scroll.widget()
    assert isinstance(third_form, ParamFormWidget)
    assert third_form is not second_form
    assert third_form.isVisible()
    assert third_form.height() > 0


def test_param_form_field_sets_visible_before_show(zero_field) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    host = QWidget()
    layout = QVBoxLayout(host)
    form = ParamFormWidget(zero_field.param_specs(), zero_field.default_params())
    layout.addWidget(form)
    host.show()
    app.processEvents()

    assert form._field_set_boxes["energy"].isVisible()
    assert form._field_set_boxes["geometry"].isVisible()
    assert form._field_set_boxes["weight"].isVisible()
    assert not form._field_set_boxes["energy"].isHidden()


def _preview_table_df(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ENERGY": [200.0] * n_rows,
            "CURRENT": [DEFAULT_CURRENT_A] * n_rows,
            "BEAM_SIZE": [3.61] * n_rows,
            "X_POSITION": [0.0] * n_rows,
            "Y_POSITION": [0.0] * n_rows,
            "CHARGE_REQ": [0.01] * n_rows,
            "VELOCITY": [0.0] * n_rows,
            "spot_no": list(range(n_rows)),
            "layer_id": [1] * n_rows,
        }
    )


def test_format_layer_selection_status() -> None:
    from scan_kit.workflows.plan_synthesis.energy_picker import (
        _format_layer_selection_status,
    )

    assert _format_layer_selection_status(0) == "No layers selected"
    assert _format_layer_selection_status(1) == "1 layer selected"
    assert _format_layer_selection_status(3) == "3 layers selected"


def test_energy_picker_status_label_updates() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.plan_synthesis.energy_picker import EnergyPickerWidget

    app = QApplication.instance() or QApplication(sys.argv)
    picker = EnergyPickerWidget(selected=[250.0, 247.5])
    app.processEvents()

    assert picker._status_label.text() == "2 layers selected"

    picker._energy_list.clearSelection()
    app.processEvents()
    assert picker._status_label.text() == "No layers selected"

    picker._energy_list.selectAll()
    app.processEvents()
    assert picker._status_label.text().endswith(" layers selected")


def test_energy_picker_whole_mev_steps_button() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.plan_synthesis.energy_picker import EnergyPickerWidget

    app = QApplication.instance() or QApplication(sys.argv)
    picker = EnergyPickerWidget(selected=[250.0])
    app.processEvents()

    picker._select_whole_mev_steps()
    app.processEvents()

    assert picker.selected_energies() == list(WHOLE_MEV_STEP_ENERGIES_MEV)[::-1]
    assert picker._status_label.text() == f"{len(WHOLE_MEV_STEP_ENERGIES_MEV)} layers selected"


def test_suggest_input_map_filename_zero_field(zero_field) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    params = _fixed_weight_params(
        selected_energies=[250.0, 247.5, 245.0],
        spots_per_layer=100,
    )
    name = suggest_input_map_filename(zero_field, params)
    assert name.endswith(".csv")
    assert len(name) <= 128
    assert name.startswith("ZeroField_")
    assert "E250-245" in name
    assert "Sp100" in name
    assert "Wfix0.02" in name


def test_suggest_input_map_filename_rectangular_field(rectangular_field) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    params = _fixed_weight_params(
        selected_energies=[200.0],
        center_x_mm=10.0,
        center_y_mm=-5.0,
        field_width_mm=100.0,
        field_height_mm=80.0,
        spots_x=33,
        spots_y=33,
        spot_weight_method=SPOT_WEIGHT_METHOD_EVEN_TOTAL,
        spot_weight_total_mu=2.0,
    )
    name = suggest_input_map_filename(rectangular_field, params)
    assert len(name) <= 128
    assert name.startswith("RectField_")
    assert "E200" in name
    assert "C10x-5" in name
    assert "100x80mm" in name
    assert "G33x33" in name
    assert "Wtot2" in name


def test_suggest_input_map_filename_all_catalog_energies(zero_field) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    params = zero_field.default_params()
    params["selected_energies"] = list(STANDARD_ENERGIES_MEV)
    name = suggest_input_map_filename(zero_field, params)
    assert "E250-70" in name
    assert "76L" not in name


def test_suggest_input_map_filename_many_noncontiguous_energies(zero_field) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    params = zero_field.default_params()
    params["selected_energies"] = [250.0, 240.0, 230.0, 220.0, 210.0]
    name = suggest_input_map_filename(zero_field, params)
    assert "E5L_250-210" in name


def test_suggest_input_map_filename_respects_max_length(zero_field) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    params = zero_field.default_params()
    params["selected_energies"] = list(STANDARD_ENERGIES_MEV)
    params["spots_per_layer"] = 100_000
    name = suggest_input_map_filename(zero_field, params, max_length=40)
    assert len(name) <= 40
    assert name.endswith(".csv")


def test_fill_preview_table_uses_export_column_order() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QTableWidget

    from scan_kit.workflows.plan_synthesis.preview import fill_preview_table

    app = QApplication.instance() or QApplication(sys.argv)
    table = QTableWidget()
    df = _preview_table_df(2)

    fill_preview_table(table, df)
    app.processEvents()

    assert [
        table.horizontalHeaderItem(col).text()
        for col in range(table.columnCount())
    ] == list(INPUT_MAP_EXPORT_COLUMNS)
    assert table.item(0, 0).text() == "1"
    assert table.item(1, 0).text() == "2"


def test_fill_preview_table_sizes_columns_to_header_text() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QHeaderView, QTableWidget

    from scan_kit.workflows.plan_synthesis.input_map import INPUT_MAP_EXPORT_COLUMNS
    from scan_kit.workflows.plan_synthesis.preview import (
        _HEADER_COLUMN_PADDING_PX,
        fill_preview_table,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    table = QTableWidget()
    fill_preview_table(table, _preview_table_df(2))
    app.processEvents()

    hh = table.horizontalHeader()
    fm = hh.fontMetrics()
    for col_idx, label in enumerate(INPUT_MAP_EXPORT_COLUMNS):
        assert hh.sectionResizeMode(col_idx) == QHeaderView.ResizeMode.Fixed
        assert table.columnWidth(col_idx) == (
            fm.horizontalAdvance(label) + _HEADER_COLUMN_PADDING_PX
        )


def test_fill_preview_table_caps_rows() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QTableWidget

    from scan_kit.workflows.plan_synthesis.preview import (
        PREVIEW_ROW_CAP,
        fill_preview_table,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    table = QTableWidget()
    df = _preview_table_df(PREVIEW_ROW_CAP + 500)

    shown = fill_preview_table(table, df)
    app.processEvents()

    assert shown == PREVIEW_ROW_CAP
    assert table.rowCount() == PREVIEW_ROW_CAP


def test_start_preview_table_fill_cancels_stale_fill() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QTableWidget

    from scan_kit.workflows.plan_synthesis.preview import (
        _count_filled_preview_cells,
        start_preview_table_fill,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    table = QTableWidget()
    df = _preview_table_df(1_000)
    current = {"ok": True}

    start_preview_table_fill(table, df, is_current=lambda: current["ok"])
    app.processEvents()

    filled_before_cancel = _count_filled_preview_cells(table)
    assert 0 < filled_before_cancel < 1_000 * len(INPUT_MAP_EXPORT_COLUMNS)

    current["ok"] = False
    for _ in range(20):
        app.processEvents()

    assert _count_filled_preview_cells(table) == filled_before_cancel


def test_clear_preview_table_releases_rows() -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QTableWidget

    from scan_kit.workflows.plan_synthesis.preview import (
        _count_filled_preview_cells,
        clear_preview_table,
        fill_preview_table,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    table = QTableWidget()
    fill_preview_table(table, _preview_table_df(50))
    app.processEvents()
    assert _count_filled_preview_cells(table) > 0

    clear_preview_table(table)
    app.processEvents()

    assert table.rowCount() == 0
    assert _count_filled_preview_cells(table) == 0


def test_generate_input_map_reports_progress(zero_field) -> None:
    params = _fixed_weight_params(
        selected_energies=[250.0, 247.5],
        spots_per_layer=2,
    )
    seen: list[int] = []
    zero_field.generate(params, progress=seen.append)
    assert seen[0] == 0
    assert seen[-1] == 100
    assert max(seen) == 100


@pytest.fixture
def dicom_rt_plan():
    template = get_template("dicom_rt_plan")
    assert template is not None
    return template


_T0G10_DCM = Path(r"c:\Users\MattNichols\Projects\spot-check\test_data\RN.15186535.T0G10.dcm")


def _make_minimal_rt_ion_dataset() -> object:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
    ds.SOPInstanceUID = generate_uid()
    ds.RTPlanLabel = "TESTPLAN"

    cp = Dataset()
    cp.NominalBeamEnergy = 100.0
    cp.NumberOfScanSpotPositions = 2
    cp.ScanSpotPositionMap = [1.0, 2.0, 3.0, 4.0]
    cp.ScanSpotMetersetWeights = [0.5, 0.25]
    cp.ScanningSpotSize = [4.0, 6.0]

    beam = Dataset()
    beam.IonControlPointSequence = Sequence([cp])
    ds.IonBeamSequence = Sequence([beam])
    return ds


def test_dicom_rt_plan_validate_requires_file(dicom_rt_plan) -> None:
    assert dicom_rt_plan.validate({"dicom_path": ""}) != []
    assert dicom_rt_plan.validate({"dicom_path": "missing.dcm"}) != []


def test_serpentine_order_matches_rectangular_fast_x_grid() -> None:
    from scan_kit.workflows.plan_synthesis.spot_order import serpentine_order_indices

    positions = rectangular_grid_positions(
        center_x_mm=0.0,
        center_y_mm=0.0,
        field_width_mm=20.0,
        field_height_mm=20.0,
        spots_x=3,
        spots_y=3,
        fast_axis=FAST_AXIS_X,
        start_corner=START_CORNER_BOTTOM_LEFT,
    )
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    ordered = [
        (xs[index], ys[index])
        for index in serpentine_order_indices(xs, ys, fast_axis=FAST_AXIS_X)
    ]
    assert ordered == positions


def test_dicom_rt_plan_minimize_travel_reorders_layer(
    dicom_rt_plan,
    tmp_path: Path,
) -> None:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid

    from scan_kit.workflows.plan_synthesis.spot_order import SPOT_ORDER_MINIMIZE_TRAVEL

    cp = Dataset()
    cp.NominalBeamEnergy = 100.0
    cp.NumberOfScanSpotPositions = 4
    cp.ScanSpotPositionMap = [20.0, 0.0, 0.0, 0.0, 20.0, 10.0, 0.0, 10.0]
    cp.ScanSpotMetersetWeights = [0.5, 0.5, 0.5, 0.5]

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
    ds.SOPInstanceUID = generate_uid()
    beam = Dataset()
    beam.IonControlPointSequence = Sequence([cp])
    ds.IonBeamSequence = Sequence([beam])

    dcm_path = tmp_path / "scrambled.dcm"
    pydicom.dcmwrite(dcm_path, ds, implicit_vr=True, little_endian=True)

    plan_params = {
        "dicom_path": str(dcm_path),
        "use_dicom_beam_size": False,
        "beam_size_override_mm": 3.61,
        "spot_order": SPOT_ORDER_MINIMIZE_TRAVEL,
        "fast_axis": FAST_AXIS_X,
    }
    df = dicom_rt_plan.generate(plan_params)
    assert list(zip(df["X_POSITION"], df["Y_POSITION"])) == [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (0.0, 10.0),
    ]


def test_dicom_rt_plan_plan_order_preserves_dicom_sequence(
    dicom_rt_plan,
    tmp_path: Path,
) -> None:
    import pydicom
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.uid import generate_uid

    from scan_kit.workflows.plan_synthesis.spot_order import SPOT_ORDER_PLAN

    cp = Dataset()
    cp.NominalBeamEnergy = 100.0
    cp.NumberOfScanSpotPositions = 3
    cp.ScanSpotPositionMap = [30.0, 0.0, 10.0, 0.0, 20.0, 0.0]
    cp.ScanSpotMetersetWeights = [0.5, 0.5, 0.5]

    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
    ds.SOPInstanceUID = generate_uid()
    beam = Dataset()
    beam.IonControlPointSequence = Sequence([cp])
    ds.IonBeamSequence = Sequence([beam])

    dcm_path = tmp_path / "ordered.dcm"
    pydicom.dcmwrite(dcm_path, ds, implicit_vr=True, little_endian=True)

    plan_params = {
        "dicom_path": str(dcm_path),
        "use_dicom_beam_size": False,
        "beam_size_override_mm": 3.61,
        "spot_order": SPOT_ORDER_PLAN,
        "fast_axis": FAST_AXIS_X,
    }
    df = dicom_rt_plan.generate(plan_params)
    assert df["X_POSITION"].tolist() == [30.0, 10.0, 20.0]


def test_dicom_rt_plan_generate_uses_beam_size_override(
    dicom_rt_plan,
    tmp_path: Path,
) -> None:
    import pydicom

    dcm_path = tmp_path / "plan.dcm"
    pydicom.dcmwrite(
        dcm_path,
        _make_minimal_rt_ion_dataset(),
        implicit_vr=True,
        little_endian=True,
    )

    params = {
        "dicom_path": str(dcm_path),
        "use_dicom_beam_size": False,
        "beam_size_override_mm": 7.5,
    }
    assert dicom_rt_plan.validate(params) == []
    df = dicom_rt_plan.generate(params)
    assert (df["BEAM_SIZE"] == 7.5).all()


def test_dicom_rt_plan_generate_from_synthetic_dataset(
    dicom_rt_plan,
    tmp_path: Path,
) -> None:
    import pydicom

    dcm_path = tmp_path / "plan.dcm"
    pydicom.dcmwrite(
        dcm_path,
        _make_minimal_rt_ion_dataset(),
        implicit_vr=True,
        little_endian=True,
    )

    params = {
        "dicom_path": str(dcm_path),
        "use_dicom_beam_size": True,
    }
    assert dicom_rt_plan.validate(params) == []
    df = dicom_rt_plan.generate(params)
    assert list(df.columns) == list(INPUT_MAP_COLUMNS)
    assert len(df) == 2
    assert df["ENERGY"].tolist() == [100.0, 100.0]
    assert df["X_POSITION"].tolist() == [1.0, 3.0]
    assert df["Y_POSITION"].tolist() == [2.0, 4.0]
    assert df["CHARGE_REQ"].tolist() == [0.5, 0.25]
    assert (df["BEAM_SIZE"] == 5.0).all()
    assert df["ENERGY"].is_monotonic_decreasing


@pytest.mark.skipif(not _T0G10_DCM.is_file(), reason="T0G10 DICOM fixture not available")
def test_dicom_rt_plan_generate_from_t0g10_example(dicom_rt_plan) -> None:
    params = {
        "dicom_path": str(_T0G10_DCM),
        "use_dicom_beam_size": True,
    }
    assert dicom_rt_plan.validate(params) == []
    df = dicom_rt_plan.generate(params)
    assert len(df) == 12_779
    assert df["ENERGY"].nunique() == 41
    assert df["CHARGE_REQ"].sum() == pytest.approx(98.27907323255204)
    assert df["BEAM_SIZE"].iloc[0] == pytest.approx(15.96453381, rel=1e-6)


def test_suggest_input_map_filename_dicom_rt_plan(dicom_rt_plan, tmp_path: Path) -> None:
    import pydicom

    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    dcm_path = tmp_path / "RN.15186535.T0G10.dcm"
    pydicom.dcmwrite(
        dcm_path,
        _make_minimal_rt_ion_dataset(),
        implicit_vr=True,
        little_endian=True,
    )

    params = {
        "dicom_path": str(dcm_path),
        "use_dicom_beam_size": True,
    }
    name = suggest_input_map_filename(dicom_rt_plan, params)
    assert name == "DicomPlan_TESTPLAN.csv"


def test_param_form_beam_size_override_visible_when_dicom_size_off(
    dicom_rt_plan,
) -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    form = ParamFormWidget(dicom_rt_plan.param_specs(), dicom_rt_plan.default_params())
    form.show()
    app.processEvents()

    def _override_row_visible() -> bool:
        row = next(
            r
            for r in form._form_rows
            if any(spec.key == "beam_size_override_mm" for spec in r["specs"])
        )
        return row["row"].isVisible()

    assert _override_row_visible() is False

    form._editors["use_dicom_beam_size"].setChecked(False)
    app.processEvents()
    assert _override_row_visible() is True
    assert form.read_params()["beam_size_override_mm"] == pytest.approx(3.61)


def test_param_form_file_path_reads_value(dicom_rt_plan) -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.workflows.plan_synthesis.param_form import ParamFormWidget

    app = QApplication.instance() or QApplication(sys.argv)
    params = dicom_rt_plan.default_params()
    params["dicom_path"] = r"C:\plans\example.dcm"
    form = ParamFormWidget(dicom_rt_plan.param_specs(), params)
    assert form.read_params()["dicom_path"] == r"C:\plans\example.dcm"


def test_resolve_plan_synthesis_save_dir_prefers_last_saved(tmp_path: Path) -> None:
    from scan_kit.workflows.plan_synthesis.paths import resolve_plan_synthesis_save_dir

    last = tmp_path / "saved-plans"
    last.mkdir()
    assert resolve_plan_synthesis_save_dir(str(last)) == last


def test_app_settings_persists_last_plan_synthesis_save_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scan_kit.common.app_settings import AppSettings

    monkeypatch.setattr("scan_kit.common.app_settings._SETTINGS_DIR", tmp_path)
    settings = AppSettings(last_plan_synthesis_save_dir=r"C:\plans\exports")
    settings.save()
    loaded = AppSettings.load()
    assert loaded.last_plan_synthesis_save_dir == r"C:\plans\exports"


@pytest.fixture
def iba_pld_plan():
    template = get_template("iba_pld_plan")
    assert template is not None
    return template


_MINIMAL_PLD = """\
Beam,Patient ID,Patient Name,Patient Initial,Patient Firstname,TestPlan,Beam1,10,10,1
Layer,Spot1,100,10,4
Element,0,0,0,0.0
Element,0,0,5,0.0
Element,10,0,0,0.0
Element,10,0,5,0.0
"""


def test_iba_pld_plan_generate_from_minimal_fixture(iba_pld_plan, tmp_path: Path) -> None:
    pld_path = tmp_path / "minimal.pld"
    pld_path.write_text(_MINIMAL_PLD, encoding="utf-8")

    params = {
        "pld_path": str(pld_path),
        "beam_size_mm": 4.0,
        "spot_order": "plan_order",
        "fast_axis": FAST_AXIS_X,
    }
    assert iba_pld_plan.validate(params) == []
    df = iba_pld_plan.generate(params)
    assert len(df) == 2
    assert df["ENERGY"].tolist() == [100.0, 100.0]
    assert df["X_POSITION"].tolist() == [0.0, 10.0]
    assert df["CHARGE_REQ"].tolist() == [5.0, 5.0]
    assert (df["BEAM_SIZE"] == 4.0).all()


def test_iba_pld_plan_minimize_travel_reorders_layer(
    iba_pld_plan,
    tmp_path: Path,
) -> None:
    from scan_kit.workflows.plan_synthesis.spot_order import SPOT_ORDER_MINIMIZE_TRAVEL

    pld_path = tmp_path / "scrambled.pld"
    pld_path.write_text(
        "\n".join(
            [
                "Beam,Patient ID,Patient Name,Patient Initial,Patient Firstname,TestPlan,Beam1,20,20,1",
                "Layer,Spot1,100,20,8",
                "Element,20,0,0,0.0",
                "Element,20,0,10,0.0",
                "Element,0,0,0,0.0",
                "Element,0,0,10,0.0",
                "Element,20,10,0,0.0",
                "Element,20,10,10,0.0",
                "Element,0,10,0,0.0",
                "Element,0,10,10,0.0",
            ]
        ),
        encoding="utf-8",
    )

    params = {
        "pld_path": str(pld_path),
        "beam_size_mm": 3.61,
        "spot_order": SPOT_ORDER_MINIMIZE_TRAVEL,
        "fast_axis": FAST_AXIS_X,
    }
    df = iba_pld_plan.generate(params)
    assert list(zip(df["X_POSITION"], df["Y_POSITION"])) == [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (0.0, 10.0),
    ]


_OOC_PLD = Path(r"c:\Users\MattNichols\Downloads\OOC_Right_scaledMU_new.pld")


@pytest.mark.skipif(not _OOC_PLD.is_file(), reason="OOC PLD fixture not available")
def test_iba_pld_plan_generate_from_ooc_example(iba_pld_plan) -> None:
    params = {
        "pld_path": str(_OOC_PLD),
        "beam_size_mm": 3.61,
        "spot_order": "plan_order",
        "fast_axis": FAST_AXIS_X,
    }
    assert iba_pld_plan.validate(params) == []
    df = iba_pld_plan.generate(params)
    assert len(df) == 118
    assert df["ENERGY"].nunique() == 1
    assert df["ENERGY"].iloc[0] == pytest.approx(228.0)
    assert df["CHARGE_REQ"].sum() == pytest.approx(11490.25, rel=1e-6)


def test_suggest_input_map_filename_iba_pld_plan(iba_pld_plan, tmp_path: Path) -> None:
    from scan_kit.workflows.plan_synthesis.export_filename import (
        suggest_input_map_filename,
    )

    pld_path = tmp_path / "OOC_Right.pld"
    pld_path.write_text(_MINIMAL_PLD.replace("TestPlan", "OOC_Right"), encoding="utf-8")
    params = {
        "pld_path": str(pld_path),
        "beam_size_mm": 3.61,
        "spot_order": "plan_order",
        "fast_axis": FAST_AXIS_X,
    }
    name = suggest_input_map_filename(iba_pld_plan, params)
    assert name == "IbaPld_OOC_Right.csv"


def test_plan_synthesis_default_save_path_uses_last_dir(
    zero_field,
    tmp_path: Path,
) -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    from scan_kit.common.app_settings import AppSettings
    from scan_kit.workflows.plan_synthesis_panel import PlanSynthesisPanel

    app = QApplication.instance() or QApplication(sys.argv)
    save_dir = tmp_path / "plans"
    save_dir.mkdir()
    settings = AppSettings(last_plan_synthesis_save_dir=str(save_dir))
    panel = PlanSynthesisPanel(app_settings=settings)
    panel._current = zero_field
    panel._last_generate_params = _fixed_weight_params(
        selected_energies=[250.0],
        spots_per_layer=1,
    )

    default_path = Path(panel._default_save_path())
    assert default_path.parent == save_dir
    assert default_path.name.endswith(".csv")
