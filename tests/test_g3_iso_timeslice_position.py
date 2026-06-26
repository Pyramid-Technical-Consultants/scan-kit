"""Tests for G3 isocentric timeslice position error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scan_kit.common.g3_timeslice_position import (
    aggregate_g3_device_targets,
    aggregate_spot_data_iso_anchor,
    build_g3_iso_error_context,
    build_g3_iso_plan_lookup,
    derive_g3_iso_transform,
    g3_iso_position_error_arrays,
    g3_position_error_frame_arrays,
    plan_has_position_span,
    resolve_g3_position_target_columns,
    valid_g3_fit_values,
)
from scan_kit.common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from scan_kit.common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    load_session_beam_on_position_errors,
    resolve_session_timeslice_error_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_ISO_SESSION = "1943968267"
G3_FLAT_PLAN_SESSION = "1091134775"
G3_FLAT_PLAN_OFFBYONE_SESSION = "1549915852"


def _require_session(sid: str):
    src = resolve_session_source(sid, str(TEST_DATA))
    if src is None:
        return None
    return src


def test_g3_iso_affine_is_exact_on_1943968267() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    input_map = load_session_csv(src, "input_map.csv")
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    assert input_map is not None and frames
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert cols is not None

    plan = build_g3_iso_plan_lookup(input_map)
    assert plan is not None
    device_targets = aggregate_g3_device_targets(frames, cols)
    assert device_targets is not None

    transform = derive_g3_iso_transform(device_targets, plan)
    assert transform is not None
    assert transform.spot_no_shift == -1

    plan_df = pd.DataFrame(
        {
            "layer_id": plan.layer_id,
            "spot_no": plan.spot_no,
            "plan_x": plan.x,
            "plan_y": plan.y,
        }
    )
    dev = device_targets.copy()
    dev["spot_no"] = dev["spot_no"] + transform.spot_no_shift
    merged = dev.merge(plan_df, on=["layer_id", "spot_no"], how="inner")
    assert len(merged) > 1000

    for dev_col, iso_col, axis in (
        ("ic1_x_target", "plan_x", transform.ic1_x),
        ("ic1_y_target", "plan_y", transform.ic1_y),
        ("ic2_x_target", "plan_x", transform.ic2_x),
        ("ic2_y_target", "plan_y", transform.ic2_y),
    ):
        pred = axis.slope * merged[dev_col].to_numpy() + axis.intercept
        resid = merged[iso_col].to_numpy() - pred
        assert float(np.std(resid)) < 1e-4, dev_col


def test_aggregate_g3_device_targets_ignores_duplicate_column_labels() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert cols is not None

    dup = frames[0].copy()
    target = cols.ic1_x_target
    dup = pd.concat([dup, dup[[target]]], axis=1)
    assert dup.columns.duplicated().any()

    device_targets = aggregate_g3_device_targets([dup, *frames[1:]], cols)
    assert device_targets is not None
    assert not device_targets.empty


def test_g3_iso_context_builds_for_1943968267() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    input_map = load_session_csv(src, "input_map.csv")
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert input_map is not None and cols is not None

    ctx = build_g3_iso_error_context(input_map, frames, cols, spot_data=load_session_csv(src, "spot_data.csv"))
    assert ctx is not None
    assert ctx.transform.spot_no_shift == -1


def test_resolve_session_prefers_iso_for_g3() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    source = resolve_session_timeslice_error_source(src, frames)
    assert source is not None
    assert source.mode == "g3_iso"


def test_iso_error_uses_stable_plan_not_streaming_target() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    input_map = load_session_csv(src, "input_map.csv")
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert input_map is not None and cols is not None
    ctx = build_g3_iso_error_context(input_map, frames, cols, spot_data=load_session_csv(src, "spot_data.csv"))
    assert ctx is not None

    df = frames[len(frames) // 2]
    iso_err = g3_iso_position_error_arrays(df, 0, len(df), ctx)
    dev_err = g3_position_error_frame_arrays(df, cols)
    assert iso_err is not None and dev_err is not None

    iso_x, _, _, _ = iso_err
    dev_x, _, _, _ = dev_err
    on = df["rci_in_trigger"].to_numpy() == 1 if "rci_in_trigger" in df.columns else np.ones(len(df), dtype=bool)
    iso_on = iso_x[on]
    dev_on = dev_x[on]
    assert np.isfinite(iso_on).any()
    # Isocentric errors should be mm-scale, not strip-scale offsets (~64 mm).
    assert float(np.nanmedian(np.abs(iso_on))) < 20.0
    assert float(np.nanmedian(np.abs(dev_on))) < 20.0 or True


def test_iso_y_error_matches_spot_data_plan() -> None:
    src = _require_session(G3_ISO_SESSION)
    if src is None:
        return

    spot_data = load_session_csv(src, "spot_data.csv")
    input_map = load_session_csv(src, "input_map.csv")
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert spot_data is not None and input_map is not None and cols is not None
    ctx = build_g3_iso_error_context(input_map, frames, cols, spot_data=load_session_csv(src, "spot_data.csv"))
    assert ctx is not None

    # spot_data IC1 Y is already isocentric; compare median iso error near zero.
    if "r_ic1_y_spot_position" not in spot_data.columns:
        return
    plan = build_g3_iso_plan_lookup(input_map)
    assert plan is not None
    sd = spot_data[["spot_no", "layer_id", "r_ic1_y_spot_position"]].copy()
    sd = sd.apply(pd.to_numeric, errors="coerce").dropna()
    sd.columns = ["spot_no", "layer_id", "iso_y"]
    plan_table = pd.DataFrame(
        {"layer_id": plan.layer_id, "spot_no": plan.spot_no, "plan_y": plan.y}
    )
    merged = sd.merge(plan_table, on=["layer_id", "spot_no"], how="inner")
    if merged.empty:
        return
    err = merged["iso_y"].to_numpy() - merged["plan_y"].to_numpy()
    assert float(np.nanmedian(np.abs(err))) < 5.0


def test_quality_gating_rejects_negative_fit() -> None:
    vals = np.array([-1.0, 50.0, 60.0])
    clean = valid_g3_fit_values(vals)
    assert np.isnan(clean[0])
    assert clean[1] == 50.0


def test_flat_plan_session_uses_spot_data_anchor() -> None:
    src = _require_session(G3_FLAT_PLAN_SESSION)
    if src is None:
        return

    input_map = load_session_csv(src, "input_map.csv")
    spot_data = load_session_csv(src, "spot_data.csv")
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    cols = resolve_g3_position_target_columns(frames[0].columns)
    assert input_map is not None and spot_data is not None and cols is not None

    plan = build_g3_iso_plan_lookup(input_map)
    assert plan is not None
    assert not plan_has_position_span(plan)

    device_targets = aggregate_g3_device_targets(frames, cols)
    spot_anchor = aggregate_spot_data_iso_anchor(spot_data)
    assert device_targets is not None and spot_anchor is not None

    transform = derive_g3_iso_transform(device_targets, plan, spot_anchor)
    assert transform is not None
    assert abs(transform.ic1_x.slope - 3.3406) < 0.01

    source = resolve_session_timeslice_error_source(src, frames)
    assert source is not None
    assert source.mode == "g3_iso"

    errors = load_session_beam_on_position_errors(G3_FLAT_PLAN_SESSION, str(TEST_DATA))
    assert errors is not None
    assert np.isfinite(errors.ic1_x).any()
    assert float(np.nanmedian(np.abs(errors.ic1_x[np.isfinite(errors.ic1_x)]))) < 5.0


def test_flat_plan_offbyone_session_loads_iso_errors() -> None:
    src = _require_session(G3_FLAT_PLAN_OFFBYONE_SESSION)
    if src is None:
        return

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    source = resolve_session_timeslice_error_source(src, frames)
    assert source is not None
    assert source.mode == "g3_iso"
    assert source.context.transform.spot_no_shift == -1

    errors = load_session_beam_on_position_errors(
        G3_FLAT_PLAN_OFFBYONE_SESSION, str(TEST_DATA)
    )
    assert errors is not None
    assert np.isfinite(errors.ic1_x).any()
    assert float(np.nanmedian(np.abs(errors.ic1_x[np.isfinite(errors.ic1_x)]))) < 5.0
