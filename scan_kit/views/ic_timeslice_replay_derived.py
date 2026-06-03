"""IC timeslice replay using IC current derived from the scan-total dose.

Mirrors :mod:`ic_timeslice_replay` but derives per-slice IC current by
differentiating scan-total dose columns instead of reading IC current directly.
"""

from __future__ import annotations

import numpy as np

from ..common import (
    C_BEAM_CURRENT,
    C_IC1_SCAN_TOTAL_DOSE,
    C_IC2_SCAN_TOTAL_DOSE,
    C_IC3_SCAN_TOTAL_DOSE,
    C_IC1_X_POS_RAW,
    C_IC1_Y_POS_RAW,
    C_IC2_X_POS_RAW,
    C_IC2_Y_POS_RAW,
    C_LAYER_ID,
    resolve_concept_column,
)
from ..common.session_source import load_session_timeslice_device_units
from ..common.schema import POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW
from ..common import transform
from .beam_off_rampdown import detect_beam_off_edges
from .timeslice_replay_common import (
    MS_PER_SLICE,
    build_digital_signals,
    detect_digital_columns,
    load_energy_by_layer,
    resolve_col,
)
from .timeslice_replay_ui import (
    ScatterSpec,
    TimesliceReplayConfig,
    TraceSpec,
    launch_timeslice_replay,
)


def _derive_current_from_dose(dose: np.ndarray) -> np.ndarray:
    """Per-slice derivative of a monotonically-accumulating dose column."""
    if len(dose) == 0:
        return np.empty(0, dtype=float)
    arr = np.asarray(dose, dtype=float)
    deriv = np.empty_like(arr)
    deriv[0] = 0.0
    if len(arr) > 1:
        deriv[1:] = (arr[1:] - arr[:-1]) / MS_PER_SLICE
    deriv[~np.isfinite(deriv)] = 0.0
    deriv[deriv < 0] = 0.0
    return deriv


def _load_session_timeline(session_id: str, base_dir: str) -> dict | None:
    """Load timeline with IC current derived from scan-total dose."""
    loaded = load_energy_by_layer(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer = loaded

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    ts_dose1 = resolve_col(df0.columns, C_IC1_SCAN_TOTAL_DOSE)
    ts_dose2 = resolve_col(df0.columns, C_IC2_SCAN_TOTAL_DOSE)
    if not all([ts_layer, ts_dose1, ts_dose2]):
        return None

    ts_dose3 = resolve_col(df0.columns, C_IC3_SCAN_TOTAL_DOSE)
    has_ic3 = ts_dose3 is not None

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    has_beam = ts_beam is not None

    pos_cols: dict[str, str] = {}
    for pos_key in (POSITION_KEY_G3_RAW, POSITION_KEY_G2_RAW):
        for concept, label in [
            (C_IC1_X_POS_RAW, "ic1_x"), (C_IC1_Y_POS_RAW, "ic1_y"),
            (C_IC2_X_POS_RAW, "ic2_x"), (C_IC2_Y_POS_RAW, "ic2_y"),
        ]:
            resolved = resolve_concept_column(df0.columns, concept, position_key=pos_key)
            if resolved and label not in pos_cols:
                pos_cols[label] = resolved
        if len(pos_cols) == 4:
            break
    has_positions = len(pos_cols) == 4

    digital_cols = detect_digital_columns(df0.columns)
    digital_parts: dict[str, list[np.ndarray]] = {col: [] for col, _ in digital_cols}

    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    ic3_parts: list[np.ndarray] = []
    beam_parts: list[np.ndarray] = []
    pos_parts: dict[str, list[np.ndarray]] = {k: [] for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y")}
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    edge_indices: dict[str, list[int]] = {"ic1": [], "ic2": [], "ic3": []}
    offset = 0

    for df in frames:
        n = len(df)
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id, 0.0)

        ic1_vals = _derive_current_from_dose(df[ts_dose1].values.astype(float))
        ic2_vals = _derive_current_from_dose(df[ts_dose2].values.astype(float))
        ic1_parts.append(ic1_vals)
        ic2_parts.append(ic2_vals)

        for key, vals in [("ic1", ic1_vals), ("ic2", ic2_vals)]:
            edges = detect_beam_off_edges(vals)
            edge_indices[key].extend((edges + offset).tolist())

        if has_ic3:
            ic3_vals = _derive_current_from_dose(df[ts_dose3].values.astype(float))
            ic3_parts.append(ic3_vals)
            edges = detect_beam_off_edges(ic3_vals)
            edge_indices["ic3"].extend((edges + offset).tolist())

        if has_beam:
            beam_parts.append(df[ts_beam].values.astype(float))
        for col, _ in digital_cols:
            if col in df.columns:
                digital_parts[col].append(df[col].values.astype(float))
            else:
                digital_parts[col].append(np.zeros(n))
        if has_positions:
            for label, col in pos_cols.items():
                pos_parts[label].append(df[col].values.astype(float))
        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    result: dict = {
        "ic1": np.concatenate(ic1_parts),
        "ic2": np.concatenate(ic2_parts),
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_ic3": has_ic3,
        "has_beam": has_beam,
        "has_positions": has_positions,
        "energy": np.concatenate(energy_parts),
        "beam_off_edges": {k: np.asarray(v, dtype=int) for k, v in edge_indices.items()},
        "digital": build_digital_signals(digital_parts, digital_cols),
    }
    if has_ic3:
        result["ic3"] = np.concatenate(ic3_parts)
    if has_beam:
        result["beam"] = np.concatenate(beam_parts)
    if has_positions:
        result["ic1_x"] = transform.remap(np.concatenate(pos_parts["ic1_x"]), *transform.IC1_X_MAP)
        result["ic1_y"] = transform.remap(np.concatenate(pos_parts["ic1_y"]), *transform.IC1_Y_MAP)
        result["ic2_x"] = transform.remap(np.concatenate(pos_parts["ic2_x"]), *transform.IC2_X_MAP)
        result["ic2_y"] = transform.remap(np.concatenate(pos_parts["ic2_y"]), *transform.IC2_Y_MAP)
        _pos_limit = transform.IC_MM_MAX
        for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y"):
            arr = result[k]
            arr[np.abs(arr) > _pos_limit] = np.nan
    return result


def _replay_config(session_data: dict[str, dict], multi: bool) -> TimesliceReplayConfig:
    show_ic3 = any(d.get("has_ic3", False) for d in session_data.values())
    show_pos = any(d.get("has_positions", False) for d in session_data.values())

    traces: list[TraceSpec] = [
        TraceSpec("ic1", "IC1 dDose/dt", "#1f77b4", linewidth=0.6, beam_off_edges=True),
        TraceSpec("ic2", "IC2 dDose/dt", "#d62728", linewidth=0.6, beam_off_edges=True),
    ]
    if show_ic3:
        traces.append(
            TraceSpec("ic3", "IC3 dDose/dt", "#2ca02c", linewidth=0.6, beam_off_edges=True),
        )

    scatter = ScatterSpec(mode="none")
    if show_pos:
        scatter = ScatterSpec(
            mode="per_trace",
            per_trace_xy={
                "ic1": ("ic1_x", "ic1_y"),
                "ic2": ("ic2_x", "ic2_y"),
            },
        )

    return TimesliceReplayConfig(
        title="IC Timeslice Replay — Current Derived from Scan-Total Dose",
        no_data_message="No valid timeslice data found for any session",
        traces=tuple(traces),
        timeline_key="ic1",
        timeline_ylabel="dDose/dt",
        figsize=(22 if show_pos else 18, 10),
        scatter=scatter,
        peer_overlay=not multi,
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC timeslice replay viewer using dose-derived IC current."""
    del settings

    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_session_timeline(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    multi = len(session_data) > 1
    launch_timeslice_replay(_replay_config(session_data, multi), session_data, base_dir)
