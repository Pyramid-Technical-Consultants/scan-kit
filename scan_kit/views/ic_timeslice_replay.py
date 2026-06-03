"""IC timeslice replay: media-player style interactive current viewer."""

from __future__ import annotations

import numpy as np

from ..common import (
    C_BEAM_CURRENT,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
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


def _load_session_timeline(session_id: str, base_dir: str, *, bg_subtract: bool = False) -> dict | None:
    """Load and concatenate all timeslice frames into a unified timeline."""
    loaded = load_energy_by_layer(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer = loaded

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames
        subtract_background_frames(frames)

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    ts_ic1 = resolve_col(df0.columns, C_IC1_CURRENT)
    ts_ic2 = resolve_col(df0.columns, C_IC2_CURRENT)
    if not all([ts_layer, ts_ic1, ts_ic2]):
        return None

    ts_ic3a = resolve_col(df0.columns, C_IC3_CURRENT_A)
    ts_ic3b = resolve_col(df0.columns, C_IC3_CURRENT_B)
    ts_ic3c = resolve_col(df0.columns, C_IC3_CURRENT_C)
    ts_ic3d = resolve_col(df0.columns, C_IC3_CURRENT_D)
    has_ic3 = all([ts_ic3a, ts_ic3b, ts_ic3c, ts_ic3d])

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

        ic1_vals = df[ts_ic1].values
        ic2_vals = df[ts_ic2].values
        ic1_parts.append(ic1_vals)
        ic2_parts.append(ic2_vals)

        for key, vals in [("ic1", ic1_vals), ("ic2", ic2_vals)]:
            edges = detect_beam_off_edges(vals)
            edge_indices[key].extend((edges + offset).tolist())

        if has_ic3:
            ic3_vals = (
                df[ts_ic3a].values
                + df[ts_ic3b].values
                + df[ts_ic3c].values
                + df[ts_ic3d].values
            )
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


def _replay_config(session_data: dict[str, dict]) -> TimesliceReplayConfig:
    show_ic3 = any(d.get("has_ic3", False) for d in session_data.values())
    show_pos = any(d.get("has_positions", False) for d in session_data.values())

    traces: list[TraceSpec] = [
        TraceSpec("ic1", "IC1", "#1f77b4", beam_off_edges=True),
        TraceSpec("ic2", "IC2", "#d62728", beam_off_edges=True),
    ]
    if show_ic3:
        traces.append(TraceSpec("ic3", "IC3 (A+B+C+D)", "#2ca02c", beam_off_edges=True))

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
        title="IC Timeslice Replay",
        no_data_message="No valid timeslice data found for any session",
        traces=tuple(traces),
        timeline_key="ic1",
        timeline_ylabel="IC1",
        figsize=(22 if show_pos else 18, 10),
        scatter=scatter,
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC timeslice replay viewer."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        bg = settings.bg_subtract if settings else False
        data = _load_session_timeline(sid, base_dir, bg_subtract=bg)
        if data is not None:
            session_data[sid] = data

    launch_timeslice_replay(_replay_config(session_data), session_data, base_dir)
