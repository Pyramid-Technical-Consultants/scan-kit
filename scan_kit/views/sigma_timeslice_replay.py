"""IC sigma timeslice replay: media-player style interactive spot-size viewer."""

from __future__ import annotations

import numpy as np

from ..common import (
    C_BEAM_CURRENT,
    C_IC1_CURRENT,
    C_LAYER_ID,
)
from ..common.session_source import load_session_timeslice_device_units
from ..common.timeslice_sigma import (
    frame_timeslice_sigma_arrays,
    resolve_timeslice_sigma_source,
)
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

_SIGMA_KEYS = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")


def _load_session_timeline(session_id: str, base_dir: str, *, bg_subtract: bool = False) -> dict | None:
    """Load and concatenate all timeslice sigma traces into a unified timeline."""
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
    source = resolve_timeslice_sigma_source(df0.columns)
    if not ts_layer or source is None:
        return None

    ts_ic1 = resolve_col(df0.columns, C_IC1_CURRENT)
    has_ic1_current = ts_ic1 is not None

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    has_beam = ts_beam is not None

    digital_cols = detect_digital_columns(df0.columns)
    digital_parts: dict[str, list[np.ndarray]] = {col: [] for col, _ in digital_cols}

    sigma_parts: dict[str, list[np.ndarray]] = {k: [] for k in _SIGMA_KEYS}
    beam_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    edge_indices: dict[str, list[int]] = {"ic1_x": []}
    offset = 0

    for df in frames:
        n = len(df)
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id, 0.0)

        frame_sigmas = frame_timeslice_sigma_arrays(df, source)
        if frame_sigmas is None:
            nan = np.full(n, np.nan)
            for key in _SIGMA_KEYS:
                sigma_parts[key].append(nan)
        else:
            ic1_x, ic1_y, ic2_x, ic2_y = frame_sigmas
            sigma_parts["ic1_x"].append(ic1_x)
            sigma_parts["ic1_y"].append(ic1_y)
            sigma_parts["ic2_x"].append(ic2_x)
            sigma_parts["ic2_y"].append(ic2_y)

        if has_ic1_current:
            ic1_vals = df[ts_ic1].values
            edges = detect_beam_off_edges(ic1_vals)
            edge_indices["ic1_x"].extend((edges + offset).tolist())

        if has_beam:
            beam_parts.append(df[ts_beam].values.astype(float))
        for col, _ in digital_cols:
            if col in df.columns:
                digital_parts[col].append(df[col].values.astype(float))
            else:
                digital_parts[col].append(np.zeros(n))
        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    result: dict = {
        **{k: np.concatenate(parts) for k, parts in sigma_parts.items()},
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_beam": has_beam,
        "energy": np.concatenate(energy_parts),
        "beam_off_edges": {k: np.asarray(v, dtype=int) for k, v in edge_indices.items()},
        "digital": build_digital_signals(digital_parts, digital_cols),
    }
    if has_beam:
        result["beam"] = np.concatenate(beam_parts)
    return result


def _replay_config(session_data: dict[str, dict]) -> TimesliceReplayConfig:
    return TimesliceReplayConfig(
        title="Sigma Timeslice Replay",
        no_data_message="No valid timeslice sigma data found for any session",
        traces=(
            TraceSpec("ic1_x", "IC1 σx (mm)", "#1f77b4", beam_off_edges=True),
            TraceSpec("ic1_y", "IC1 σy (mm)", "#aec7e8"),
            TraceSpec("ic2_x", "IC2 σx (mm)", "#d62728"),
            TraceSpec("ic2_y", "IC2 σy (mm)", "#ff9896"),
        ),
        timeline_key="ic1_x",
        timeline_ylabel="IC1 σx (mm)",
        figsize=(22, 12),
        scatter=ScatterSpec(
            mode="per_trace",
            per_trace_xy={
                "ic1_x": ("ic1_x", "ic1_y"),
                "ic1_y": ("ic1_x", "ic1_y"),
                "ic2_x": ("ic2_x", "ic2_y"),
                "ic2_y": ("ic2_x", "ic2_y"),
            },
            per_trace_title_suffix=" σ (mm)",
        ),
    )


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the IC sigma timeslice replay viewer."""
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
