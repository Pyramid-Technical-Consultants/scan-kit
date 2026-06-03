"""Magnetic-field timeslice replay: interactive Bx/By viewer (gauss)."""

from __future__ import annotations

import numpy as np

from ..common import (
    C_BEAM_CURRENT,
    C_LAYER_ID,
    C_MAG_FIELD_X,
    C_MAG_FIELD_Y,
)
from ..common.session_source import load_session_timeslice_device_units
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

_FIELD_TRACES = (
    TraceSpec("bx", "Bx (G)", "#1f77b4"),
    TraceSpec("by", "By (G)", "#d62728"),
)


def _load_session_timeline(session_id: str, base_dir: str) -> dict | None:
    """Load and concatenate scan-magnet field probes into a unified timeline."""
    loaded = load_energy_by_layer(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer = loaded

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    ts_bx = resolve_col(df0.columns, C_MAG_FIELD_X)
    ts_by = resolve_col(df0.columns, C_MAG_FIELD_Y)
    if not all([ts_layer, ts_bx, ts_by]):
        return None

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    has_beam = ts_beam is not None
    digital_cols = detect_digital_columns(df0.columns)
    digital_parts: dict[str, list[np.ndarray]] = {col: [] for col, _ in digital_cols}

    bx_parts: list[np.ndarray] = []
    by_parts: list[np.ndarray] = []
    beam_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    offset = 0

    for df in frames:
        n = len(df)
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id, 0.0)

        bx_parts.append(df[ts_bx].values.astype(float))
        by_parts.append(df[ts_by].values.astype(float))

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

    bx = np.concatenate(bx_parts)
    by = np.concatenate(by_parts)

    result: dict = {
        "bx": bx,
        "by": by,
        "b_mag": np.hypot(bx, by),
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_beam": has_beam,
        "energy": np.concatenate(energy_parts),
        "digital": build_digital_signals(digital_parts, digital_cols),
    }
    if has_beam:
        result["beam"] = np.concatenate(beam_parts)
    return result


_FIELD_CONFIG = TimesliceReplayConfig(
    title="Magnetic Field Timeslice Replay",
    no_data_message="No valid magnetic-field timeslice data found for any session",
    traces=_FIELD_TRACES,
    timeline_key="b_mag",
    timeline_ylabel="|B| (G)",
    figsize=(22, 9),
    scatter=ScatterSpec(
        mode="single",
        x_key="bx",
        y_key="by",
        title="B vector (G)",
        xlabel="Bx (G)",
        ylabel="By (G)",
    ),
)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Launch the magnetic-field timeslice replay viewer."""
    del settings

    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_session_timeline(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    launch_timeslice_replay(_FIELD_CONFIG, session_data, base_dir)
