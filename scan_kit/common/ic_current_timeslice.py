"""Per-sample IC beam current vs energy from timeslice data."""

from __future__ import annotations

import numpy as np

from . import detect_beam_on_mask
from .timeslice_energy import load_energy_lookups, resolve_frame_energy, resolve_layer_col
from .timeslice_ic_current import resolve_ic_current_columns, sum_ic3_current
from .timeslice_table import load_session_timeslice_frames


def load_session_ic_current_timeslice(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    """Load per-timeslice-sample IC currents, energy, and beam-on flags."""
    opened = load_session_timeslice_frames(session_id, base_dir, bg_subtract=bg_subtract)
    if opened is None:
        return None
    _src, frames, energy_by_layer, energy_by_idx, layer_col = opened

    cols = resolve_ic_current_columns(frames[0].columns)
    if cols is None:
        return None
    has_ic3 = bool(cols.ic3_parts)

    energy_parts: list[np.ndarray] = []
    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    ic3_parts: list[np.ndarray] = []
    beam_on_parts: list[np.ndarray] = []

    for frame_i, df in enumerate(frames):
        if df.empty:
            continue
        energy = resolve_frame_energy(
            df,
            frame_i,
            energy_by_layer=energy_by_layer,
            energy_by_idx=energy_by_idx,
            layer_col=layer_col,
        )
        if energy is None:
            continue
        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        n = len(df)
        energy_parts.append(np.full(n, energy, dtype=float))
        ic1_parts.append(df[cols.ic1].to_numpy(dtype=float))
        ic2_parts.append(df[cols.ic2].to_numpy(dtype=float))
        if has_ic3:
            ic3_parts.append(sum_ic3_current(df, cols.ic3_parts))
        beam_on_parts.append(beam_on.astype(bool))

    if not energy_parts:
        return None

    result: dict = {
        "session_id": session_id,
        "energy": np.concatenate(energy_parts),
        "ic1_current": np.concatenate(ic1_parts),
        "ic2_current": np.concatenate(ic2_parts),
        "beam_on": np.concatenate(beam_on_parts),
        "has_ic3": has_ic3,
    }
    if has_ic3:
        result["ic3_current"] = np.concatenate(ic3_parts)
    return result
