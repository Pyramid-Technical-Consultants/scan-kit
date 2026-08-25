"""Build per-sample timeslice tables tagged with energy and beam_on."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import detect_beam_on_mask, subtract_background_frames
from .session_source import SessionSource, load_session_timeslice_device_units
from .timeslice_energy import load_energy_lookups, resolve_frame_energy, resolve_layer_col


def load_session_timeslice_frames(
    session_id: str,
    base_dir: str,
    *,
    usecols: list[str] | None = None,
    bg_subtract: bool = False,
) -> tuple[SessionSource, list[pd.DataFrame], dict | None, dict[int, float], str] | None:
    """Open a session and return frames plus energy lookups."""
    opened = load_energy_lookups(session_id, base_dir)
    if opened is None:
        return None
    src, energy_by_layer, energy_by_idx = opened

    frames = load_session_timeslice_device_units(src, usecols=usecols)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    layer_col = resolve_layer_col(frames[0].columns)
    if layer_col is None:
        return None

    return src, frames, energy_by_layer, energy_by_idx, layer_col


def load_energy_tagged_table(
    session_id: str,
    base_dir: str,
    *,
    usecols: list[str],
    bg_subtract: bool = False,
    prepare: Callable[[SessionSource, list[pd.DataFrame]], Any],
    extract: Callable[[pd.DataFrame, Any], tuple[np.ndarray, ...] | None],
    keys: tuple[str, ...],
) -> dict | None:
    """Build a per-sample dict tagged with energy and beam_on."""
    opened = load_session_timeslice_frames(
        session_id, base_dir, usecols=usecols, bg_subtract=bg_subtract,
    )
    if opened is None:
        return None
    _src, frames, energy_by_layer, energy_by_idx, layer_col = opened

    context = prepare(_src, frames)
    if context is None:
        return None

    energy_parts: list[np.ndarray] = []
    beam_on_parts: list[np.ndarray] = []
    value_parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    for frame_i, df in enumerate(frames):
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
        arrays = extract(df, context)
        if arrays is None:
            continue
        if len(arrays) != len(keys):
            raise ValueError("extract() must return one array per output key")
        n = len(arrays[0])
        if n == 0:
            continue
        energy_parts.append(np.full(n, energy, dtype=float))
        beam_on_parts.append(beam_on.astype(bool))
        for key, arr in zip(keys, arrays):
            value_parts[key].append(arr)

    if not energy_parts:
        return None

    result: dict = {
        "session_id": session_id,
        "energy": np.concatenate(energy_parts),
        "beam_on": np.concatenate(beam_on_parts),
    }
    for key in keys:
        result[key] = np.concatenate(value_parts[key])
    return result
