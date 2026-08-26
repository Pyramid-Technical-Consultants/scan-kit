"""Layer-aggregated beam-on IC current ratios vs energy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    C_ENERGY,
    C_LAYER_ID,
    resolve_concept_column,
    subtract_background_frames,
)
from .processing import _detect_beam_off_mask
from .session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)
from .timeslice_energy import build_energy_lookups, resolve_frame_energy
from .timeslice_ic_current import resolve_ic_current_columns, sum_ic3_current

MIN_BEAM_SAMPLES = 10
DISPLAY_P95_FRAC = 0.75


def sym_pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Symmetric relative difference in percent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / ((a + b) / 2.0) * 100.0


def load_session_current_ratios(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    """Return per-energy symmetric IC current ratios from beam-on plateau means."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    if col_energy is None:
        return None

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = next((frame for frame in frames if not frame.empty), frames[0])
    cols = resolve_ic_current_columns(df0.columns)
    if cols is None:
        return None
    has_ic3 = bool(cols.ic3_parts)

    lookups = build_energy_lookups(input_map)
    if lookups is None:
        return None
    energy_by_layer, energy_by_idx = lookups
    layer_col = resolve_concept_column(df0.columns, C_LAYER_ID) or ""

    ic_keys = ["ic1", "ic2"]
    if has_ic3:
        ic_keys.append("ic3")

    energies_out: list[float] = []
    ic_plateau: dict[str, list[float]] = {ic: [] for ic in ic_keys}

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
            energy = energy_by_idx.get(frame_i, 0.0)
        energies_out.append(float(energy))

        beam_off = _detect_beam_off_mask(df)
        beam_on = ~beam_off if beam_off is not None else None

        ic_signals = {
            "ic1": df[cols.ic1].to_numpy(dtype=np.float64, na_value=0.0),
            "ic2": df[cols.ic2].to_numpy(dtype=np.float64, na_value=0.0),
        }
        if has_ic3:
            ic_signals["ic3"] = sum_ic3_current(df, cols.ic3_parts)

        for ic in ic_keys:
            sig = ic_signals[ic]
            if beam_on is None or beam_on.sum() < MIN_BEAM_SAMPLES:
                ic_plateau[ic].append(np.nan)
                continue

            sig = np.array(sig, dtype=np.float64, copy=True)
            sig[~np.isfinite(sig)] = 0.0
            on_samples = sig[beam_on]
            p95 = float(np.nanpercentile(on_samples, 95))
            floor = max(5.0, p95 * DISPLAY_P95_FRAC)
            beam_cluster = on_samples[on_samples >= floor]
            if len(beam_cluster) >= 3:
                ic_plateau[ic].append(float(np.mean(beam_cluster)))
            else:
                ic_plateau[ic].append(np.nan)

    if not energies_out:
        return None

    ic1 = np.asarray(ic_plateau["ic1"], dtype=float)
    ic2 = np.asarray(ic_plateau["ic2"], dtype=float)
    result: dict = {
        "session_id": session_id,
        "energy": np.asarray(energies_out, dtype=float),
        "ic21_ratio": sym_pct(ic2, ic1),
    }
    if "ic3" in ic_keys:
        ic3 = np.asarray(ic_plateau["ic3"], dtype=float)
        result["ic31_ratio"] = sym_pct(ic3, ic1)
        result["ic32_ratio"] = sym_pct(ic3, ic2)
    return result
