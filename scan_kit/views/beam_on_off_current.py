"""Beam-on vs beam-off current box plots (IC1, IC2, IC3) from timeslice data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    C_LAYER_ID,
    plot_boxplots_for_column,
    finish_view,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    view_grid,
)
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

import logging

_log = logging.getLogger(__name__)

ON_FRAC = 0.10
OFF_FRAC = 0.02

_TIMESLICE_COLS = [
    C_LAYER_ID,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
]


def _classify_timeslices(signal: np.ndarray):
    """Return boolean masks (beam_on, beam_off) for timeslice classification.

    Two thresholds derived from the signal's dynamic range:
      beam-on  — above  bg + ON_FRAC  * (peak - bg)
      beam-off — below  bg + OFF_FRAC * (peak - bg)
    Samples between are transition (excluded from both).
    """
    bg = np.nanpercentile(signal, 25)
    pk = np.nanpercentile(signal, 99)
    dyn = pk - bg
    if pk == 0 or abs(dyn / pk) < 0.05:
        return None, None
    on_mask = signal > (bg + ON_FRAC * dyn)
    off_mask = signal < (bg + OFF_FRAC * dyn)
    return on_mask, off_mask


def _extract_on_off_distributions(session_id: str, base_dir: str, *, bg_subtract: bool = False):
    """Extract per-timeslice beam-on and beam-off current for each IC.

    Returns dict with keys ic1_on, ic1_off, ic2_on, ic2_off, (optionally
    ic3_on, ic3_off), energy, and ``has_ic3`` flag.  Returns None on failure.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    if C_ENERGY not in input_map.columns:
        return None

    energy_by_layer_id: dict | None = None
    energy_by_idx: dict[int, float] | None = None

    if C_LAYER_ID in input_map.columns:
        energy_by_layer_id = input_map.groupby(C_LAYER_ID)[C_ENERGY].first().to_dict()

    if energy_by_layer_id is None or len(energy_by_layer_id) <= 1:
        ordered_energies = list(dict.fromkeys(input_map[C_ENERGY].values))
        energy_by_idx = {i: e for i, e in enumerate(ordered_energies)}

    frames = load_session_timeslice_device_units(src, usecols=_TIMESLICE_COLS)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames
        subtract_background_frames(frames)

    df0 = frames[0].loc[:, ~frames[0].columns.duplicated()]
    if C_IC1_CURRENT not in df0.columns or C_IC2_CURRENT not in df0.columns:
        return None

    ic3_cols = [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D]
    has_ic3 = all(col in df0.columns for col in ic3_cols)

    ic_keys = ["ic1", "ic2"] + (["ic3"] if has_ic3 else [])
    accum = {f"{ic}_{state}": [] for ic in ic_keys for state in ("on", "off", "energy_on", "energy_off")}

    for df in frames:
        df = df.loc[:, ~df.columns.duplicated()]

        energy = None
        if energy_by_idx is not None and "_layer_idx" in df.columns:
            idx = int(df["_layer_idx"].iloc[0])
            energy = energy_by_idx.get(idx)
        if energy is None and energy_by_layer_id is not None and C_LAYER_ID in df.columns:
            layer_id = df[C_LAYER_ID].iloc[0]
            energy = energy_by_layer_id.get(layer_id)
        if energy is None:
            continue

        signals = {
            "ic1": df[C_IC1_CURRENT].values.astype(float),
            "ic2": df[C_IC2_CURRENT].values.astype(float),
        }
        if has_ic3:
            signals["ic3"] = (
                df[C_IC3_CURRENT_A].values.astype(float)
                + df[C_IC3_CURRENT_B].values.astype(float)
                + df[C_IC3_CURRENT_C].values.astype(float)
                + df[C_IC3_CURRENT_D].values.astype(float)
            )

        e_arr = np.full(len(df), energy)

        for ic in ic_keys:
            on_mask, off_mask = _classify_timeslices(signals[ic])
            if on_mask is None:
                continue
            accum[f"{ic}_on"].append(signals[ic][on_mask])
            accum[f"{ic}_energy_on"].append(e_arr[on_mask])
            accum[f"{ic}_off"].append(signals[ic][off_mask])
            accum[f"{ic}_energy_off"].append(e_arr[off_mask])

    if not any(accum[f"{ic}_energy_on"] for ic in ic_keys):
        return None

    result: dict = {"has_ic3": has_ic3}
    for ic in ic_keys:
        for state in ("on", "off"):
            ekey = f"{ic}_energy_{state}"
            vkey = f"{ic}_{state}"
            result[vkey] = np.concatenate(accum[vkey]) if accum[vkey] else np.array([])
            result[ekey] = np.concatenate(accum[ekey]) if accum[ekey] else np.array([])

    return result


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run beam-on / beam-off current box-plot analysis."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        bg = settings.bg_subtract if settings else False
        data = _extract_on_off_distributions(sid, base_dir, bg_subtract=bg)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid beam-on/off data found for any session")
        return

    has_ic3 = any(d.get("has_ic3", False) for d in session_data.values())
    ic_keys = ["ic1", "ic2"] + (["ic3"] if has_ic3 else [])
    ic_titles = ["IC1", "IC2"] + (["IC3 (sum A+B+C+D)"] if has_ic3 else [])
    n_cols = len(ic_keys)

    all_energies: set[float] = set()
    for data in session_data.values():
        for ic in ic_keys:
            ekey_on = f"{ic}_energy_on"
            ekey_off = f"{ic}_energy_off"
            if ekey_on in data:
                all_energies.update(data[ekey_on])
            if ekey_off in data:
                all_energies.update(data[ekey_off])
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    # All panels share the energy x-axis; current y-axis is shared within each row.
    fig, axes = view_grid(
        2, n_cols, cell_w=6.0, cell_h=4.5, sharex=True, sharey="row",
    )

    for col, (ic, title) in enumerate(zip(ic_keys, ic_titles)):
        ax_on = axes[0, col]
        ax_off = axes[1, col]

        on_data = {
            sid: {ic: d[f"{ic}_on"], "energy": pd.Series(d[f"{ic}_energy_on"])}
            for sid, d in session_data.items()
            if f"{ic}_on" in d and d[f"{ic}_on"].size > 0
        }
        off_data = {
            sid: {ic: d[f"{ic}_off"], "energy": pd.Series(d[f"{ic}_energy_off"])}
            for sid, d in session_data.items()
            if f"{ic}_off" in d and d[f"{ic}_off"].size > 0
        }
        on_colors = [colors[loaded_ids.index(sid)] for sid in on_data]
        off_colors = [colors[loaded_ids.index(sid)] for sid in off_data]

        if on_data:
            plot_boxplots_for_column(ax_on, on_data, ic, energies, on_colors, width=0.3)
        ax_on.set_title(f"{title} — Beam On")
        style_energy_axes(ax_on, energies, ylabel="Current" if col == 0 else None)
        ax_on.set_xlabel("")

        if off_data:
            plot_boxplots_for_column(ax_off, off_data, ic, energies, off_colors, width=0.3)
        ax_off.set_title(f"{title} — Beam Off")
        style_energy_axes(ax_off, energies, ylabel="Current" if col == 0 else None)

    for ax in axes.flat:
        ax.label_outer()

    finish_view(
        fig,
        "Beam-On / Beam-Off Current by Energy",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
