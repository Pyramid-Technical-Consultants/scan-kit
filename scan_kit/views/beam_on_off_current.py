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
    resolve_concept_column,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
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


def _extract_on_off_distributions(session_id: str, base_dir: str):
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

    col_layer_im = resolve_concept_column(input_map.columns, C_LAYER_ID)
    col_energy_im = resolve_concept_column(input_map.columns, C_ENERGY)
    if col_layer_im is None or col_energy_im is None:
        return None
    energy_by_layer = input_map.groupby(col_layer_im)[col_energy_im].first().to_dict()

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None

    df0 = frames[0].loc[:, ~frames[0].columns.duplicated()]
    col_layer = resolve_concept_column(df0.columns, C_LAYER_ID)
    col_ic1 = resolve_concept_column(df0.columns, C_IC1_CURRENT)
    col_ic2 = resolve_concept_column(df0.columns, C_IC2_CURRENT)
    if not all([col_layer, col_ic1, col_ic2]):
        return None

    col_ic3a = resolve_concept_column(df0.columns, C_IC3_CURRENT_A)
    col_ic3b = resolve_concept_column(df0.columns, C_IC3_CURRENT_B)
    col_ic3c = resolve_concept_column(df0.columns, C_IC3_CURRENT_C)
    col_ic3d = resolve_concept_column(df0.columns, C_IC3_CURRENT_D)
    has_ic3 = all([col_ic3a, col_ic3b, col_ic3c, col_ic3d])

    ic_keys = ["ic1", "ic2"] + (["ic3"] if has_ic3 else [])
    accum = {f"{ic}_{state}": [] for ic in ic_keys for state in ("on", "off", "energy_on", "energy_off")}

    for df in frames:
        df = df.loc[:, ~df.columns.duplicated()]
        layer_id = df[col_layer].iloc[0]
        energy = energy_by_layer.get(layer_id)
        if energy is None:
            continue

        signals = {
            "ic1": df[col_ic1].values.astype(float),
            "ic2": df[col_ic2].values.astype(float),
        }
        if has_ic3:
            signals["ic3"] = (
                df[col_ic3a].values.astype(float)
                + df[col_ic3b].values.astype(float)
                + df[col_ic3c].values.astype(float)
                + df[col_ic3d].values.astype(float)
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


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run beam-on / beam-off current box-plot analysis."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _extract_on_off_distributions(sid, base_dir)
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

    fig = plt.figure(figsize=(6 * n_cols, 10))
    fig.suptitle("Beam-On / Beam-Off Current by Energy", **SUPTITLE_KW)
    axes = np.empty((2, n_cols), dtype=object)
    axes[0, 0] = fig.add_subplot(2, n_cols, 1)
    for c in range(1, n_cols):
        axes[0, c] = fig.add_subplot(2, n_cols, c + 1, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(2, n_cols, n_cols + 1, sharex=axes[0, 0])
    for c in range(1, n_cols):
        axes[1, c] = fig.add_subplot(2, n_cols, n_cols + c + 1, sharex=axes[0, 0], sharey=axes[1, 0])

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

    make_session_legend(axes[0, 0], loaded_ids, colors)

    plt.tight_layout()
    plt.show()
