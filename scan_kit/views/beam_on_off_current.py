"""Beam-on vs beam-off current box plots (IC1, IC2, IC3) from timeslice data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
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

ON_FRAC = 0.10
OFF_FRAC = 0.02

_TIMESLICE_COLS = [
    "layer_id",
    "ic1_primary_channel",
    "ic2_primary_channel",
    "ic3_current_A",
    "ic3_current_B",
    "ic3_current_C",
    "ic3_current_D",
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
    if dyn < 1.0:
        return None, None
    on_mask = signal > (bg + ON_FRAC * dyn)
    off_mask = signal < (bg + OFF_FRAC * dyn)
    return on_mask, off_mask


def _extract_on_off_distributions(session_id: str, base_dir: str):
    """Extract per-timeslice beam-on and beam-off current for each IC.

    Returns dict with keys ic1_on, ic1_off, ic2_on, ic2_off, ic3_on, ic3_off,
    and energy — each a 1-D array aligned element-wise.  Returns None on failure.
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None

    energy_by_layer = input_map.groupby("layer_id")["ENERGY"].first().to_dict()

    frames = load_session_timeslice_device_units(src, usecols=_TIMESLICE_COLS)
    if not frames:
        return None

    ic_keys = ["ic1", "ic2", "ic3"]
    accum = {f"{ic}_{state}": [] for ic in ic_keys for state in ("on", "off", "energy_on", "energy_off")}

    for df in frames:
        layer_id = df["layer_id"].iloc[0]
        energy = energy_by_layer.get(layer_id)
        if energy is None:
            continue

        signals = {
            "ic1": df["ic1_primary_channel"].values,
            "ic2": df["ic2_primary_channel"].values,
            "ic3": (
                df["ic3_current_A"].values
                + df["ic3_current_B"].values
                + df["ic3_current_C"].values
                + df["ic3_current_D"].values
            ),
        }

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

    result = {}
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
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _extract_on_off_distributions(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid beam-on/off data found for any session")
        return

    all_energies: set[float] = set()
    for data in session_data.values():
        for ic in ("ic1", "ic2", "ic3"):
            all_energies.update(data[f"{ic}_energy_on"])
            all_energies.update(data[f"{ic}_energy_off"])
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Beam-On / Beam-Off Current by Energy", **SUPTITLE_KW)
    axes = np.empty((2, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 3, 1)
    axes[0, 1] = fig.add_subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[0, 2] = fig.add_subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(2, 3, 4, sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(2, 3, 5, sharex=axes[0, 0], sharey=axes[1, 0])
    axes[1, 2] = fig.add_subplot(2, 3, 6, sharex=axes[0, 0], sharey=axes[1, 0])

    ic_keys = ["ic1", "ic2", "ic3"]
    ic_titles = ["IC1", "IC2", "IC3 (sum A+B+C+D)"]

    for col, (ic, title) in enumerate(zip(ic_keys, ic_titles)):
        ax_on = axes[0, col]
        ax_off = axes[1, col]

        on_data = {
            sid: {ic: d[f"{ic}_on"], "energy": pd.Series(d[f"{ic}_energy_on"])}
            for sid, d in session_data.items()
            if d[f"{ic}_on"].size > 0
        }
        off_data = {
            sid: {ic: d[f"{ic}_off"], "energy": pd.Series(d[f"{ic}_energy_off"])}
            for sid, d in session_data.items()
            if d[f"{ic}_off"].size > 0
        }
        on_colors = [colors[loaded_ids.index(sid)] for sid in on_data]
        off_colors = [colors[loaded_ids.index(sid)] for sid in off_data]

        if on_data:
            plot_boxplots_for_column(ax_on, on_data, ic, energies, on_colors, width=0.3)
        ax_on.set_title(f"{title} — Beam On")
        style_energy_axes(ax_on, energies, ylabel="Current (nA)")

        if off_data:
            plot_boxplots_for_column(ax_off, off_data, ic, energies, off_colors, width=0.3)
        ax_off.set_title(f"{title} — Beam Off")
        style_energy_axes(ax_off, energies, ylabel="Current (nA)")

    make_session_legend(axes[0, 0], loaded_ids, colors)

    plt.tight_layout()
    plt.show()
