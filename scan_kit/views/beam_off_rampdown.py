"""Beam-off ramp-down curve analysis (IC1, IC2, IC3) from timeslice data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

from ..common import (
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    C_LAYER_ID,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
    GRID_KW,
    REFLINE_KW,
)
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

import logging

_log = logging.getLogger(__name__)

# ---- Tweakable window parameters ------------------------------------------
PRE_OFF_SLICES = 2  # timeslices shown before the falling edge
POST_OFF_SLICES = 10  # timeslices shown after the falling edge
MIN_ON_SLICES = 2  # minimum consecutive above-threshold slices right
# before the falling edge (filters brief spikes)
THRESHOLD_FRAC = 0.10  # fraction of (peak − background) used to define the
# beam-on / beam-off boundary on IC1 current
# ---------------------------------------------------------------------------

_TIMESLICE_COLS = [
    C_LAYER_ID,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
]


def _extract_windows(signal: np.ndarray) -> list[np.ndarray]:
    """Extract qualifying ramp-down windows from one IC signal (one layer).

    Each window is background-subtracted and centred on the last beam-on
    slice (t=0).  Returns a list of raw (not normalised) windows.
    """
    if np.isnan(signal).any():
        signal = signal.copy()
        signal[np.isnan(signal)] = np.nanmedian(signal)
    low_mask = signal <= np.percentile(signal, 25)
    bg = np.median(signal[low_mask]) if low_mask.any() else np.percentile(signal, 25)
    peak = np.percentile(signal, 99)
    if peak - bg < 1.0:
        return []

    sig = signal - bg

    thresh = THRESHOLD_FRAC * (peak - bg)
    beam_on = sig > thresh

    diff = np.diff(beam_on.astype(np.int8))
    edge_indices = np.where(diff == -1)[0] + 1

    windows: list[np.ndarray] = []
    n = len(sig)

    for idx in edge_indices:
        anchor = idx - 1
        start = anchor - PRE_OFF_SLICES
        end = anchor + POST_OFF_SLICES
        if start < 0 or end > n:
            continue
        if not np.all(beam_on[anchor - MIN_ON_SLICES + 1 : idx]):
            continue
        if np.any(beam_on[idx:end]):
            continue
        windows.append(sig[start:end])

    return windows


def _normalise_windows(windows: list[np.ndarray]) -> np.ndarray | None:
    """Average and normalise a collection of ramp-down windows to 0-100 %."""
    if not windows:
        return None
    avg = np.mean(windows, axis=0)
    pk = np.nanmax(np.abs(avg))
    if pk < 1.0:
        return None
    return (avg / pk) * 100.0


def _extract_rampdown_curves(session_id: str, base_dir: str):
    """Extract averaged, normalised beam-off ramp-down curves per energy.

    Windows from all layers sharing the same energy are pooled before
    averaging, so single-energy sessions still produce good curves from
    the combined data of all layers.

    Returns
    -------
    dict mapping energy (float) -> dict with keys
        ``ic1_curve``, ``ic2_curve``, ``ic3_curve``
        each a 1-D array of length ``PRE_OFF_SLICES + POST_OFF_SLICES``
        scaled to 0–100 % (or None if that IC had no valid edges).
    Returns None on failure.
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

    ic3_cols = [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D]

    windows_by_energy: dict[float, dict[str, list[np.ndarray]]] = {}

    for df in frames:
        energy = None
        if energy_by_idx is not None and "_layer_idx" in df.columns:
            idx = int(df["_layer_idx"].iloc[0])
            energy = energy_by_idx.get(idx)
        if energy is None and energy_by_layer_id is not None and C_LAYER_ID in df.columns:
            layer_id = df[C_LAYER_ID].iloc[0]
            energy = energy_by_layer_id.get(layer_id)
        if energy is None:
            continue

        if C_IC1_CURRENT not in df.columns and C_IC2_CURRENT not in df.columns:
            continue

        if energy not in windows_by_energy:
            windows_by_energy[energy] = {"ic1": [], "ic2": [], "ic3": []}
        bucket = windows_by_energy[energy]

        if C_IC1_CURRENT in df.columns:
            bucket["ic1"].extend(_extract_windows(df[C_IC1_CURRENT].values))
        if C_IC2_CURRENT in df.columns:
            bucket["ic2"].extend(_extract_windows(df[C_IC2_CURRENT].values))

        has_ic3 = all(col in df.columns for col in ic3_cols)
        if has_ic3:
            ic3_sig = (
                df[C_IC3_CURRENT_A].values
                + df[C_IC3_CURRENT_B].values
                + df[C_IC3_CURRENT_C].values
                + df[C_IC3_CURRENT_D].values
            )
            bucket["ic3"].extend(_extract_windows(ic3_sig))

    per_energy: dict[float, dict[str, np.ndarray | None]] = {}
    all_windows: dict[str, list[np.ndarray]] = {"ic1": [], "ic2": [], "ic3": []}

    for energy, bucket in windows_by_energy.items():
        c1 = _normalise_windows(bucket["ic1"])
        c2 = _normalise_windows(bucket["ic2"])
        c3 = _normalise_windows(bucket["ic3"])
        if c1 is None and c2 is None and c3 is None:
            continue
        per_energy[energy] = {
            "ic1_curve": c1,
            "ic2_curve": c2,
            "ic3_curve": c3,
        }
        for ic in ("ic1", "ic2", "ic3"):
            all_windows[ic].extend(bucket[ic])

    if not per_energy:
        return None

    return {
        "per_energy": per_energy,
        "avg": {
            "ic1_curve": _normalise_windows(all_windows["ic1"]),
            "ic2_curve": _normalise_windows(all_windows["ic2"]),
            "ic3_curve": _normalise_windows(all_windows["ic3"]),
        },
    }


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run beam-off ramp-down analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_results: dict[str, dict] = {}
    for sid in session_ids:
        result = _extract_rampdown_curves(sid, base_dir)
        if result is not None:
            session_results[sid] = result

    if not session_results:
        _log.debug("No valid ramp-down data found for any session")
        return

    loaded_ids = list(session_results.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    ic_defs = [
        ("ic1_curve", "IC1"),
        ("ic2_curve", "IC2"),
        ("ic3_curve", "IC3 (A+B+C+D)"),
    ]
    show_ic3 = any(
        r["avg"]["ic3_curve"] is not None for r in session_results.values()
    )
    if not show_ic3:
        ic_defs = ic_defs[:2]
    n_ic = len(ic_defs)

    t_axis = np.arange(-PRE_OFF_SLICES, POST_OFF_SLICES, dtype=float)
    win_len = PRE_OFF_SLICES + POST_OFF_SLICES

    all_energies: set[float] = set()
    for r in session_results.values():
        all_energies.update(r["per_energy"].keys())
    

    n_sessions = len(loaded_ids)
    n_rows = 1 + n_sessions  # row 0 = averages, rows 1..N = heatmaps
    height_ratios = [1] + [1.3] * n_sessions
    n_cols_grid = n_ic + 1  # extra narrow column for the colorbar
    width_ratios = [1] * n_ic + [0.04]

    fig, axes = plt.subplots(
        n_rows, n_cols_grid,
        figsize=(6 * n_ic + 0.6, 4 + 4 * n_sessions),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios, "width_ratios": width_ratios},
    )
    fig.suptitle("Beam-Off Ramp-Down Curves  (normalised to beam-on)", **SUPTITLE_KW)

    for row_idx in range(n_rows):
        axes[row_idx, -1].set_visible(False)

    # --- Row 0: grand-average per session ---
    for col, (key, label) in enumerate(ic_defs):
        ax = axes[0, col]
        for si, sid in enumerate(loaded_ids):
            curve = session_results[sid]["avg"][key]
            if curve is None:
                continue
            ax.plot(
                t_axis, curve,
                color=colors[si], linewidth=1.5,
                label=sid,
            )
        ax.axvline(x=0.5, **REFLINE_KW)
        ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(label)
        if col == 0:
            ax.set_ylabel("Current (%)")
        ax.set_ylim(-15, 115)
        ax.grid(**GRID_KW)
        ax.xaxis.set_major_locator(MultipleLocator(1))

    axes[0, 0].legend(loc="upper right", fontsize=8)

    # --- Rows 1..N: one heatmap row per session (ramp-down only: t >= 1) ---
    heatmap_cmap = "turbo"
    ramp_start = PRE_OFF_SLICES + 1
    ramp_t = t_axis[ramp_start:]
    ramp_len = len(ramp_t)

    heatmap_vals: list[float] = []
    for r in session_results.values():
        for pe_data in r["per_energy"].values():
            for key, _ in ic_defs:
                c = pe_data[key]
                if c is not None:
                    ramp = c[ramp_start:]
                    heatmap_vals.extend(ramp[np.isfinite(ramp)])
    if heatmap_vals:
        heat_vmin = float(np.min(heatmap_vals))
        heat_vmax = float(np.max(heatmap_vals))
    else:
        heat_vmin, heat_vmax = 0.0, 100.0

    for si, sid in enumerate(loaded_ids):
        row = 1 + si
        pe = session_results[sid]["per_energy"]
        energies = sorted(pe.keys())

        _UPSAMPLE_X = 8
        x_orig = np.arange(ramp_len, dtype=float)
        x_fine = np.linspace(0, ramp_len - 1, ramp_len * _UPSAMPLE_X)

        for col, (key, label) in enumerate(ic_defs):
            ax = axes[row, col]
            img = np.full((len(energies), ramp_len), np.nan)
            for ei, e in enumerate(energies):
                c = pe[e][key]
                if c is not None:
                    img[ei, :] = c[ramp_start:]

            img_smooth = np.full((len(energies), len(x_fine)), np.nan)
            for ri in range(len(energies)):
                if np.any(np.isfinite(img[ri])):
                    img_smooth[ri] = np.interp(x_fine, x_orig, img[ri])

            extent = [ramp_t[0] - 0.5, ramp_t[-1] + 0.5,
                      energies[0], energies[-1]]
            ax.imshow(
                img_smooth, aspect="auto", origin="lower",
                extent=extent, cmap=heatmap_cmap,
                vmin=heat_vmin, vmax=heat_vmax,
                interpolation="nearest",
            )
            if col == 0:
                ax.set_ylabel(f"{sid}\nEnergy (MeV)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Time after beam-off (ms)")
            ax.xaxis.set_major_locator(MultipleLocator(1))

    cbar_ax = fig.add_subplot(
        axes[1, -1].get_gridspec()[1:, -1],
    )
    cbar_ax.set_visible(True)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=mcolors.Normalize(heat_vmin, heat_vmax)),
        cax=cbar_ax,
    )
    cbar.set_label("Current (%)")

    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.show()
