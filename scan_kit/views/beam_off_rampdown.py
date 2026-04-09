"""Beam-off ramp-down curve analysis (IC1, IC2, IC3) from timeslice data."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from ..common import (
    load_csv_from_zip,
    load_timeslice_device_units,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    GRID_KW,
    REFLINE_KW,
)

# ---- Tweakable window parameters ------------------------------------------
PRE_OFF_SLICES = 10  # timeslices shown before the falling edge
POST_OFF_SLICES = 10  # timeslices shown after the falling edge
MIN_ON_SLICES = 2  # minimum consecutive above-threshold slices right
# before the falling edge (filters brief spikes)
THRESHOLD_FRAC = 0.10  # fraction of (peak − background) used to define the
# beam-on / beam-off boundary on IC1 current
# ---------------------------------------------------------------------------

_TIMESLICE_COLS = [
    "layer_id",
    "ic1_primary_channel",
    "ic2_primary_channel",
    "ic3_current_A",
    "ic3_current_B",
    "ic3_current_C",
    "ic3_current_D",
]

SESSION_LINESTYLES = ["-", "--", ":", "-."]


def _rampdown_for_signal(signal: np.ndarray) -> np.ndarray | None:
    """Fully self-contained ramp-down extraction for one IC signal.

    1. Compute threshold from this signal's own dynamic range.
    2. Detect falling edges (above → below threshold).
    3. Anchor each window on the last above-threshold slice (t=0).
    4. Validate pre/post windows.
    5. Average all qualifying windows.
    6. Normalise to 0-100 % (peak = 100).

    Returns a 1-D array of length ``PRE_OFF_SLICES + POST_OFF_SLICES``,
    or None if no valid edges are found.
    """
    bg = np.percentile(signal, 25)
    peak = np.percentile(signal, 99)
    if peak - bg < 1.0:
        return None

    thresh = bg + THRESHOLD_FRAC * (peak - bg)
    beam_on = signal > thresh

    diff = np.diff(beam_on.astype(np.int8))
    edge_indices = np.where(diff == -1)[0] + 1

    windows: list[np.ndarray] = []
    n = len(signal)

    for idx in edge_indices:
        anchor = idx - 1  # last above-threshold slice = t=0
        start = anchor - PRE_OFF_SLICES
        end = anchor + POST_OFF_SLICES
        if start < 0 or end > n:
            continue
        if not np.all(beam_on[anchor - MIN_ON_SLICES + 1 : idx]):
            continue
        if np.any(beam_on[idx:end]):
            continue
        windows.append(signal[start:end])

    if not windows:
        return None

    avg = np.mean(windows, axis=0)
    pk = np.nanmax(np.abs(avg))
    if pk < 1.0:
        return None
    return (avg / pk) * 100.0


def _extract_rampdown_curves(session_id: str, base_dir: str):
    """Extract averaged, normalised beam-off ramp-down curves per layer.

    Each IC independently detects edges, extracts windows, and normalises
    using only its own signal.  This keeps each IC's t=0 aligned to its
    own last beam-on moment.

    Returns
    -------
    dict mapping energy (float) -> dict with keys
        ``ic1_curve``, ``ic2_curve``, ``ic3_curve``
        each a 1-D array of length ``PRE_OFF_SLICES + POST_OFF_SLICES``
        scaled to 0–100 % (or None if that IC had no valid edges).
    Returns None on failure.
    """
    zip_path = str(Path(base_dir) / f"{session_id}.zip")

    input_map = load_csv_from_zip(zip_path, "input_map.csv", session_id)
    if input_map is None:
        return None

    energy_by_layer = input_map.groupby("layer_id")["ENERGY"].first().to_dict()

    frames = load_timeslice_device_units(zip_path, session_id, usecols=_TIMESLICE_COLS)
    if not frames:
        return None

    result: dict[float, dict[str, np.ndarray | None]] = {}

    for df in frames:
        layer_id = df["layer_id"].iloc[0]
        energy = energy_by_layer.get(layer_id)
        if energy is None:
            continue

        ic1 = df["ic1_primary_channel"].values
        ic2 = df["ic2_primary_channel"].values
        ic3 = (
            df["ic3_current_A"].values
            + df["ic3_current_B"].values
            + df["ic3_current_C"].values
            + df["ic3_current_D"].values
        )

        c1 = _rampdown_for_signal(ic1)
        c2 = _rampdown_for_signal(ic2)
        c3 = _rampdown_for_signal(ic3)

        if c1 is None and c2 is None and c3 is None:
            continue

        result[energy] = {
            "ic1_curve": c1,
            "ic2_curve": c2,
            "ic3_curve": c3,
        }

    return result if result else None


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run beam-off ramp-down analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_curves: dict[str, dict] = {}
    for sid in session_ids:
        curves = _extract_rampdown_curves(sid, base_dir)
        if curves is not None:
            session_curves[sid] = curves

    if not session_curves:
        print("No valid ramp-down data found for any session")
        return

    all_energies: set[float] = set()
    for curves in session_curves.values():
        all_energies.update(curves.keys())
    e_min = min(all_energies)
    e_max = max(all_energies)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=e_min, vmax=e_max)

    # t=0 = last beam-on slice, t=1 = first beam-off slice
    t_axis = np.arange(-PRE_OFF_SLICES, POST_OFF_SLICES, dtype=float)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2,
        2,
        figsize=FIG_SIZE_2x2,
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Beam-Off Ramp-Down Curves  (normalised to beam-on)", **SUPTITLE_KW)

    loaded_ids = list(session_curves.keys())

    for ax, key, title in [
        (ax1, "ic1_curve", "IC1 Ramp-Down"),
        (ax2, "ic2_curve", "IC2 Ramp-Down"),
        (ax3, "ic3_curve", "IC3 Ramp-Down (sum A+B+C+D)"),
    ]:
        for si, sid in enumerate(loaded_ids):
            ls = SESSION_LINESTYLES[si % len(SESSION_LINESTYLES)]
            for energy in sorted(session_curves[sid].keys()):
                curve = session_curves[sid][energy][key]
                if curve is None:
                    continue
                ax.plot(
                    t_axis,
                    curve,
                    color=cmap(norm(energy)),
                    linestyle=ls,
                    linewidth=1.2,
                    alpha=0.8,
                )

        ax.axvline(x=0.5, **REFLINE_KW)
        ax.axhline(y=100, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(title)
        ax.set_ylabel("Current (%)")
        ax.set_ylim(-15, 150)
        ax.grid(**GRID_KW)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(labelbottom=True, labelleft=True)

    ax3.set_xlabel("Timeslice relative to beam-off (ms)")
    ax2.set_xlabel("Timeslice relative to beam-off (ms)")

    # Place energy colorbar and optional session legend inside the empty ax4 slot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, fraction=0.6, pad=0.05)
    cbar.set_label("Energy (MeV)")
    ax4.axis("off")

    if len(loaded_ids) > 1:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color="gray",
                linestyle=SESSION_LINESTYLES[i % len(SESSION_LINESTYLES)],
                linewidth=1.5,
                label=f"Session {sid}",
            )
            for i, sid in enumerate(loaded_ids)
        ]
        ax4.legend(
            handles=legend_handles, loc="lower center", fontsize=11, frameon=True
        )

    plt.tight_layout()
    plt.show()
