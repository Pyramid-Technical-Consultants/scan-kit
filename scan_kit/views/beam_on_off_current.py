"""Beam-on vs beam-off current analysis (IC1, IC2, IC3) from timeslice data."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    load_csv_from_zip,
    load_timeslice_device_units,
    FIG_SIZE_1x2,
    SUPTITLE_KW,
    GRID_KW,
)

# ---- Tweakable parameters -------------------------------------------------
ON_FRAC = 0.10   # fraction of dynamic range above background → beam-on
OFF_FRAC = 0.02  # fraction of dynamic range above background → beam-off ceiling
# Samples between OFF_FRAC and ON_FRAC are transition (ramp-up/down) and excluded.
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


def _classify_signal(signal: np.ndarray):
    """Return (beam_on_mean, beam_off_mean) for a single IC signal.

    Two thresholds derived from the signal's dynamic range split the data
    into three bands:

    * **beam-on**  — above ``bg + ON_FRAC * (peak − bg)``
    * **beam-off** — below ``bg + OFF_FRAC * (peak − bg)``
    * **transition** (ramp-up / ramp-down) — between the two, excluded
      from both averages so it does not dilute either measurement.
    """
    clean = signal[~np.isnan(signal)]
    if len(clean) == 0:
        return None, None

    bg = np.percentile(clean, 25)
    pk = np.percentile(clean, 99)
    dyn = pk - bg
    if dyn < 1.0:
        return None, None

    on_thresh = bg + ON_FRAC * dyn
    off_thresh = bg + OFF_FRAC * dyn

    on_mask = clean > on_thresh
    off_mask = clean < off_thresh

    on_mean = float(np.mean(clean[on_mask])) if on_mask.any() else None
    off_mean = float(np.mean(clean[off_mask])) if off_mask.any() else None
    return on_mean, off_mean


def _extract_on_off(session_id: str, base_dir: str):
    """Extract per-layer beam-on and beam-off mean current for each IC.

    Returns
    -------
    dict mapping energy (float) -> dict with keys
        ``ic1_on``, ``ic1_off``, ``ic2_on``, ``ic2_off``,
        ``ic3_on``, ``ic3_off`` — each a float or None.
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

    result: dict[float, dict] = {}

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

        ic1_on, ic1_off = _classify_signal(ic1)
        ic2_on, ic2_off = _classify_signal(ic2)
        ic3_on, ic3_off = _classify_signal(ic3)

        result[energy] = {
            "ic1_on": ic1_on, "ic1_off": ic1_off,
            "ic2_on": ic2_on, "ic2_off": ic2_off,
            "ic3_on": ic3_on, "ic3_off": ic3_off,
        }

    return result if result else None


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run beam-on / beam-off current analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _extract_on_off(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid beam-on/off data found for any session")
        return

    loaded_ids = list(session_data.keys())

    IC_COLORS = {"ic1": "tab:blue", "ic2": "tab:orange", "ic3": "tab:green"}
    IC_LABELS = {"ic1": "IC1", "ic2": "IC2", "ic3": "IC3 (sum A+B+C+D)"}
    SESSION_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, (ax_on, ax_off) = plt.subplots(
        1, 2, figsize=FIG_SIZE_1x2, sharex=True,
    )
    fig.suptitle("Beam-On vs Beam-Off Current by Energy", **SUPTITLE_KW)

    for ax, state, title in [
        (ax_on, "on", "Beam-On Current"),
        (ax_off, "off", "Beam-Off Current"),
    ]:
        for si, sid in enumerate(loaded_ids):
            data = session_data[sid]
            energies = sorted(data.keys())
            marker = SESSION_MARKERS[si % len(SESSION_MARKERS)]

            for prefix in ["ic1", "ic2", "ic3"]:
                es, vs = [], []
                for e in energies:
                    v = data[e][f"{prefix}_{state}"]
                    if v is not None:
                        es.append(e)
                        vs.append(v)
                label = (
                    f"{IC_LABELS[prefix]}"
                    if si == 0 else None
                )
                ax.plot(
                    es, vs,
                    marker=marker, markersize=4, linewidth=1, alpha=0.8,
                    color=IC_COLORS[prefix], label=label,
                )

        ax.set_title(title)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Current (nA)")
        ax.grid(**GRID_KW)

    # Build legend: IC colors + session markers
    legend_handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, label=IC_LABELS[k])
        for k, c in IC_COLORS.items()
    ]
    if len(loaded_ids) > 1:
        legend_handles.append(plt.Line2D(
            [0], [0], color="none", label="",
        ))
        for si, sid in enumerate(loaded_ids):
            legend_handles.append(plt.Line2D(
                [0], [0], color="gray",
                marker=SESSION_MARKERS[si % len(SESSION_MARKERS)],
                markersize=6, linewidth=0,
                label=f"Session {sid}",
            ))

    ax_on.legend(handles=legend_handles, loc="best", fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()
