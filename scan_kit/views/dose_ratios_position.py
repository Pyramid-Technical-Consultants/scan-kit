"""Dose ratio vs beam position scatter plots (IC2/IC1, IC3/IC1, IC3/IC2)."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    GRID_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)

POSITION_KEY_G2 = "spot_raw"
POSITION_KEY_G3 = "spot_position_raw"

EXTRA_SPOT_G3 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
    "r_ic3_total_dose_spot_raw",
]
EXTRA_SPOT_G2 = [
    "ic1_total_dose_spot_raw",
    "ic2_total_dose_spot_raw",
]


def _process_session(session_id: str, position_key: str, base_dir: str):
    """Load session, compute dose ratios and radial distance."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3 else EXTRA_SPOT_G2
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra, base_dir=base_dir
    )
    if data is None:
        return None

    ic1_dose = data["ic1_total_dose_spot_raw"]
    ic2_dose = data["ic2_total_dose_spot_raw"]
    data = data.copy()
    data["ic21_ratio"] = ((ic2_dose / ic1_dose) - 1.0) * 100.0
    data["ic1_dist"] = np.sqrt(data["ic1_x"] ** 2 + data["ic1_y"] ** 2)

    if position_key == POSITION_KEY_G3:
        ic3_dose = data["r_ic3_total_dose_spot_raw"]
        data["ic31_ratio"] = ((ic3_dose / ic1_dose) - 1.0) * 100.0
        data["ic32_ratio"] = ((ic3_dose / ic2_dose) - 1.0) * 100.0

    return data


def _scatter_with_trend(ax, x, y, color, label):
    """Scatter *y* vs *x* with a linear trend line and slope annotation."""
    ax.scatter(x, y, c=color, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
               edgecolors="none", label=label)

    mask = np.isfinite(x) & np.isfinite(y)
    xf, yf = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(xf) < 2:
        return

    slope, intercept = np.polyfit(xf, yf, 1)
    x_range = np.array([xf.min(), xf.max()])
    ax.plot(
        x_range,
        slope * x_range + intercept,
        color=color,
        linewidth=2,
        linestyle="-",
        zorder=5,
    )
    return slope


def _plot_ratio_vs_distance(ax, session_data, ratio_col, colors, title):
    """Scatter ratio vs IC1 radial distance for each session."""
    slope_labels = []
    for i, (sid, data) in enumerate(session_data.items()):
        if ratio_col not in data:
            continue
        slope = _scatter_with_trend(
            ax,
            data["ic1_dist"],
            data[ratio_col],
            colors[i],
            f"Session {sid}",
        )
        if slope is not None:
            prefix = f"{sid}: " if len(session_data) > 1 else ""
            slope_labels.append((f"{prefix}{slope:+.4g} % per mm", colors[i]))

    ax.set_title(title)
    ax.set_xlabel("IC1 Radial Distance (mm)")
    ax.set_ylabel("Ratio Difference (%)")
    ax.grid(**GRID_KW)

    if slope_labels:
        annotate_slopes(ax, slope_labels)


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run dose-ratio vs position analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _process_session(sid, POSITION_KEY_G3, base_dir)
        if data is None:
            data = _process_session(sid, POSITION_KEY_G2, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIG_SIZE_2x2, sharex=False, sharey=False
    )
    fig.suptitle("Dose Ratios vs Beam Position", **SUPTITLE_KW)

    if session_data_g3:
        _plot_ratio_vs_distance(
            ax1, session_data_g3, "ic31_ratio", colors_g3,
            "IC3/IC1 vs Radial Distance",
        )

    _plot_ratio_vs_distance(
        ax2, session_data, "ic21_ratio", colors,
        "IC2/IC1 vs Radial Distance",
    )

    if session_data_g3:
        _plot_ratio_vs_distance(
            ax3, session_data_g3, "ic32_ratio", colors_g3,
            "IC3/IC2 vs Radial Distance",
        )

    ax4.axis("off")
    make_session_legend(ax1, loaded_ids, colors)

    plt.tight_layout()
    plt.show()
