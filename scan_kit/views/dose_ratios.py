"""Dose ratio box plots (IC2/IC1, IC3/IC1, IC3/IC2) for G2 and G3 sessions."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    plot_boxplots_for_column,
    annotate_slopes,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
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


def _process_ratios_session(session_id: str, position_key: str, base_dir: str):
    """Process session and compute dose ratios."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3 else EXTRA_SPOT_G2
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra, base_dir=base_dir
    )
    if data is None:
        return None

    ic1_dose = data["ic1_total_dose_spot_raw"]
    ic2_dose = data["ic2_total_dose_spot_raw"]
    data = dict(data)
    data["ic21_ratio"] = ((ic2_dose / ic1_dose) - 1.0) * 100.0

    if position_key == POSITION_KEY_G3:
        ic3_dose = data["r_ic3_total_dose_spot_raw"]
        data["ic31_ratio"] = ((ic3_dose / ic1_dose) - 1.0) * 100.0
        data["ic32_ratio"] = ((ic3_dose / ic2_dose) - 1.0) * 100.0

    return data


def _trend_line_color(face_color):
    """Slightly darker line color from a patch face color name or tuple."""
    try:
        rgb = mcolors.to_rgb(face_color)
    except ValueError:
        rgb = mcolors.to_rgb("C0")
    return tuple(max(0.0, c * 0.55) for c in rgb)


def _add_median_trend_lines(
    ax,
    session_data: dict,
    column_name: str,
    energies: list,
    colors: list,
    *,
    position_offset: float = 0.35,
    zorder: float = 5,
):
    """Linear trend through per-energy medians.

    ``column_name`` values are ratio *differences in percent* (see
    ``(ratio - 1) * 100`` in :func:`_process_ratios_session`). Fitting
    ``y`` vs beam energy in MeV gives slope in **percent per MeV** (change in
    that plotted quantity per MeV).
    """
    n_sessions = len(session_data)
    # (label text, line color) for a compact in-axes legend
    slope_labels: list[tuple[str, tuple[float, float, float]]] = []

    for enum_i, (sid, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue
        df = pd.DataFrame({column_name: data[column_name], "energy": data["energy"]})
        e_mev = []
        y_med = []
        for energy in energies:
            vals = df.loc[df["energy"] == energy, column_name].values
            if vals.size == 0:
                continue
            e_mev.append(float(energy))
            y_med.append(float(np.median(vals)))

        if len(e_mev) < 2:
            continue

        # y_med is in %; e_mev is in MeV -> slope is % per MeV (not a hidden /100)
        slope, intercept = np.polyfit(np.array(e_mev), np.array(y_med), 1)
        line_color = _trend_line_color(colors[enum_i])
        xs_line = [
            j + (enum_i - 0.5) * position_offset for j in range(len(energies))
        ]
        ys_line = [slope * float(energies[j]) + intercept for j in range(len(energies))]
        ax.plot(
            xs_line,
            ys_line,
            color=line_color,
            linewidth=2.0,
            linestyle="-",
            solid_capstyle="round",
            zorder=zorder,
            clip_on=True,
        )
        prefix = f"{sid}: " if n_sessions > 1 else ""
        slope_labels.append(
            (
                f"{prefix}{slope:+.4g} % per MeV",
                line_color,
            )
        )

    if slope_labels:
        annotate_slopes(ax, slope_labels)


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run dose ratios analysis and show matplotlib window."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = _process_ratios_session(sid, POSITION_KEY_G3, base_dir)
        if data is None:
            data = _process_ratios_session(sid, POSITION_KEY_G2, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=FIG_SIZE_2x2, sharex=False, sharey=False
    )
    fig.suptitle("Dose Ratios vs Energy", **SUPTITLE_KW)

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    all_values = np.concatenate([data["ic21_ratio"] for data in session_data.values()])
    min_val = all_values.min()
    max_val = all_values.max()
    padding = (max_val - min_val) * 0.05
    y_min = min_val - padding
    y_max = max_val + padding

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    if session_data_g3:
        plot_boxplots_for_column(
            ax1, session_data_g3, "ic31_ratio", energies, colors_g3, width=0.3
        )
        _add_median_trend_lines(
            ax1, session_data_g3, "ic31_ratio", energies, colors_g3, position_offset=0.35
        )
    ax1.set_title("IC3/IC1 Ratio Difference (%)")
    style_energy_axes(ax1, energies, ylabel="IC3/IC1 Ratio Difference (%)")
    ax1.set_ylim(y_min, y_max)

    plot_boxplots_for_column(
        ax2, session_data, "ic21_ratio", energies, colors, width=0.3
    )
    _add_median_trend_lines(
        ax2, session_data, "ic21_ratio", energies, colors, position_offset=0.35
    )
    ax2.set_title("IC2/IC1 Ratio Difference (%)")
    style_energy_axes(ax2, energies, ylabel="IC2/IC1 Ratio Difference (%)")
    ax2.set_ylim(y_min, y_max)

    if session_data_g3:
        plot_boxplots_for_column(
            ax3, session_data_g3, "ic32_ratio", energies, colors_g3, width=0.3
        )
        _add_median_trend_lines(
            ax3, session_data_g3, "ic32_ratio", energies, colors_g3, position_offset=0.35
        )
    ax3.set_title("IC3/IC2 Ratio Difference (%)")
    style_energy_axes(ax3, energies, ylabel="IC3/IC2 Ratio Difference (%)")
    ax3.set_ylim(y_min, y_max)

    ax4.axis("off")
    make_session_legend(ax1, loaded_ids, colors)

    plt.tight_layout()
    plt.show()
