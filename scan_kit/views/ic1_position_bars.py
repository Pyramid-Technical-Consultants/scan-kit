"""IC1 X/Y Position Error box plots for multiple sessions."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    process_position_data,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    SUPTITLE_KW,
    REFLINE_KW,
)

POSITION_KEY = "spot_position_raw"
# Prescribed positions: input map only (spot_data has no expected columns).
EXPECTED_COLS = ("X_POSITION", "Y_POSITION")
# Measured positions in the same mm frame as the input map (raw-remapped ic1_* is a different space).
MEASURED_COLS = ("r_ic1_x_spot_position", "r_ic1_y_spot_position")


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Run IC1 X/Y position error analysis and show matplotlib window.

    Error is **measured** ``r_ic1_*_spot_position`` (scaled mm, same coordinate
    frame as the plan) minus **prescribed** ``X_POSITION`` / ``Y_POSITION`` from
    ``input_map.csv``. Remapped ``ic1_x``/``ic1_y`` from raw registers are not
    subtracted from plan positions — that mixes coordinate systems.
    """
    if not session_ids:
        print("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = process_position_data(
            sid,
            POSITION_KEY,
            extra_spot_columns=list(MEASURED_COLS),
            extra_input_columns=list(EXPECTED_COLS),
            base_dir=base_dir,
        )
        if data is None:
            continue
        data = dict(data)
        if EXPECTED_COLS[0] not in data or EXPECTED_COLS[1] not in data:
            print(
                f"Session {sid}: input_map missing {EXPECTED_COLS[0]}/"
                f"{EXPECTED_COLS[1]}; skipping"
            )
            continue
        if MEASURED_COLS[0] not in data or MEASURED_COLS[1] not in data:
            print(
                f"Session {sid}: spot_data missing {MEASURED_COLS[0]}/"
                f"{MEASURED_COLS[1]}; skipping"
            )
            continue
        data["ic1_x_err"] = np.asarray(data[MEASURED_COLS[0]], dtype=float) - np.asarray(
            data[EXPECTED_COLS[0]], dtype=float
        )
        data["ic1_y_err"] = np.asarray(data[MEASURED_COLS[1]], dtype=float) - np.asarray(
            data[EXPECTED_COLS[1]], dtype=float
        )
        session_data[sid] = data

    if not session_data:
        print("No valid data found for any session")
        return

    all_energies = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    err_cols = ["ic1_x_err", "ic1_y_err"]
    labels = {"ic1_x_err": "IC1 X", "ic1_y_err": "IC1 Y"}

    fig, axes = plt.subplots(2, 2, figsize=(FIG_SIZE_1x2[0], FIG_SIZE_1x2[1] * 2))
    fig.suptitle("IC1 X/Y Position Error by Energy", **SUPTITLE_KW)

    # Row 1: boxplots by energy
    ax_bx, ax_by = axes[0]
    plot_boxplots_for_column(ax_bx, session_data, "ic1_x_err", energies, colors)
    plot_boxplots_for_column(ax_by, session_data, "ic1_y_err", energies, colors)

    box_y_lo = min(ax.get_ylim()[0] for ax in axes[0])
    box_y_hi = max(ax.get_ylim()[1] for ax in axes[0])

    for ax, col in zip([ax_bx, ax_by], err_cols):
        ax.set_title(f"{labels[col]} Position Error")
        style_energy_axes(ax, energies, ylabel="Position Error (mm)")
        ax.set_ylim(box_y_lo, box_y_hi)
        ax.axhline(y=0, **REFLINE_KW)

    make_session_legend(ax_bx, loaded_ids, colors)

    # Row 2: interactive histograms linked to boxplots via SpanSelector
    _selectors = link_boxplot_to_histogram(
        list(axes[0]), list(axes[1]),
        session_data, energies, err_cols, colors, loaded_ids,
        hist_xlabels=[f"{labels[c]} Position Error (mm)" for c in err_cols],
        hist_titles=[f"{labels[c]} error distribution" for c in err_cols],
        hist_refs=[0, 0],
    )
    for ax in axes[1]:
        make_session_legend(ax, loaded_ids, colors)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35)
    plt.show()
