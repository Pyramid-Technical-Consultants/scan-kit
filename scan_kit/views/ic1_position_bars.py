"""IC1 X/Y Position Error box plots for multiple sessions.

Uses the **non-raw** (processed) position columns which are already in
the same mm coordinate frame as the plan.  Raw register-level positions
are a different space and must NOT be subtracted from plan positions.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    SUPTITLE_KW,
    REFLINE_KW,
)

import logging

_log = logging.getLogger(__name__)


def _process_session(session_id: str, position_key: str, base_dir: str):
    """Load non-raw position data and compute IC1 X/Y error vs plan."""
    data = process_position_data(
        session_id,
        position_key,
        extra_input_columns=[C_X_POSITION, C_Y_POSITION],
        base_dir=base_dir,
    )
    if data is None:
        return None
    data = dict(data)

    if C_X_POSITION not in data or C_Y_POSITION not in data:
        _log.debug("Session %s: input_map missing plan position columns; skipping", session_id)
        return None

    data["ic1_x_err"] = np.asarray(data["ic1_x"], dtype=float) - np.asarray(
        data[C_X_POSITION], dtype=float
    )
    data["ic1_y_err"] = np.asarray(data["ic1_y"], dtype=float) - np.asarray(
        data[C_Y_POSITION], dtype=float
    )
    return data


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Run IC1 X/Y position error analysis and show matplotlib window.

    Error is **measured** non-raw IC1 position (already in plan mm coordinates)
    minus **prescribed** ``X_POSITION`` / ``Y_POSITION`` from ``input_map.csv``.
    Raw register-level positions are NOT used here — they live in a different
    coordinate space.
    """
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _process_session, raw=False)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid data found for any session")
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
