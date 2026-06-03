"""Per-spot IC1/IC2 position error (mm) vs beam energy — violin plots.

Uses the **non-raw** (processed) position columns which are already in
the same mm coordinate frame as the plan.  Raw register-level positions
are a different space and must NOT be subtracted from plan positions.

Layout: two columns (IC1, IC2); X and Y error share a column (top/bottom rows).
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_X_POSITION,
    C_Y_POSITION,
    process_position_data,
    try_load_position_data,
    plot_violins_for_column,
    apply_shared_block_labels,
    set_view_header,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    apply_tight_layout,
    REFLINE_KW,
)

_log = logging.getLogger(__name__)

ENERGY_XLABEL = "Energy (MeV)"
ROW_YLABELS = ("X Error (mm)", "Y Error (mm)")

IC_PANELS = (
    ("ic1", "IC1", ("ic1_x_err", "ic1_y_err")),
    ("ic2", "IC2", ("ic2_x_err", "ic2_y_err")),
)
AXIS_LABELS = ("X", "Y")


def _link_axes_keep_tick_labels(axes, master):
    """Share x/y with *master* while keeping tick labels on every axis."""
    for ax in axes:
        if ax is master:
            continue
        ax.sharex(master)
        ax.sharey(master)
    for ax in axes:
        ax.tick_params(labelbottom=True, labelleft=True)


def _process_session(session_id: str, position_key: str, base_dir: str):
    """Load non-raw position data and compute IC1/IC2 X/Y error vs plan."""
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

    plan_x = np.asarray(data[C_X_POSITION], dtype=float)
    plan_y = np.asarray(data[C_Y_POSITION], dtype=float)
    data["ic1_x_err"] = np.asarray(data["ic1_x"], dtype=float) - plan_x
    data["ic1_y_err"] = np.asarray(data["ic1_y"], dtype=float) - plan_y
    data["ic2_x_err"] = np.asarray(data["ic2_x"], dtype=float) - plan_x
    data["ic2_y_err"] = np.asarray(data["ic2_y"], dtype=float) - plan_y
    return data


def _shared_panel_ylim(session_data, err_cols, *, pad_frac=0.05):
    """Y limits spanning every finite error point across violin panels."""
    parts = []
    for col in err_cols:
        for data in session_data.values():
            if col not in data:
                continue
            err = np.asarray(data[col], dtype=float)
            finite = err[np.isfinite(err)]
            if finite.size:
                parts.append(finite)
    if not parts:
        return None
    cat = np.concatenate(parts)
    lo, hi = float(cat.min()), float(cat.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Violin IC1/IC2 X/Y position error by energy and show matplotlib window.

    Error is **measured** non-raw IC position (already in plan mm coordinates)
    minus **prescribed** ``X_POSITION`` / ``Y_POSITION`` from ``input_map.csv``.
    """
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _process_session, raw=False)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid position data found for any session")
        return

    all_energies: set = set()
    for data in session_data.values():
        all_energies.update(np.unique(data["energy"]))
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_SIZE_1x2[0], FIG_SIZE_1x2[1] * 2),
        squeeze=False,
    )

    master = axes[0, 0]
    _link_axes_keep_tick_labels(axes.flat, master)

    err_cols: list[str] = []
    ic_titles = [title for _ic, title, _cols in IC_PANELS]

    for col_idx, (_ic, _ic_title, (x_col, y_col)) in enumerate(IC_PANELS):
        for row_idx, (err_col, _axis_label) in enumerate(zip((x_col, y_col), AXIS_LABELS)):
            ax = axes[row_idx, col_idx]
            plot_violins_for_column(ax, session_data, err_col, energies, colors)

            style_energy_axes(ax, energies, ylabel=None)
            ax.set_xlabel("")
            ax.axhline(y=0, **REFLINE_KW)
            err_cols.append(err_col)

    ylim = _shared_panel_ylim(session_data, err_cols)
    if ylim is not None:
        master.set_ylim(ylim)

    apply_shared_block_labels(
        axes,
        column_titles=ic_titles,
        row_ylabels=ROW_YLABELS,
        xlabel=ENERGY_XLABEL,
        bottom_row=1,
    )

    set_view_header(
        fig,
        "Position Error vs Energy (mm vs plan)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    fig.align_ylabels(axes[:, 0])
    apply_tight_layout()
    plt.show()
