"""IC1/IC2 sigma X/Y (mm) vs beam energy — violin plots.

Sigma spot columns are resolved from ``spot_sigma`` / ``spot_sigma_raw`` and
scaled ×2 to mm (same convention as the former box-plot view).

Layout: two columns (IC1, IC2); X and Y share a column (top/bottom rows).
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    load_session_raw,
    create_valid_mask,
    resolve_concept_column,
    C_ENERGY,
    plot_violins_for_column,
    apply_shared_block_labels,
    set_view_header,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_1x2,
    apply_tight_layout,
)

_log = logging.getLogger(__name__)

ENERGY_XLABEL = "Energy (MeV)"
ROW_YLABELS = ("X Sigma (mm)", "Y Sigma (mm)")

_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")

IC_PANELS = (
    ("ic1", "IC1", ("ic1_sig_x", "ic1_sig_y")),
    ("ic2", "IC2", ("ic2_sig_x", "ic2_sig_y")),
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


def _resolve_sigma_col(columns, ic: str, axis: str) -> str | None:
    """Find a sigma column for a given IC and axis, trying known key variants."""
    for key in _SIG_KEY_VARIANTS:
        for prefix in (f"r_{ic}_{axis}_{key}", f"{ic}_{axis}_{key}"):
            if prefix in columns:
                return prefix
    return None


def _process_session(session_id: str, base_dir: str):
    """Load per-spot sigma columns and energy for one session."""
    input_map, spot_data = load_session_raw(session_id, base_dir=base_dir)
    if input_map is None or spot_data is None:
        return None

    energy_col = resolve_concept_column(input_map.columns, C_ENERGY)
    if energy_col is None:
        _log.debug("Session %s: no energy column found", session_id)
        return None

    found: dict[str, str] = {}
    for label, ic, axis in (
        ("ic1_sig_x", "ic1", "x"),
        ("ic1_sig_y", "ic1", "y"),
        ("ic2_sig_x", "ic2", "x"),
        ("ic2_sig_y", "ic2", "y"),
    ):
        col = _resolve_sigma_col(spot_data.columns, ic, axis)
        if col is not None:
            found[label] = col

    if not found:
        _log.debug("Session %s: no sigma columns found", session_id)
        return None

    spot_data = spot_data[list(found.values())].copy().join(input_map[energy_col])
    spot_data = spot_data.apply(pd.to_numeric, errors="coerce")
    spot_data_clean = spot_data[create_valid_mask(spot_data)]

    if spot_data_clean.empty:
        return None

    result: dict = {
        "session_id": session_id,
        "energy": spot_data_clean[energy_col].values,
    }
    for label, raw_col in found.items():
        result[label] = spot_data_clean[raw_col].values * 2.0
    return result


def _shared_panel_ylim(session_data, value_cols, *, pad_frac=0.05):
    """Y limits spanning every finite sigma value across violin panels."""
    parts = []
    for col in value_cols:
        for data in session_data.values():
            if col not in data:
                continue
            vals = np.asarray(data[col], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                parts.append(finite)
    if not parts:
        return None
    cat = np.concatenate(parts)
    lo = max(0.0, float(cat.min()))
    hi = float(cat.max())
    if hi <= lo:
        hi = lo + 1.0
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Violin IC1/IC2 X/Y sigma by energy and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        data = _process_session(sid, base_dir)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid sigma data found for any session")
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

    value_cols: list[str] = []
    ic_titles = [title for _ic, title, _cols in IC_PANELS]

    for col_idx, (_ic, _ic_title, (x_col, y_col)) in enumerate(IC_PANELS):
        for row_idx, (value_col, _axis_label) in enumerate(zip((x_col, y_col), AXIS_LABELS)):
            ax = axes[row_idx, col_idx]
            plot_violins_for_column(ax, session_data, value_col, energies, colors)

            style_energy_axes(ax, energies, ylabel=None)
            ax.set_xlabel("")
            value_cols.append(value_col)

    ylim = _shared_panel_ylim(session_data, value_cols)
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
        "Sigma vs Energy (mm)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    fig.align_ylabels(axes[:, 0])
    apply_tight_layout()
    plt.show()
