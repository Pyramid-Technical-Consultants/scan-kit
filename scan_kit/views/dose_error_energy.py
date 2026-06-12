"""Dose error vs prescribed target (%) for IC1, IC2, and IC3 by energy."""

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_CHARGE_REQ,
    POSITION_KEY_G2_RAW,
    POSITION_KEY_G3_RAW,
    ViewSettings,
    apply_auto_calibration,
    apply_calibration_factors,
    process_position_data,
    add_dose_error_columns,
    DELIVERED_DOSE_COLS,
    plot_boxplots_for_column,
    finish_view,
    style_energy_axes,
    add_energy_trend,
    add_correlation_scatter,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    view_grid,
    REFLINE_KW,
)

import logging

_log = logging.getLogger(__name__)

DELIVERED_COLS = DELIVERED_DOSE_COLS
TARGET_COL = C_CHARGE_REQ


def _process_session(session_id: str, position_key: str, base_dir: str,
                     settings: ViewSettings | None = None):
    data = process_position_data(
        session_id,
        position_key,
        extra_spot_columns=list(DELIVERED_COLS.values()),
        extra_input_columns=[TARGET_COL],
        base_dir=base_dir,
    )
    if data is None or TARGET_COL not in data:
        return None

    data = dict(data)
    if settings and settings.auto_calibrate:
        if settings.cal_factors:
            data = apply_calibration_factors(data, list(DELIVERED_COLS.values()), settings.cal_factors)
        else:
            data = apply_auto_calibration(data, TARGET_COL, list(DELIVERED_COLS.values()))
    return add_dose_error_columns(data, target_col=TARGET_COL, delivered_cols=DELIVERED_COLS)


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Plot dose error (% of scan target) per IC vs beam energy."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        d = _process_session(sid, POSITION_KEY_G3_RAW, base_dir, settings=settings)
        if d is None:
            d = _process_session(sid, POSITION_KEY_G2_RAW, base_dir, settings=settings)
        if d is not None:
            session_data[sid] = d

    if not session_data:
        _log.debug("No valid dose / target data found for any session")
        return

    all_energies: set = set()
    for d in session_data.values():
        all_energies.update(np.unique(d["energy"]))
    energies = sorted(all_energies)

    err_cols = []
    for ic in ("ic1", "ic2", "ic3"):
        key = f"{ic}_dose_err_pct"
        if any(key in d for d in session_data.values()):
            err_cols.append(key)

    if not err_cols:
        _log.debug("No dose error columns available across all sessions")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    titles = {
        "ic1_dose_err_pct": "IC1",
        "ic2_dose_err_pct": "IC2",
        "ic3_dose_err_pct": "IC3",
    }
    # Circular pairing: IC1→IC2, IC2→IC3, IC3→IC1
    _ALL_CORR = [
        ("ic1_dose_err_pct", "ic2_dose_err_pct"),
        ("ic2_dose_err_pct", "ic3_dose_err_pct"),
        ("ic3_dose_err_pct", "ic1_dose_err_pct"),
    ]
    err_set = set(err_cols)
    corr_pairs = [(a, b) for a, b in _ALL_CORR if a in err_set and b in err_set]
    n_err_rows = len(err_cols)
    n_corr_rows = len(corr_pairs)
    n_rows = max(n_err_rows, n_corr_rows)

    fig, axes = view_grid(
        n_rows, 3, cell_h=3.5,
        gridspec_kw={"width_ratios": [4, 1, 1]},
    )

    box_axes = [axes[r, 0] for r in range(n_rows)]
    hist_axes = [axes[r, 1] for r in range(n_rows)]
    corr_axes = [axes[r, 2] for r in range(n_rows)]

    # Link box axes before plotting so autoscale covers all rows
    for row in range(1, n_err_rows):
        box_axes[row].sharex(box_axes[0])
        box_axes[row].sharey(box_axes[0])

    for row, col in enumerate(err_cols):
        plot_boxplots_for_column(box_axes[row], session_data, col, energies, colors, width=0.3)
        add_energy_trend(
            box_axes[row], session_data, col, energies, colors,
            agg="median", position_offset=0.35, show_mean=True,
        )
        style_energy_axes(box_axes[row], energies, ylabel=f"{titles[col]} Error (% of target)")
        box_axes[row].axhline(y=0, **REFLINE_KW)

    # Hide unused box/hist axes when corr_pairs outnumber err_cols
    for row in range(n_err_rows, n_rows):
        box_axes[row].set_visible(False)
        hist_axes[row].set_visible(False)

    _selectors = link_boxplot_to_histogram(
        box_axes[:n_err_rows], hist_axes[:n_err_rows],
        session_data, energies, err_cols, colors, loaded_ids,
        hist_xlabels=["Error (% of target)"] * n_err_rows,
        hist_refs=[0] * n_err_rows,
        hist_percentile_clip=99.9,
    )

    for row in range(1, n_err_rows):
        hist_axes[row].sharex(hist_axes[0])

    # Error correlation scatter plots (right column)
    for row, (col_x, col_y) in enumerate(corr_pairs):
        add_correlation_scatter(
            corr_axes[row], session_data, col_x, col_y, loaded_ids, colors,
            xlabel=f"{titles[col_x]} Error (%)",
            ylabel=f"{titles[col_y]} Error (%)",
            percentile_clip=99.9,
        )

    for row in range(n_corr_rows, n_rows):
        corr_axes[row].set_visible(False)

    finish_view(
        fig,
        "Dose Error vs Energy (% of prescribed dose)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
