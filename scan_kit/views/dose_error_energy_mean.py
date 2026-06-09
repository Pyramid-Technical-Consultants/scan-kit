"""Mean dose error vs prescribed target (%) per energy — scatter (IC1, IC2, IC3)."""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

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
    finish_view,
    style_energy_axes,
    add_energy_trend,
    add_correlation_scatter,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    view_grid,
    REFLINE_KW,
    SCATTER_ALPHA,
)

import logging

_log = logging.getLogger(__name__)

DELIVERED_COLS = DELIVERED_DOSE_COLS
TARGET_COL = C_CHARGE_REQ

# Fewer points than raw spot scatter — keep markers readable
MEAN_SCATTER_SIZE = 72


def _plot_mean_error_scatter(ax, session_data, column_name, energies, colors):
    """One marker per (session, energy): mean spot error at that energy."""
    for i, (sid, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue
        df = pd.DataFrame({column_name: data[column_name], "energy": data["energy"]})
        x_idx, y_mean = [], []
        for j, energy in enumerate(energies):
            vals = df.loc[df["energy"] == energy, column_name].values
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                x_idx.append(float(j))
                y_mean.append(float(np.mean(vals)))
        if x_idx:
            ax.scatter(
                x_idx,
                y_mean,
                c=colors[i],
                alpha=SCATTER_ALPHA,
                s=MEAN_SCATTER_SIZE,
                edgecolors="none",
                zorder=4,
            )


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
    """Plot mean dose error (% of scan target) per IC vs beam energy (scatter)."""
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

    main_axes = [axes[r, 0] for r in range(n_rows)]
    hist_axes = [axes[r, 1] for r in range(n_rows)]
    corr_axes = [axes[r, 2] for r in range(n_rows)]

    for row in range(1, n_err_rows):
        main_axes[row].sharex(main_axes[0])
        main_axes[row].sharey(main_axes[0])

    for row, col in enumerate(err_cols):
        _plot_mean_error_scatter(main_axes[row], session_data, col, energies, colors)
        add_energy_trend(
            main_axes[row], session_data, col, energies, colors,
            agg="mean", show_mean=True,
        )
        style_energy_axes(main_axes[row], energies, ylabel=f"{titles[col]} Error (% of target)")
        main_axes[row].axhline(y=0, **REFLINE_KW)
        main_axes[row].set_xlim(-0.5, len(energies) - 0.5)

    for row in range(n_err_rows, n_rows):
        main_axes[row].set_visible(False)
        hist_axes[row].set_visible(False)

    _selectors = link_boxplot_to_histogram(
        main_axes[:n_err_rows], hist_axes[:n_err_rows],
        session_data, energies, err_cols, colors, loaded_ids,
        hist_xlabels=["Error (% of target)"] * n_err_rows,
        hist_refs=[0] * n_err_rows,
        hist_percentile_clip=99.9,
    )

    for row in range(1, n_err_rows):
        hist_axes[row].sharex(hist_axes[0])

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
        "Mean Dose Error vs Energy (% of prescribed dose)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
    plt.show()
