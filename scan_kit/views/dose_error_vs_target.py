"""Dose error vs prescribed target (%) for IC1, IC2, and IC3 by energy."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd
import matplotlib.colors as mcolors

from ..common import (
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_CHARGE_REQ,
    POSITION_KEY_G2_RAW,
    POSITION_KEY_G3_RAW,
    ViewSettings,
    apply_auto_calibration,
    apply_calibration_factors,
    process_position_data,
    plot_boxplots_for_column,
    make_session_legend,
    style_energy_axes,
    annotate_slopes,
    link_boxplot_to_histogram,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)

import logging

_log = logging.getLogger(__name__)

DELIVERED_COLS = {
    "ic1": C_IC1_TOTAL_DOSE,
    "ic2": C_IC2_TOTAL_DOSE,
    "ic3": C_IC3_TOTAL_DOSE,
}
TARGET_COL = C_CHARGE_REQ


def _trend_line_color(face_color):
    """Slightly darker line color from a patch face color."""
    try:
        rgb = mcolors.to_rgb(face_color)
    except ValueError:
        rgb = mcolors.to_rgb("C0")
    return tuple(max(0.0, c * 0.55) for c in rgb)


def _add_trend_and_mean(ax, session_data, column_name, energies, colors,
                        *, position_offset=0.35, zorder=5):
    """Linear trend through per-energy medians, annotated with slope *and* mean error."""
    n_sessions = len(session_data)
    labels: list[tuple[str, tuple[float, float, float]]] = []

    for i, (sid, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue
        all_vals = np.asarray(data[column_name], dtype=float)
        finite = all_vals[np.isfinite(all_vals)]
        mean_err = float(np.mean(finite)) if finite.size else float("nan")

        df = pd.DataFrame({column_name: data[column_name], "energy": data["energy"]})
        e_mev, y_med = [], []
        for energy in energies:
            vals = df.loc[df["energy"] == energy, column_name].values
            if vals.size:
                e_mev.append(float(energy))
                y_med.append(float(np.median(vals)))

        line_color = _trend_line_color(colors[i])
        if len(e_mev) >= 2:
            slope, intercept = np.polyfit(np.array(e_mev), np.array(y_med), 1)
            xs = [j + (i - 0.5) * position_offset for j in range(len(energies))]
            ys = [slope * float(energies[j]) + intercept for j in range(len(energies))]
            ax.plot(xs, ys, color=line_color, linewidth=2.0, linestyle="-",
                    solid_capstyle="round", zorder=zorder, clip_on=True)
            prefix = f"{sid}: " if n_sessions > 1 else ""
            labels.append((
                f"{prefix}{slope:+.4g} %/MeV, mean {mean_err:+.2f}%",
                line_color,
            ))

    if labels:
        annotate_slopes(ax, labels)


def _pct_err_vs_target(delivered, target) -> np.ndarray:
    """``(delivered - target) / target * 100`` where target > 0; else NaN."""
    d = np.asarray(delivered, dtype=float)
    t = np.asarray(target, dtype=float)
    out = np.full_like(d, np.nan, dtype=float)
    ok = np.isfinite(d) & np.isfinite(t) & (np.abs(t) > 1e-15)
    out[ok] = (d[ok] - t[ok]) / t[ok] * 100.0
    return out


CORR_PERCENTILE_CLIP = 100


def _plot_error_correlation(ax, session_data, col_x, col_y, loaded_ids, colors,
                            *, xlabel=None, ylabel=None, percentile_clip=None):
    """Scatter *col_x* vs *col_y* error and annotate with R² and CCC."""
    pclip = percentile_clip if percentile_clip is not None else CORR_PERCENTILE_CLIP

    raw_pairs = []
    for sid in loaded_ids:
        data = session_data[sid]
        if col_x not in data or col_y not in data:
            continue
        x = np.asarray(data[col_x], dtype=float)
        y = np.asarray(data[col_y], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            raw_pairs.append((sid, x[mask], y[mask]))

    if not raw_pairs:
        ax.set_visible(False)
        return

    if pclip < 100:
        all_vals = np.concatenate([np.concatenate([p[1], p[2]]) for p in raw_pairs])
        lo, hi = np.percentile(all_vals, [100 - pclip, pclip])
    else:
        lo, hi = -np.inf, np.inf

    labels: list[tuple[str, tuple[float, float, float]]] = []
    n_sessions = len(loaded_ids)
    all_xy = []
    sid_index = {sid: i for i, sid in enumerate(loaded_ids)}

    for sid, x_raw, y_raw in raw_pairs:
        keep = (x_raw >= lo) & (x_raw <= hi) & (y_raw >= lo) & (y_raw <= hi)
        x, y = x_raw[keep], y_raw[keep]
        if x.size < 2:
            continue

        i = sid_index[sid]
        all_xy.append((x, y))
        ax.scatter(x, y, c=colors[i], alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                   edgecolors="none")

        r, _ = stats.pearsonr(x, y)
        sx, sy = x.std(), y.std()
        mx, my = x.mean(), y.mean()
        ccc = (2 * r * sx * sy) / (sx**2 + sy**2 + (mx - my)**2)

        line_color = _trend_line_color(colors[i])
        slope, intercept = np.polyfit(x, y, 1)
        x_range = np.array([x.min(), x.max()])
        ax.plot(x_range, slope * x_range + intercept, color=line_color,
                linewidth=1.8, linestyle="-", zorder=5)

        prefix = f"{sid}: " if n_sessions > 1 else ""
        labels.append((f"{prefix}R\u00b2={r**2:.5f}  CCC={ccc:.5f}", line_color))

    if all_xy:
        # Let matplotlib autoscale freely, then draw y=x across the visible overlap
        ax.autoscale_view()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ref_lo = max(xlim[0], ylim[0])
        ref_hi = min(xlim[1], ylim[1])
        if ref_lo < ref_hi:
            ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], **REFLINE_KW)

    if labels:
        annotate_slopes(ax, labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(visible=True, alpha=0.3)


def _process_session(session_id: str, position_key: str, base_dir: str,
                     settings: ViewSettings | None = None):
    data = process_position_data(
        session_id,
        position_key,
        extra_spot_columns=list(DELIVERED_COLS.values()),
        extra_input_columns=[TARGET_COL],
        base_dir=base_dir,
    )
    if data is None:
        return None
    if TARGET_COL not in data:
        return None
    if not any(col in data for col in DELIVERED_COLS.values()):
        return None

    data = dict(data)
    if settings and settings.auto_calibrate:
        if settings.cal_factors:
            data = apply_calibration_factors(data, list(DELIVERED_COLS.values()), settings.cal_factors)
        else:
            data = apply_auto_calibration(data, TARGET_COL, list(DELIVERED_COLS.values()))
    target = data[TARGET_COL]
    for ic, col in DELIVERED_COLS.items():
        if col in data:
            data[f"{ic}_dose_err_pct"] = _pct_err_vs_target(data[col], target)
    return data


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

    fig, axes = plt.subplots(
        n_rows, 3, figsize=(FIG_SIZE_2x2[0] + 5, 3.5 * n_rows),
        gridspec_kw={"width_ratios": [4, 1, 1]},
    )
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    fig.suptitle(
        "Dose Error vs Scan Target (% of prescribed dose)",
        **SUPTITLE_KW,
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
        _add_trend_and_mean(
            box_axes[row], session_data, col, energies, colors, position_offset=0.35
        )
        style_energy_axes(box_axes[row], energies, ylabel=f"{titles[col]} Error (% of target)")
        box_axes[row].axhline(y=0, **REFLINE_KW)

    # Hide unused box/hist axes when corr_pairs outnumber err_cols
    for row in range(n_err_rows, n_rows):
        box_axes[row].set_visible(False)
        hist_axes[row].set_visible(False)

    make_session_legend(box_axes[0], loaded_ids, colors)

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
        _plot_error_correlation(
            corr_axes[row], session_data, col_x, col_y, loaded_ids, colors,
            xlabel=f"{titles[col_x]} Error (%)",
            ylabel=f"{titles[col_y]} Error (%)",
            percentile_clip=99.9,
        )

    for row in range(n_corr_rows, n_rows):
        corr_axes[row].set_visible(False)

    plt.tight_layout()
    plt.show()
