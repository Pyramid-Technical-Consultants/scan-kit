"""Per-spot dose error (% of target) vs target MU — scatter + per-MU histograms.

Row 1: per-spot scatter (IC1, IC2, IC3) vs target MU.
Rows 2+: overlapping error histograms, one row per shared target-MU bin.
"""

import logging

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
    add_scatter_trend,
    make_trend_legend,
    trend_session_prefix,
    finish_view,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    REFLINE_KW,
    GRID_KW,
    DELIVERED_DOSE_COLS,
    SCATTER_SIZE,
)

_log = logging.getLogger(__name__)

TARGET_COL = C_CHARGE_REQ
IC_TITLES = {"ic1": "IC1", "ic2": "IC2", "ic3": "IC3"}
SCATTER_YLABEL = "Error (% of target)"
SCATTER_XLABEL = "Target MU"
HIST_XLABEL = "Dose Error (%)"

# Histogram binning (shared MU edges across sessions; error edges per MU bin).
N_MU_BINS = 5
N_ERR_BINS = 64

# Safety-gate tolerance on delivered dose: fixed floor + fraction of target MU.
GATE_ABS_MU = 0.002  # absolute MU floor
GATE_LEVELS = (
    (0.01, "green", "Gate \u00b1(0.002 MU + 1%)"),
    (0.02, "orange", "Gate \u00b1(0.002 MU + 2%)"),
    (0.03, "red", "Gate \u00b1(0.002 MU + 3%)"),
)
GATE_LINE_KW = dict(linestyle="--", linewidth=1.3, alpha=0.85, zorder=6)
HIST_GATE_KW = dict(linestyle="--", linewidth=1.0, alpha=0.7, zorder=4)


def _gate_threshold_pct(mu, gate_frac):
    """Gate tolerance as a percentage of target MU: ``(0.002 + frac*mu)/mu*100``."""
    mu = np.asarray(mu, dtype=float)
    return (GATE_ABS_MU + gate_frac * mu) / mu * 100.0


def _compute_mu_bin_edges(mu_values, n_bins):
    """Log-spaced MU bin edges shared by every session."""
    mu = np.asarray(mu_values, dtype=float)
    mu = mu[np.isfinite(mu) & (mu > 0)]
    if mu.size == 0:
        return None
    lo, hi = float(mu.min()), float(mu.max())
    if hi <= lo * (1 + 1e-9):
        pad = max(lo * 0.05, 1e-6)
        return np.array([lo - pad, lo + pad], dtype=float)
    return np.geomspace(lo, hi, n_bins + 1)


def _compute_err_bin_edges(err_lo, err_hi, n_bins):
    """Linear error-% bin edges between *err_lo* and *err_hi*."""
    if not np.isfinite(err_lo) or not np.isfinite(err_hi) or err_hi <= err_lo:
        err_lo, err_hi = -5.0, 5.0
    pad = 0.02 * (err_hi - err_lo)
    return np.linspace(err_lo - pad, err_hi + pad, n_bins + 1)


def _format_mu_bin_label(mu_lo, mu_hi):
    """Representative target MU for a log-spaced bin (geometric mean, 1 s.f.)."""
    mu_center = float(np.sqrt(mu_lo * mu_hi))
    return f"{mu_center:.1g} MU"


def _gate_limits_at_mu(mu_center):
    """Return (±3% gate threshold, all gate thresholds) at *mu_center*."""
    thresholds = [
        float(_gate_threshold_pct(mu_center, gate_frac))
        for gate_frac, _, _ in GATE_LEVELS
    ]
    return max(thresholds), thresholds


def _collect_mu_bin_errors(session_data, err_col, mu_lo, mu_hi, *, last_bin, loaded_ids):
    """Pool finite dose-error values for one shared MU bin across sessions."""
    parts = []
    for sid in loaded_ids:
        data = session_data[sid]
        if err_col not in data or TARGET_COL not in data:
            continue
        mu = np.asarray(data[TARGET_COL], dtype=float)
        err = np.asarray(data[err_col], dtype=float)
        keep = (
            _mu_bin_mask(mu, mu_lo, mu_hi, last_bin=last_bin)
            & np.isfinite(err)
            & np.isfinite(mu)
            & (mu > 0)
        )
        if keep.any():
            parts.append(err[keep])
    if not parts:
        return None
    return np.concatenate(parts)


def _compute_mu_bin_err_bin_edges(
    session_data,
    err_col,
    mu_lo,
    mu_hi,
    *,
    last_bin,
    loaded_ids,
    n_bins,
):
    """Error-bin edges tailored to one MU bin's gates and spot distribution."""
    mu_center = float(np.sqrt(mu_lo * mu_hi))
    gate_max, _ = _gate_limits_at_mu(mu_center)
    pooled = _collect_mu_bin_errors(
        session_data, err_col, mu_lo, mu_hi,
        last_bin=last_bin, loaded_ids=loaded_ids,
    )
    if pooled is not None:
        err_lo = float(np.percentile(pooled, 0.5))
        err_hi = float(np.percentile(pooled, 99.5))
        err_lo = min(err_lo, -gate_max)
        err_hi = max(err_hi, gate_max)
    else:
        err_lo, err_hi = -gate_max, gate_max
    return _compute_err_bin_edges(err_lo, err_hi, n_bins)


def _mu_bin_mask(mu, mu_lo, mu_hi, *, last_bin):
    values = np.asarray(mu, dtype=float)
    if last_bin:
        return (values >= mu_lo) & (values <= mu_hi)
    return (values >= mu_lo) & (values < mu_hi)


def _draw_gate_curves(ax, mu_lo, mu_hi):
    """Draw the ±gate-threshold curves vs target MU on *ax*.

    Returns upper-curve line handles for the legend, or an empty list.
    """
    if mu_lo is None or not (mu_hi > mu_lo > 0):
        return []

    xs = np.geomspace(mu_lo, mu_hi, 400)
    handles = []
    for gate_frac, color, label in GATE_LEVELS:
        thr = _gate_threshold_pct(xs, gate_frac)
        (gate_line,) = ax.plot(xs, thr, color=color, label=label, **GATE_LINE_KW)
        ax.plot(xs, -thr, color=color, **GATE_LINE_KW)
        handles.append(gate_line)
    return handles


def _draw_mu_bin_gate_marks(ax, mu_lo, mu_hi):
    """Vertical gate lines at this MU bin's error thresholds."""
    mu_center = float(np.sqrt(mu_lo * mu_hi))
    _, thresholds = _gate_limits_at_mu(mu_center)
    for (_, color, _label), thr in zip(GATE_LEVELS, thresholds):
        ax.axvline(-thr, color=color, **HIST_GATE_KW)
        ax.axvline(thr, color=color, **HIST_GATE_KW)


def _plot_mu_bin_histogram(
    ax,
    session_data,
    err_col,
    mu_lo,
    mu_hi,
    *,
    last_bin,
    loaded_ids,
    colors,
):
    """Overlapping probability histogram for one MU bin and IC."""
    err_edges = _compute_mu_bin_err_bin_edges(
        session_data,
        err_col,
        mu_lo,
        mu_hi,
        last_bin=last_bin,
        loaded_ids=loaded_ids,
        n_bins=N_ERR_BINS,
    )
    err_lo = float(err_edges[0])
    err_hi = float(err_edges[-1])
    bar_widths = np.diff(err_edges)
    x_centers = err_edges[:-1] + 0.5 * bar_widths
    has_data = False

    for si, sid in enumerate(loaded_ids):
        data = session_data[sid]
        if err_col not in data or TARGET_COL not in data:
            continue

        mu = np.asarray(data[TARGET_COL], dtype=float)
        err = np.asarray(data[err_col], dtype=float)
        keep = (
            _mu_bin_mask(mu, mu_lo, mu_hi, last_bin=last_bin)
            & np.isfinite(err)
            & np.isfinite(mu)
            & (mu > 0)
        )
        if not keep.any():
            continue

        vals = err[keep]
        weights = np.full_like(vals, 100.0 / vals.size)
        prob, _ = np.histogram(vals, bins=err_edges, weights=weights)
        if prob.max() <= 0:
            continue

        has_data = True
        ax.bar(
            x_centers,
            prob,
            width=bar_widths,
            color=colors[si],
            alpha=0.5,
            edgecolor="none",
            align="center",
            zorder=3 + si,
        )

    _draw_mu_bin_gate_marks(ax, mu_lo, mu_hi)
    ax.axvline(x=0, **REFLINE_KW)
    ax.set_xlim(err_lo, err_hi)
    ax.grid(**GRID_KW)
    if not has_data:
        ax.text(
            0.5, 0.5, "No spots",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9, color="0.45",
        )


def _shared_scatter_ylim(session_data, err_cols, *, pad_frac=0.05):
    """Y limits spanning every finite error point across all IC scatter panels."""
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


def _process_session(session_id: str, position_key: str, base_dir: str,                     settings: ViewSettings | None = None):
    """Load a session and attach per-spot dose-error columns + target MU."""
    data = process_position_data(
        session_id,
        position_key,
        extra_spot_columns=list(DELIVERED_DOSE_COLS.values()),
        extra_input_columns=[TARGET_COL],
        base_dir=base_dir,
    )
    if data is None or TARGET_COL not in data:
        return None

    data = dict(data)
    if settings and settings.auto_calibrate:
        cols = list(DELIVERED_DOSE_COLS.values())
        if settings.cal_factors:
            data = apply_calibration_factors(data, cols, settings.cal_factors)
        else:
            data = apply_auto_calibration(data, TARGET_COL, cols)
    return add_dose_error_columns(data, target_col=TARGET_COL)


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Scatter dose error vs target MU (row 1) and per-MU-bin histograms (rows 2+)."""
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

    err_cols = [
        f"{ic}_dose_err_pct"
        for ic in ("ic1", "ic2", "ic3")
        if any(f"{ic}_dose_err_pct" in d for d in session_data.values())
    ]
    if not err_cols:
        _log.debug("No dose error columns available across all sessions")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    # Shared ranges and bin edges across every session.
    all_mu_parts = [
        t[np.isfinite(t) & (t > 0)]
        for t in (np.asarray(d[TARGET_COL], dtype=float) for d in session_data.values())
    ]
    all_mu_parts = [a for a in all_mu_parts if a.size]
    mu_lo = mu_hi = None
    mu_edges = None
    if all_mu_parts:
        all_mu_cat = np.concatenate(all_mu_parts)
        mu_lo, mu_hi = float(all_mu_cat.min()), float(all_mu_cat.max())
        mu_edges = _compute_mu_bin_edges(all_mu_cat, N_MU_BINS)

    n_cols = len(err_cols)
    n_mu = len(mu_edges) - 1 if mu_edges is not None else 0
    n_rows = 1 + n_mu
    # Scatter and all MU-bin histogram rows split vertical space 50/50.
    height_ratios = [1.0] + ([1.0 / n_mu] * n_mu if n_mu else [])
    fig_h = FIG_SIZE_2x2[1] * 2.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6, 5 * n_cols), fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    scatter_axes = axes[0]
    hist_axes = axes[1:] if n_mu else None

    # Row 1: scatter + trend (shared Y driven by all IC data).
    for col_idx in range(1, n_cols):
        scatter_axes[col_idx].sharey(scatter_axes[0])

    for col_idx, col in enumerate(err_cols):
        ax = scatter_axes[col_idx]
        ic = col.split("_", 1)[0]

        slope_labels = []
        for si, sid in enumerate(loaded_ids):
            data = session_data[sid]
            if col not in data or TARGET_COL not in data:
                continue
            prefix = trend_session_prefix(sid, n_sessions=len(loaded_ids))
            res = add_scatter_trend(
                ax,
                data[TARGET_COL],
                data[col],
                color=colors[si],
                unit="%/MU",
                prefix=prefix,
                label=f"Session {sid}",
                size=SCATTER_SIZE * 0.75,
            )
            if res is not None:
                slope_labels.append(res)

        if slope_labels:
            make_trend_legend(ax, slope_labels)

        ax.set_title(IC_TITLES.get(ic, ic.upper()))
        ax.set_xlabel(SCATTER_XLABEL)
        ax.axhline(y=0, **REFLINE_KW)
        ax.grid(**GRID_KW)
        ax.tick_params(labelleft=True)

    scatter_ylim = _shared_scatter_ylim(session_data, err_cols)
    if scatter_ylim is not None:
        scatter_axes[0].set_ylim(scatter_ylim)
    scatter_axes[0].set_ylabel(SCATTER_YLABEL)

    for ax in scatter_axes:
        gate_handles = _draw_gate_curves(ax, mu_lo, mu_hi)
        if gate_handles:
            ax.add_artist(ax.legend(handles=gate_handles, loc="lower right", fontsize=8))

    # One histogram row per shared MU bin.
    if mu_edges is not None and hist_axes is not None:
        for bin_idx in range(n_mu):
            mu_lo = float(mu_edges[bin_idx])
            mu_hi = float(mu_edges[bin_idx + 1])
            last_bin = bin_idx == n_mu - 1
            mu_label = _format_mu_bin_label(mu_lo, mu_hi)

            for col_idx, col in enumerate(err_cols):
                ax = hist_axes[bin_idx, col_idx]
                _plot_mu_bin_histogram(
                    ax,
                    session_data,
                    col,
                    mu_lo,
                    mu_hi,
                    last_bin=last_bin,
                    loaded_ids=loaded_ids,
                    colors=colors,
                )
                ax.tick_params(labelleft=True)
                if col_idx == 0:
                    ax.set_ylabel(mu_label)
                if bin_idx == n_mu - 1:
                    ax.set_xlabel(HIST_XLABEL)

    fig.align_ylabels(axes[:, 0])
    finish_view(
        fig,
        "Dose Error vs Target MU (% of prescribed dose)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
