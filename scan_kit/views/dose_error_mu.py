"""Per-spot dose error (% of target) vs target MU — scatter (IC1, IC2, IC3).

Same per-spot error as :mod:`dose_error_energy`, but organized along the spot's
target MU (x-axis in MU) instead of beam energy. Y stays in % of target.
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
    annotate_slopes,
    make_session_legend,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    REFLINE_KW,
    GRID_KW,
    DELIVERED_DOSE_COLS,
)

_log = logging.getLogger(__name__)

TARGET_COL = C_CHARGE_REQ
IC_TITLES = {"ic1": "IC1", "ic2": "IC2", "ic3": "IC3"}

# Safety-gate tolerance on delivered dose: fixed floor + fraction of target MU.
GATE_ABS_MU = 0.002  # absolute MU floor
GATE_FRAC = 0.02  # fraction of target MU (2%)

GATE_LINE_KW = dict(color="red", linestyle="--", linewidth=1.3, alpha=0.85, zorder=6)


def _gate_threshold_pct(mu):
    """Gate tolerance as a percentage of target MU: ``(0.002 + 0.02*mu)/mu*100``.

    Asymptotes to ``GATE_FRAC * 100`` (2%) for large MU and flares up as MU -> 0.
    """
    mu = np.asarray(mu, dtype=float)
    return (GATE_ABS_MU + GATE_FRAC * mu) / mu * 100.0


def _draw_gate_curves(ax, mu_lo, mu_hi):
    """Draw the ±gate-threshold curves vs target MU, keeping the data y-limits.

    Returns the upper-curve line handle (for a legend), or ``None``.
    """
    if mu_lo is None or not (mu_hi > mu_lo > 0):
        return None

    data_ylim = ax.get_ylim()
    xs = np.geomspace(mu_lo, mu_hi, 400)
    thr = _gate_threshold_pct(xs)
    (gate_line,) = ax.plot(xs, thr, label="Gate \u00b1(0.002 MU + 2%)", **GATE_LINE_KW)
    ax.plot(xs, -thr, **GATE_LINE_KW)

    # Frame on the data, but make sure the ~2% asymptote is visible; let the
    # small-MU flare clip rather than dictate the scale.
    gate_ref = float(_gate_threshold_pct(mu_hi)) * 1.3
    ax.set_ylim(min(data_ylim[0], -gate_ref), max(data_ylim[1], gate_ref))
    return gate_line


def _process_session(session_id: str, position_key: str, base_dir: str,
                     settings: ViewSettings | None = None):
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
    """Scatter per-spot dose error (% of target) vs target MU, per IC."""
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

    # Shared gate-threshold x-range: positive target MU across all sessions.
    mu_lo = mu_hi = None
    all_mu = [
        t[np.isfinite(t) & (t > 0)]
        for t in (np.asarray(d[TARGET_COL], dtype=float) for d in session_data.values())
    ]
    all_mu = [a for a in all_mu if a.size]
    if all_mu:
        cat = np.concatenate(all_mu)
        mu_lo, mu_hi = float(cat.min()), float(cat.max())

    n_cols = len(err_cols)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(max(6, 5 * n_cols), FIG_SIZE_2x2[1]), squeeze=False,
    )
    axes = axes[0]
    fig.suptitle("Dose Error vs Target MU (% of prescribed dose)", **SUPTITLE_KW)

    for col_idx, col in enumerate(err_cols):
        ax = axes[col_idx]
        ic = col.split("_", 1)[0]

        slope_labels = []
        for si, sid in enumerate(loaded_ids):
            data = session_data[sid]
            if col not in data or TARGET_COL not in data:
                continue
            prefix = f"{sid}: " if len(loaded_ids) > 1 else ""
            res = add_scatter_trend(
                ax,
                data[TARGET_COL],
                data[col],
                color=colors[si],
                unit="%/MU",
                prefix=prefix,
                label=f"Session {sid}",
            )
            if res is not None:
                slope_labels.append(res)

        if slope_labels:
            annotate_slopes(ax, slope_labels)

        ax.set_title(IC_TITLES.get(ic, ic.upper()))
        ax.set_xlabel("Target (MU)")
        ax.set_ylabel(f"{IC_TITLES.get(ic, ic.upper())} Error (% of target)")
        ax.axhline(y=0, **REFLINE_KW)
        ax.grid(**GRID_KW)

        gate_line = _draw_gate_curves(ax, mu_lo, mu_hi)
        if gate_line is not None:
            ax.add_artist(ax.legend(handles=[gate_line], loc="lower right", fontsize=8))

    make_session_legend(axes[0], loaded_ids, colors)

    plt.tight_layout()
    plt.show()
