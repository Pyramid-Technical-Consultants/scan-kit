"""Interlock threshold overlays for binned summary plots."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .devices_xml import IC_DEVICE_TO_SIG_KEY, load_session_devices_config
from .plotting import LIMIT_LINE_KW, POSITION_MM_TOLERANCE_LEVELS

Y_POSITION_ERROR = "position_error"
Y_DOSE_ERROR = "dose_error"
Y_SIGMA = "sigma"
Y_SIGMA_ERROR = "sigma_error"
X_ENERGY = "energy"
X_TARGET_MU = "target_mu"

# Delivered-dose safety gate: fixed MU floor + fraction of target MU.
GATE_ABS_MU = 0.002
DOSE_GATE_LEVELS: tuple[tuple[float, str, str], ...] = (
    (0.01, "green", "±(0.002 MU + 1%)"),
    (0.02, "orange", "±(0.002 MU + 2%)"),
    (0.03, "red", "±(0.002 MU + 3%)"),
)

# Symmetric ± bands around devices.xml expected sigma (mm).
SIGMA_TOLERANCE_FRACTIONS: tuple[tuple[float, str, float, str], ...] = (
    (0.20, ":", 0.55, "±20% of expected"),
    (0.40, "--", 0.65, "±40% of expected"),
)

_SIGMA_SERIES_TO_DEVICE = {
    **{sig_key: device for device, sig_key in IC_DEVICE_TO_SIG_KEY.items()},
    **{
        f"{sig_key}_err": device
        for device, sig_key in IC_DEVICE_TO_SIG_KEY.items()
    },
}

def interlock_overlay_available(
    y_group: str,
    x_param: str,
    *,
    glyph: str | None = None,
) -> bool:
    """Return whether interlock overlays apply to the current view selection."""
    if y_group == Y_POSITION_ERROR:
        return True
    if y_group == Y_DOSE_ERROR:
        return x_param == X_TARGET_MU
    if y_group in (Y_SIGMA, Y_SIGMA_ERROR):
        if x_param != X_ENERGY:
            return False
        if glyph in ("scatter", "contour"):
            return False
        return True
    return False


def dose_gate_threshold_pct(mu, gate_frac: float):
    """Gate tolerance as % of target MU: ``(0.002 + frac*mu)/mu*100``."""
    mu_arr = np.asarray(mu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (GATE_ABS_MU + gate_frac * mu_arr) / mu_arr * 100.0
    pct = np.where(mu_arr > 0, pct, np.nan)
    if np.ndim(mu_arr) == 0:
        return float(pct)
    return pct


def histogram_tolerance_levels(y_group: str) -> list[tuple[float, str, str]] | None:
    """Fixed histogram ±levels for metrics that do not vary with energy."""
    if y_group == Y_POSITION_ERROR:
        return list(POSITION_MM_TOLERANCE_LEVELS)
    return None


def load_expected_sigmas_by_session(
    session_ids: Sequence[str],
    energies: Sequence[float],
    base_dir: str,
    *,
    series_key: str,
) -> dict[str, dict[float, float]]:
    """Per-session expected sigma (mm) keyed by energy category."""
    device = _SIGMA_SERIES_TO_DEVICE.get(series_key)
    if device is None:
        return {}
    expected: dict[str, dict[float, float]] = {}
    for sid in session_ids:
        config = load_session_devices_config(sid, base_dir)
        if config is None:
            continue
        sig_key = IC_DEVICE_TO_SIG_KEY[device]
        by_key = config.expected_sigmas_by_key(energies, keys=(sig_key,))
        per_energy = by_key.get(sig_key)
        if per_energy:
            expected[sid] = per_energy
    return expected


def sigma_tolerance_band(expected: float, frac: float) -> tuple[float, float]:
    """Return lower/upper sigma (mm) for a symmetric ±*frac* band."""
    center = float(expected)
    return max(0.0, center * (1.0 - frac)), center * (1.0 + frac)


def _curve_points(
    expected_by_energy: dict[float, float],
    categories: Sequence[float],
    *,
    symmetric_error: bool,
    frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(len(categories), dtype=float)
    lower_pts: list[float] = []
    upper_pts: list[float] = []
    for energy in categories:
        sigma = expected_by_energy.get(float(energy))
        if sigma is None or not np.isfinite(sigma):
            lower_pts.append(np.nan)
            upper_pts.append(np.nan)
            continue
        if symmetric_error:
            margin = float(sigma) * frac
            lower_pts.append(-margin)
            upper_pts.append(margin)
        else:
            lower, upper = sigma_tolerance_band(float(sigma), frac)
            lower_pts.append(lower)
            upper_pts.append(upper)
    return xs, np.asarray(lower_pts, dtype=float), np.asarray(upper_pts, dtype=float)


def plot_config_sigma_tolerance_curves(
    ax: Axes,
    expected_by_energy: dict[float, float],
    categories: Sequence[float],
    color,
    *,
    symmetric_error: bool = False,
) -> None:
    """Draw ±20% / ±40% curves across categorical energy bins."""
    for frac, linestyle, alpha, _label in SIGMA_TOLERANCE_FRACTIONS:
        xs, lower, upper = _curve_points(
            expected_by_energy,
            categories,
            symmetric_error=symmetric_error,
            frac=frac,
        )
        for ys in (lower, upper):
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                alpha=alpha,
                zorder=7,
            )


def plot_dose_gate_curves_binned(
    ax: Axes,
    categories: Sequence[float],
    *,
    add_legend: bool = False,
) -> None:
    """Draw ±dose-gate curves across categorical target-MU bins."""
    xs = np.arange(len(categories), dtype=float)
    for gate_frac, color, _label in DOSE_GATE_LEVELS:
        thresholds = [
            float(dose_gate_threshold_pct(float(mu), gate_frac))
            for mu in categories
        ]
        upper = np.asarray(thresholds, dtype=float)
        for ys in (upper, -upper):
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.85,
                zorder=7,
            )
    if add_legend:
        add_dose_gate_legend(ax)


def plot_dose_gate_curves_linear(
    ax: Axes,
    mu_lo: float,
    mu_hi: float,
    *,
    add_legend: bool = False,
) -> None:
    """Draw ±dose-gate curves vs raw target MU on a linear x-axis."""
    if not (np.isfinite(mu_lo) and np.isfinite(mu_hi) and mu_hi > mu_lo > 0):
        return
    xs = np.geomspace(mu_lo, mu_hi, 400)
    for gate_frac, color, _label in DOSE_GATE_LEVELS:
        thr = dose_gate_threshold_pct(xs, gate_frac)
        ax.plot(xs, thr, color=color, linestyle="--", linewidth=1.3, alpha=0.85, zorder=7)
        ax.plot(xs, -thr, color=color, linestyle="--", linewidth=1.3, alpha=0.85, zorder=7)
    if add_legend:
        add_dose_gate_legend(ax)


def add_dose_gate_legend(ax: Axes) -> None:
    handles = [
        Line2D([0], [0], color=color, linestyle="--", linewidth=1.3, alpha=0.85)
        for _frac, color, _label in DOSE_GATE_LEVELS
    ]
    labels = [label for _frac, _color, label in DOSE_GATE_LEVELS]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title="Dose gate",
        title_fontsize=8,
    )


def add_sigma_tolerance_legend(ax: Axes) -> None:
    """Legend for tolerance line styles (first panel only)."""
    handles = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=linestyle,
            linewidth=1.3,
            alpha=alpha,
        )
        for _frac, linestyle, alpha, _label in SIGMA_TOLERANCE_FRACTIONS
    ]
    labels = [label for *_rest, label in SIGMA_TOLERANCE_FRACTIONS]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        title="Interlock band",
        title_fontsize=8,
    )


def plot_position_interlock_lines(ax: Axes, *, legend: bool = False) -> None:
    """Draw fixed ±mm position interlock lines."""
    from .plotting import draw_symmetric_limit_lines

    draw_symmetric_limit_lines(
        ax,
        POSITION_MM_TOLERANCE_LEVELS,
        orientation="horizontal",
        legend=legend,
    )


def apply_binned_interlock_overlays(
    ax: Axes,
    *,
    y_group: str,
    series_key: str,
    categories: Sequence[float],
    session_ids: Sequence[str],
    colors: Sequence,
    base_dir: str,
    add_sigma_legend: bool = False,
) -> None:
    """Draw interlock overlays on one binned-summary main axis."""
    if y_group == Y_POSITION_ERROR:
        plot_position_interlock_lines(ax, legend=add_sigma_legend)
        return

    if y_group == Y_DOSE_ERROR:
        plot_dose_gate_curves_binned(ax, categories, add_legend=add_sigma_legend)
        return

    if y_group not in (Y_SIGMA, Y_SIGMA_ERROR):
        return

    expected_by_session = load_expected_sigmas_by_session(
        session_ids,
        categories,
        base_dir,
        series_key=series_key,
    )
    if not expected_by_session:
        return

    symmetric_error = y_group == Y_SIGMA_ERROR
    for session_idx, sid in enumerate(session_ids):
        per_energy = expected_by_session.get(sid)
        if not per_energy:
            continue
        plot_config_sigma_tolerance_curves(
            ax,
            per_energy,
            categories,
            colors[session_idx],
            symmetric_error=symmetric_error,
        )

    if add_sigma_legend:
        add_sigma_tolerance_legend(ax)


def apply_linear_interlock_overlays(
    ax: Axes,
    *,
    y_group: str,
    x_param: str,
    x_values: np.ndarray,
    add_legend: bool = False,
) -> None:
    """Draw interlock overlays on scatter/contour axes with a linear x-axis."""
    if y_group != Y_DOSE_ERROR or x_param != X_TARGET_MU:
        return
    finite = x_values[np.isfinite(x_values) & (x_values > 0)]
    if finite.size == 0:
        return
    plot_dose_gate_curves_linear(
        ax,
        float(np.min(finite)),
        float(np.max(finite)),
        add_legend=add_legend,
    )
