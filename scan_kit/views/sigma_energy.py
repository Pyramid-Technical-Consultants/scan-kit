"""IC1/IC2 sigma X/Y (mm) vs beam energy — violin plots.

Sigma spot columns are resolved from ``spot_sigma`` / ``spot_sigma_raw`` and
scaled ×2 to mm (same convention as the former box-plot view). Tolerance bands
are drawn symmetrically around the expected sigma from ``devices.xml`` (±20%
and ±40%).

Layout: two rows (IC1, IC2) with row labels on the left; X and Y in columns.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..common import (
    load_session_raw,
    load_session_devices_config,
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
COLUMN_TITLES = ("X Sigma (mm)", "Y Sigma (mm)")
ROW_LABELS = ("IC1", "IC2")

_SIG_KEY_VARIANTS = ("spot_sigma_raw", "spot_sigma")

IC_PANELS = (
    ("ic1", "IC1", ("ic1_sig_x", "ic1_sig_y")),
    ("ic2", "IC2", ("ic2_sig_x", "ic2_sig_y")),
)

# Symmetric ± bands around devices.xml expected sigma (mm).
SIGMA_TOLERANCE_FRACTIONS = (
    (0.20, ":", 0.55, "±20% of expected"),
    (0.40, "--", 0.65, "±40% of expected"),
)


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


def _sigma_tolerance_band(expected: float, frac: float) -> tuple[float, float]:
    """Return lower/upper sigma (mm) for a symmetric ±*frac* band around *expected*."""
    center = float(expected)
    return max(0.0, center * (1.0 - frac)), center * (1.0 + frac)


def _plot_sigma_tolerance_lines(
    ax,
    expected_by_energy: dict[float, float],
    energies,
    color,
):
    """Draw ±20% and ±40% limits as continuous curves across energies."""
    xs = np.arange(len(energies), dtype=float)
    for frac, linestyle, alpha, _label in SIGMA_TOLERANCE_FRACTIONS:
        lower_pts: list[float] = []
        upper_pts: list[float] = []
        for energy in energies:
            sigma = expected_by_energy.get(float(energy))
            if sigma is None or not np.isfinite(sigma):
                lower_pts.append(np.nan)
                upper_pts.append(np.nan)
                continue
            lower, upper = _sigma_tolerance_band(sigma, frac)
            lower_pts.append(lower)
            upper_pts.append(upper)
        for ys in (lower_pts, upper_pts):
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                alpha=alpha,
                zorder=7,
            )


def _add_sigma_tolerance_legend(ax) -> None:
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
        title="Sigma tolerance",
        title_fontsize=8,
    )


def _load_expected_sigmas(session_ids, energies, base_dir: str) -> dict[str, dict[str, dict[float, float]]]:
    """Per-session expected sigma (mm) keyed by view column name and energy."""
    expected: dict[str, dict[str, dict[float, float]]] = {}
    for sid in session_ids:
        config = load_session_devices_config(sid, base_dir)
        if config is None:
            continue
        by_key = config.expected_sigmas_by_key(energies)
        if by_key:
            expected[sid] = by_key
    return expected


def _shared_panel_ylim(session_data, value_cols, expected_sigmas=None, *, pad_frac=0.05):
    """Y limits spanning measured and expected sigma values across violin panels."""
    parts = []
    for col in value_cols:
        for data in session_data.values():
            if col not in data:
                continue
            vals = np.asarray(data[col], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                parts.append(finite)
        if expected_sigmas:
            for by_key in expected_sigmas.values():
                per_energy = by_key.get(col)
                if not per_energy:
                    continue
                expected_vals = np.asarray(list(per_energy.values()), dtype=float)
                finite = expected_vals[np.isfinite(expected_vals)]
                if not finite.size:
                    continue
                for frac, *_rest in SIGMA_TOLERANCE_FRACTIONS:
                    lower = np.maximum(0.0, finite * (1.0 - frac))
                    upper = finite * (1.0 + frac)
                    parts.append(lower)
                    parts.append(upper)
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
    expected_sigmas = _load_expected_sigmas(loaded_ids, energies, base_dir)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_SIZE_1x2[0], FIG_SIZE_1x2[1] * 2),
        squeeze=False,
    )

    master = axes[0, 0]
    _link_axes_keep_tick_labels(axes.flat, master)

    value_cols: list[str] = []
    tolerance_legend_added = False

    for row_idx, (_ic, _ic_title, (x_col, y_col)) in enumerate(IC_PANELS):
        for col_idx, value_col in enumerate((x_col, y_col)):
            ax = axes[row_idx, col_idx]
            plot_violins_for_column(ax, session_data, value_col, energies, colors)
            for session_idx, sid in enumerate(loaded_ids):
                session_expected = expected_sigmas.get(sid, {}).get(value_col)
                if session_expected:
                    _plot_sigma_tolerance_lines(
                        ax,
                        session_expected,
                        energies,
                        colors[session_idx],
                    )

            style_energy_axes(ax, energies, ylabel=None)
            ax.set_xlabel("")
            value_cols.append(value_col)

            if not tolerance_legend_added and expected_sigmas:
                _add_sigma_tolerance_legend(ax)
                tolerance_legend_added = True

    ylim = _shared_panel_ylim(session_data, value_cols, expected_sigmas)
    if ylim is not None:
        master.set_ylim(ylim)

    apply_shared_block_labels(
        axes,
        column_titles=COLUMN_TITLES,
        row_ylabels=ROW_LABELS,
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
