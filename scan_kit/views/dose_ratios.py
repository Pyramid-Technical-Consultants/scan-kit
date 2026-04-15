"""Dose ratio box plots (IC2/IC1, IC3/IC1, IC3/IC2) for G2 and G3 sessions."""

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ..common import (
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_CHARGE_REQ,
    POSITION_KEY_G3_RAW,
    ViewSettings,
    apply_auto_calibration,
    add_dose_ratio_columns,
    process_position_data,
    try_load_position_data,
    plot_boxplots_for_column,
    annotate_slopes,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    SUPTITLE_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
)

import logging

_log = logging.getLogger(__name__)

EXTRA_SPOT_G3 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE]
EXTRA_SPOT_G2 = [C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE]


def _process_ratios_session(session_id: str, position_key: str, base_dir: str,
                            settings: ViewSettings | None = None):
    """Process session and compute dose ratios."""
    extra = EXTRA_SPOT_G3 if position_key == POSITION_KEY_G3_RAW else EXTRA_SPOT_G2
    extra_input = [C_CHARGE_REQ] if settings and settings.auto_calibrate else None
    data = process_position_data(
        session_id, position_key, extra_spot_columns=extra,
        extra_input_columns=extra_input, base_dir=base_dir,
    )
    if data is None:
        return None
    if settings and settings.auto_calibrate:
        data = apply_auto_calibration(data, C_CHARGE_REQ, extra)
    return add_dose_ratio_columns(data, include_ic3=position_key == POSITION_KEY_G3_RAW)


def _trend_line_color(face_color):
    """Slightly darker line color from a patch face color name or tuple."""
    try:
        rgb = mcolors.to_rgb(face_color)
    except ValueError:
        rgb = mcolors.to_rgb("C0")
    return tuple(max(0.0, c * 0.55) for c in rgb)


def _add_median_trend_lines(
    ax,
    session_data: dict,
    column_name: str,
    energies: list,
    colors: list,
    *,
    position_offset: float = 0.35,
    zorder: float = 5,
):
    """Linear trend through per-energy medians.

    ``column_name`` values are ratio *differences in percent* (see
    ``(ratio - 1) * 100`` in :func:`_process_ratios_session`). Fitting
    ``y`` vs beam energy in MeV gives slope in **percent per MeV** (change in
    that plotted quantity per MeV).
    """
    n_sessions = len(session_data)
    # (label text, line color) for a compact in-axes legend
    slope_labels: list[tuple[str, tuple[float, float, float]]] = []

    for enum_i, (sid, data) in enumerate(session_data.items()):
        if column_name not in data:
            continue
        df = pd.DataFrame({column_name: data[column_name], "energy": data["energy"]})
        e_mev = []
        y_med = []
        for energy in energies:
            vals = df.loc[df["energy"] == energy, column_name].values
            if vals.size == 0:
                continue
            e_mev.append(float(energy))
            y_med.append(float(np.median(vals)))

        if len(e_mev) < 2:
            continue

        # y_med is in %; e_mev is in MeV -> slope is % per MeV (not a hidden /100)
        slope, intercept = np.polyfit(np.array(e_mev), np.array(y_med), 1)
        line_color = _trend_line_color(colors[enum_i])
        xs_line = [
            j + (enum_i - 0.5) * position_offset for j in range(len(energies))
        ]
        ys_line = [slope * float(energies[j]) + intercept for j in range(len(energies))]
        ax.plot(
            xs_line,
            ys_line,
            color=line_color,
            linewidth=2.0,
            linestyle="-",
            solid_capstyle="round",
            zorder=zorder,
            clip_on=True,
        )
        prefix = f"{sid}: " if n_sessions > 1 else ""
        slope_labels.append(
            (
                f"{prefix}{slope:+.4g} % per MeV",
                line_color,
            )
        )

    if slope_labels:
        annotate_slopes(ax, slope_labels)


CORR_PERCENTILE_CLIP = 100  # default percentile for correlation outlier filtering


def _plot_correlation(ax, session_data, col_x, col_y, loaded_ids, colors,
                      *, xlabel=None, ylabel=None, percentile_clip=None):
    """Scatter *col_x* vs *col_y* for each session and annotate with R²."""
    pclip = percentile_clip if percentile_clip is not None else CORR_PERCENTILE_CLIP

    # Collect all finite pairs across sessions to compute shared clip bounds
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
        all_x = np.concatenate([xy[0] for xy in all_xy])
        all_y = np.concatenate([xy[1] for xy in all_xy])
        v_lo = min(all_x.min(), all_y.min())
        v_hi = max(all_x.max(), all_y.max())
        pad = (v_hi - v_lo) * 0.03
        ax.plot([v_lo - pad, v_hi + pad], [v_lo - pad, v_hi + pad], **REFLINE_KW)
        ax.set_aspect("equal", adjustable="datalim")

    if labels:
        annotate_slopes(ax, labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(visible=True, alpha=0.3)


CORR_PAIRS = [
    (C_IC1_TOTAL_DOSE, C_IC2_TOTAL_DOSE, "IC1 Dose", "IC2 Dose"),
    (C_IC1_TOTAL_DOSE, C_IC3_TOTAL_DOSE, "IC1 Dose", "IC3 Dose"),
    (C_IC2_TOTAL_DOSE, C_IC3_TOTAL_DOSE, "IC2 Dose", "IC3 Dose"),
]


def run(session_ids: list[str], base_dir: str = "test_data",
        *, settings: ViewSettings | None = None) -> None:
    """Run dose ratios analysis and show matplotlib window."""
    if not session_ids:
        _log.debug("No sessions selected")
        return

    def _loader(sid, pkey, bdir):
        return _process_ratios_session(sid, pkey, bdir, settings=settings)

    session_data = {}
    for sid in session_ids:
        data = try_load_position_data(sid, base_dir, _loader)
        if data is not None:
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid data found for any session")
        return

    fig, axes = plt.subplots(
        3, 2, figsize=(FIG_SIZE_2x2[0] + 4, FIG_SIZE_2x2[1] * 1.4),
        gridspec_kw={"width_ratios": [4, 1]},
    )
    fig.suptitle("Dose Ratios vs Energy", **SUPTITLE_KW)

    box_axes = [axes[0, 0], axes[1, 0], axes[2, 0]]
    corr_axes = [axes[0, 1], axes[1, 1], axes[2, 1]]

    box_axes[1].sharex(box_axes[0])
    box_axes[2].sharex(box_axes[0])

    all_energies = set()
    for data in session_data.values():
        all_energies.update(data["energy"].unique())
    energies = sorted(all_energies)

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    session_data_g3 = {k: v for k, v in session_data.items() if "ic31_ratio" in v}
    colors_g3 = [colors[loaded_ids.index(sid)] for sid in session_data_g3]

    plot_boxplots_for_column(
        box_axes[0], session_data, "ic21_ratio", energies, colors, width=0.3
    )
    _add_median_trend_lines(
        box_axes[0], session_data, "ic21_ratio", energies, colors, position_offset=0.35
    )
    style_energy_axes(box_axes[0], energies, ylabel="IC2/IC1 Ratio Difference (%)")

    if session_data_g3:
        plot_boxplots_for_column(
            box_axes[1], session_data_g3, "ic31_ratio", energies, colors_g3, width=0.3
        )
        _add_median_trend_lines(
            box_axes[1], session_data_g3, "ic31_ratio", energies, colors_g3, position_offset=0.35
        )
    style_energy_axes(box_axes[1], energies, ylabel="IC3/IC1 Ratio Difference (%)")

    if session_data_g3:
        plot_boxplots_for_column(
            box_axes[2], session_data_g3, "ic32_ratio", energies, colors_g3, width=0.3
        )
        _add_median_trend_lines(
            box_axes[2], session_data_g3, "ic32_ratio", energies, colors_g3, position_offset=0.35
        )
    style_energy_axes(box_axes[2], energies, ylabel="IC3/IC2 Ratio Difference (%)")

    y_lo = min(ax.get_ylim()[0] for ax in box_axes)
    y_hi = max(ax.get_ylim()[1] for ax in box_axes)
    for ax in box_axes:
        ax.set_ylim(y_lo, y_hi)

    # Correlation scatter plots (right column)
    corr_data = [
        (session_data, loaded_ids, colors),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
        (session_data_g3, list(session_data_g3.keys()), colors_g3),
    ]
    for row, (col_x, col_y, xlabel, ylabel) in enumerate(CORR_PAIRS):
        sdata, sids, scols = corr_data[row]
        if sdata:
            _plot_correlation(corr_axes[row], sdata, col_x, col_y, sids, scols,
                              xlabel=xlabel, ylabel=ylabel, percentile_clip=99.9)
        else:
            corr_axes[row].set_visible(False)

    make_session_legend(box_axes[0], loaded_ids, colors)

    plt.tight_layout()
    plt.show()
