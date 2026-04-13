"""Dose error vs prescribed target (%) for IC1, IC2, and IC3 by energy."""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.colors as mcolors

from ..common import (
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
)

POSITION_KEY_G2 = "spot_raw"
POSITION_KEY_G3 = "spot_position_raw"

DELIVERED_COLS = {
    "ic1": "ic1_total_dose_spot",
    "ic2": "ic2_total_dose_spot",
    "ic3": "r_ic3_total_dose_spot",
}
TARGET_COL = "CHARGE_REQ"


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


def _process_session(session_id: str, position_key: str, base_dir: str):
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
    target = data[TARGET_COL]
    for ic, col in DELIVERED_COLS.items():
        if col in data:
            data[f"{ic}_dose_err_pct"] = _pct_err_vs_target(data[col], target)
    return data


def run(session_ids: list[str], base_dir: str = "test_data") -> None:
    """Plot dose error (% of scan target) per IC vs beam energy."""
    if not session_ids:
        print("No sessions selected")
        return

    session_data: dict = {}
    for sid in session_ids:
        d = _process_session(sid, POSITION_KEY_G3, base_dir)
        if d is None:
            d = _process_session(sid, POSITION_KEY_G2, base_dir)
        if d is not None:
            session_data[sid] = d

    if not session_data:
        print("No valid dose / target data found for any session")
        return

    all_energies: set = set()
    for d in session_data.values():
        all_energies.update(np.unique(d["energy"]))
    energies = sorted(all_energies)

    err_cols = []
    for ic in ("ic1", "ic2", "ic3"):
        key = f"{ic}_dose_err_pct"
        if all(key in d for d in session_data.values()):
            err_cols.append(key)

    if not err_cols:
        print("No dose error columns available across all sessions")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    n_panels = len(err_cols)
    fig_w = max(12, 5.5 * n_panels)
    fig, axes = plt.subplots(
        2, n_panels, figsize=(fig_w, FIG_SIZE_2x2[1] * 2)
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        "Dose Error vs Scan Target (% of prescribed dose)",
        **SUPTITLE_KW,
    )

    titles = {
        "ic1_dose_err_pct": "IC1",
        "ic2_dose_err_pct": "IC2",
        "ic3_dose_err_pct": "IC3",
    }

    # Row 1: boxplots by energy
    for ax, col in zip(axes[0], err_cols):
        plot_boxplots_for_column(ax, session_data, col, energies, colors, width=0.3)
        _add_trend_and_mean(
            ax, session_data, col, energies, colors, position_offset=0.35
        )
        ax.set_title(f"{titles[col]} error vs energy")
        style_energy_axes(ax, energies, ylabel="Error (% of target)")
        ax.axhline(y=0, **REFLINE_KW)

    box_y_lo = min(ax.get_ylim()[0] for ax in axes[0])
    box_y_hi = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(box_y_lo, box_y_hi)

    make_session_legend(axes[0][0], loaded_ids, colors)

    # Row 2: interactive histograms linked to boxplots via SpanSelector
    _selectors = link_boxplot_to_histogram(
        list(axes[0]), list(axes[1]),
        session_data, energies, err_cols, colors, loaded_ids,
        hist_xlabels=["Error (% of target)"] * n_panels,
        hist_titles=[f"{titles[c]} error distribution" for c in err_cols],
        hist_refs=[0] * n_panels,
    )
    for ax in axes[1]:
        make_session_legend(ax, loaded_ids, colors)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.35)
    plt.show()
