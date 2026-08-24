"""Shared IC X/Y density/scatter plots with optional planned column and histograms."""

from __future__ import annotations

from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from . import (
    CELL_SQUARE,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    LIMIT_LINE_KW,
    POSITION_MM_TOLERANCE_LEVELS,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    finish_view,
    view_grid,
)
from .session_ic_xy import SessionIcXYData, any_session_has_plan
from .timeslice_position_error import SessionPositionErrors
from .timeslice_sigma import SessionIcSigmas

LimitMode = Literal["symmetric", "positive", "square"]
PlotStyle = Literal["contour", "scatter"]

REF_CIRCLE_RADIUS_MM = 1.0
HIST_BINS = 101
DENSITY_BINS = 80
CONTOUR_LEVEL_PERCENTILES = (40, 55, 68, 80, 90, 97)
HIST_PERCENTILE = 99.95
HEIGHT_RATIOS = (2.4, 1, 1)
HIST_TITLE_PAD = 6.0
CONTOUR_FILL_ALPHA_PER_LAYER = 0.13
CONTOUR_LINE_ALPHA = 0.85
CONTOUR_LINE_WIDTH = 0.65
_MAX_DENSITY_SAMPLES = 8000

IC_PANELS = (
    ("IC1", "ic1_x", "ic1_y"),
    ("IC2", "ic2_x", "ic2_y"),
)
PLAN_PANEL = ("Planned", "plan_x", "plan_y")


def _finite_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if positive_only:
        mask &= (x > 0) & (y > 0)
    return x[mask], y[mask]


def _finite_values(arr: np.ndarray, *, positive_only: bool = False) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    if positive_only:
        vals = vals[vals > 0]
    return vals


def _shared_symmetric_limits(*arrays: np.ndarray) -> tuple[float, float]:
    parts = [_finite_values(a) for a in arrays if a.size]
    parts = [p for p in parts if p.size]
    if not parts:
        return -1.0, 1.0
    cat = np.concatenate(parts)
    bound = float(np.percentile(np.abs(cat), HIST_PERCENTILE))
    if bound <= 0:
        bound = 1.0
    return -bound, bound


def _shared_positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    parts = [_finite_values(a, positive_only=True) for a in arrays if a.size]
    parts = [p for p in parts if p.size]
    if not parts:
        return 0.0, 5.0
    cat = np.concatenate(parts)
    hi = float(np.percentile(cat, HIST_PERCENTILE))
    if hi <= 0:
        hi = 5.0
    return 0.0, hi


def _shared_square_limits(*arrays: np.ndarray) -> tuple[float, float]:
    parts = [_finite_values(a) for a in arrays if a.size]
    parts = [p for p in parts if p.size]
    if not parts:
        return -1.0, 1.0
    cat = np.concatenate(parts)
    lo = float(np.percentile(cat, 100.0 - HIST_PERCENTILE))
    hi = float(np.percentile(cat, HIST_PERCENTILE))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return -1.0, 1.0
    mid = 0.5 * (lo + hi)
    half = max(mid - lo, hi - mid)
    if half <= 0:
        half = 1.0
    return mid - half, mid + half


def _collect_arrays(
    session_data: dict[str, SessionIcXYData | SessionPositionErrors | SessionIcSigmas],
    attrs: tuple[str, ...],
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for data in session_data.values():
        for attr in attrs:
            if attr == "plan_x":
                if isinstance(data, SessionIcXYData) and data.plan_x is not None:
                    arrays.append(data.plan_x)
            elif attr == "plan_y":
                if isinstance(data, SessionIcXYData) and data.plan_y is not None:
                    arrays.append(data.plan_y)
            else:
                arrays.append(getattr(data, attr))
    return arrays


def _draw_reference_circle(ax) -> None:
    ax.add_patch(
        Circle(
            (0, 0),
            REF_CIRCLE_RADIUS_MM,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            zorder=0,
        )
    )


def _style_xy_axis(
    ax,
    title: str,
    *,
    lim: tuple[float, float],
    limit_mode: LimitMode,
    reference_circle: bool,
) -> None:
    if reference_circle:
        _draw_reference_circle(ax)
        ax.axhline(0, **REFLINE_KW)
        ax.axvline(0, **REFLINE_KW)
    elif limit_mode == "positive":
        lo, hi = lim
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", alpha=0.35, zorder=0)
    else:
        ax.axhline(0, **REFLINE_KW)
        ax.axvline(0, **REFLINE_KW)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(**GRID_KW)


def _plot_density_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    lim: tuple[float, float],
    positive_only: bool,
) -> None:
    lo, hi = lim
    x, y = _finite_xy(x, y, positive_only=positive_only)
    if x.size > _MAX_DENSITY_SAMPLES:
        step = max(1, x.size // _MAX_DENSITY_SAMPLES)
        x = x[::step]
        y = y[::step]
    in_range = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
    x = x[in_range]
    y = y[in_range]
    if x.size < 20:
        return

    density, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=DENSITY_BINS,
        range=((lo, hi), (lo, hi)),
    )
    if not np.any(density > 0):
        return

    xc = (x_edges[:-1] + x_edges[1:]) * 0.5
    yc = (y_edges[:-1] + y_edges[1:]) * 0.5
    grid_x, grid_y = np.meshgrid(xc, yc)

    positive = density[density > 0]
    levels = np.unique(np.percentile(positive, CONTOUR_LEVEL_PERCENTILES))
    if levels.size == 0:
        return

    z = density.T
    z_max = float(z.max())
    rgba = mcolors.to_rgba(color)
    for level in levels:
        ax.contourf(
            grid_x,
            grid_y,
            z,
            levels=[level, z_max + 1.0],
            colors=[(rgba[0], rgba[1], rgba[2], CONTOUR_FILL_ALPHA_PER_LAYER)],
            antialiased=True,
            zorder=1,
        )

    ax.contour(
        grid_x,
        grid_y,
        z,
        levels=levels,
        colors=[color],
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
        zorder=2,
    )


def _plot_scatter(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    x, y = _finite_xy(x, y)
    if x.size == 0:
        return
    ax.scatter(
        x,
        y,
        color=color,
        alpha=SCATTER_ALPHA,
        marker="o",
        s=SCATTER_SIZE,
        edgecolors="none",
        zorder=3,
    )


def _plot_xy_panel(
    ax,
    session_data: dict,
    x_attr: str,
    y_attr: str,
    loaded_ids: list[str],
    colors: list,
    *,
    lim: tuple[float, float],
    plot_style: PlotStyle,
    positive_only: bool,
) -> None:
    for sid, color in zip(loaded_ids, colors):
        data = session_data[sid]
        if not hasattr(data, x_attr) or not hasattr(data, y_attr):
            continue
        x = getattr(data, x_attr)
        y = getattr(data, y_attr)
        if x is None or y is None:
            continue
        if plot_style == "scatter":
            _plot_scatter(ax, x, y, color)
        else:
            _plot_density_contours(
                ax, x, y, color, lim=lim, positive_only=positive_only,
            )


def _plot_histogram(
    ax,
    session_data: dict,
    attr: str,
    loaded_ids: list[str],
    colors: list,
    *,
    bin_edges: np.ndarray,
    title: str,
    show_ylabel: bool,
    tolerance_lines: bool,
    center_line: bool,
) -> None:
    lo, hi = float(bin_edges[0]), float(bin_edges[-1])
    positive_only = lo >= 0 and not center_line

    for sid, color in zip(loaded_ids, colors):
        data = session_data[sid]
        if not hasattr(data, attr):
            continue
        vals = _finite_values(getattr(data, attr), positive_only=positive_only)
        vals = vals[(vals >= lo) & (vals <= hi)]
        if vals.size == 0:
            continue
        weights = np.full_like(vals, 100.0 / vals.size)
        ax.hist(
            vals,
            bins=bin_edges,
            alpha=0.5,
            color=color,
            edgecolor="none",
            weights=weights,
        )

    if center_line:
        ax.axvline(0, **REFLINE_KW)
    if tolerance_lines:
        for value, color, _label in POSITION_MM_TOLERANCE_LEVELS:
            ax.axvline(value, color=color, **LIMIT_LINE_KW)
            ax.axvline(-value, color=color, **LIMIT_LINE_KW)

    ax.set_title(title, fontsize=9, pad=HIST_TITLE_PAD)
    if show_ylabel:
        ax.set_ylabel("Probability (%)")
    ax.grid(**GRID_KW)


def render_ic_xy_distribution(
    session_data: dict[str, SessionIcXYData | SessionPositionErrors | SessionIcSigmas],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
    limit_mode: LimitMode,
    plot_style: PlotStyle = "contour",
    show_plan: bool | None = None,
    reference_circle: bool = False,
    tolerance_lines: bool = False,
    x_hist_label: str = "X (mm)",
    y_hist_label: str = "Y (mm)",
    fig=None,
    show: bool = True,
) -> None:
    """Render optional planned + IC1/IC2 XY panels and X/Y histograms."""
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    positive_only = limit_mode == "positive"

    include_plan = (
        show_plan
        if show_plan is not None
        else any_session_has_plan(session_data)  # type: ignore[arg-type]
    )
    panels: list[tuple[str, str, str]] = []
    if include_plan:
        panels.append(PLAN_PANEL)
    panels.extend(IC_PANELS)

    attrs = tuple({attr for _title, x_attr, y_attr in panels for attr in (x_attr, y_attr)})
    all_arrays = _collect_arrays(session_data, attrs)

    if limit_mode == "symmetric":
        shared_lim = _shared_symmetric_limits(*all_arrays)
    elif limit_mode == "positive":
        shared_lim = _shared_positive_limits(*all_arrays)
    else:
        shared_lim = _shared_square_limits(*all_arrays)

    bin_edges = np.linspace(shared_lim[0], shared_lim[1], HIST_BINS)
    n_cols = len(panels)
    cell_h = CELL_SQUARE * sum(HEIGHT_RATIOS) / (HEIGHT_RATIOS[0] * 3)
    fig, axes = view_grid(
        3,
        n_cols,
        cell_w=CELL_SQUARE,
        cell_h=cell_h,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS},
        fig=fig,
        squeeze=False,
    )

    for col_idx, (panel_title, x_attr, y_attr) in enumerate(panels):
        ax_xy = axes[0, col_idx]
        ax_x = axes[1, col_idx]
        ax_y = axes[2, col_idx]

        _style_xy_axis(
            ax_xy,
            panel_title,
            lim=shared_lim,
            limit_mode=limit_mode,
            reference_circle=reference_circle and panel_title != "Planned",
        )
        ax_xy.set_xlim(shared_lim)
        ax_xy.set_ylim(shared_lim)
        _plot_xy_panel(
            ax_xy,
            session_data,
            x_attr,
            y_attr,
            loaded_ids,
            colors,
            lim=shared_lim,
            plot_style=plot_style,
            positive_only=positive_only,
        )

        _plot_histogram(
            ax_x,
            session_data,
            x_attr,
            loaded_ids,
            colors,
            bin_edges=bin_edges,
            title=x_hist_label,
            show_ylabel=(col_idx == 0),
            tolerance_lines=tolerance_lines,
            center_line=limit_mode == "symmetric",
        )
        _plot_histogram(
            ax_y,
            session_data,
            y_attr,
            loaded_ids,
            colors,
            bin_edges=bin_edges,
            title=y_hist_label,
            show_ylabel=(col_idx == 0),
            tolerance_lines=tolerance_lines,
            center_line=limit_mode == "symmetric",
        )

    for row in (1, 2):
        for col in range(n_cols):
            axes[row, col].set_xlim(shared_lim)

    finish_view(fig, title, loaded_ids, colors, base_dir=base_dir, show=show)
