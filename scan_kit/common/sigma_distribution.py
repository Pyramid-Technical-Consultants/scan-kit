"""Shared rendering for IC sigma density contours and X/Y histograms."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import (
    CELL_SQUARE,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    finish_view,
    view_grid,
)
from .timeslice_sigma import SessionIcSigmas

HIST_BINS = 101
DENSITY_BINS = 80
CONTOUR_LEVEL_PERCENTILES = (40, 55, 68, 80, 90, 97)
HIST_PERCENTILE = 99.95

HEIGHT_RATIOS = (2.4, 1, 1)
HIST_TITLE_PAD = 6.0
CONTOUR_FILL_ALPHA_PER_LAYER = 0.13
CONTOUR_LINE_ALPHA = 0.85
CONTOUR_LINE_WIDTH = 0.65

IC_PANELS = (
    ("IC1", "ic1_x", "ic1_y"),
    ("IC2", "ic2_x", "ic2_y"),
)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[mask], y[mask]


def _finite_values(arr: np.ndarray) -> np.ndarray:
    vals = arr[np.isfinite(arr)]
    return vals[vals > 0]


def _shared_positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    """Shared [0, hi] limits from the *HIST_PERCENTILE* of sigma values."""
    parts = [_finite_values(a) for a in arrays if a.size]
    parts = [p for p in parts if p.size]
    if not parts:
        return 0.0, 5.0
    cat = np.concatenate(parts)
    hi = float(np.percentile(cat, HIST_PERCENTILE))
    if hi <= 0:
        hi = 5.0
    return 0.0, hi


def _style_density_axis(ax, title: str, *, lim: tuple[float, float]) -> None:
    lo, hi = lim
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", alpha=0.35, zorder=0)
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
) -> None:
    lo, hi = lim
    x, y = _finite_xy(x, y)
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


def _plot_sigma_histogram(
    ax,
    session_data: dict[str, SessionIcSigmas],
    attr: str,
    loaded_ids: list[str],
    colors: list,
    *,
    bin_edges: np.ndarray,
    title: str,
    show_ylabel: bool,
) -> None:
    lo, hi = float(bin_edges[0]), float(bin_edges[-1])

    for sid, color in zip(loaded_ids, colors):
        vals = _finite_values(getattr(session_data[sid], attr))
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

    ax.axvline(0, **REFLINE_KW)
    ax.set_title(title, fontsize=9, pad=HIST_TITLE_PAD)
    if show_ylabel:
        ax.set_ylabel("Probability (%)")
    ax.grid(**GRID_KW)


def render_sigma_distribution(
    session_data: dict[str, SessionIcSigmas],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
) -> None:
    """Render IC1/IC2 sigma density contours and X/Y histograms."""
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    all_arrays: list[np.ndarray] = []
    for data in session_data.values():
        all_arrays.extend((data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y))

    shared_lim = _shared_positive_limits(*all_arrays)
    bin_edges = np.linspace(shared_lim[0], shared_lim[1], HIST_BINS)

    cell_h = CELL_SQUARE * sum(HEIGHT_RATIOS) / (HEIGHT_RATIOS[0] * 3)
    fig, axes = view_grid(
        3,
        2,
        cell_w=CELL_SQUARE,
        cell_h=cell_h,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS},
    )

    hist_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]

    for col_idx, (ic_title, x_attr, y_attr) in enumerate(IC_PANELS):
        ax_density = axes[0, col_idx]
        ax_x = axes[1, col_idx]
        ax_y = axes[2, col_idx]

        _style_density_axis(ax_density, ic_title, lim=shared_lim)
        ax_density.set_xlim(shared_lim)
        ax_density.set_ylim(shared_lim)

        for sid, color in zip(loaded_ids, colors):
            data = session_data[sid]
            _plot_density_contours(
                ax_density,
                getattr(data, x_attr),
                getattr(data, y_attr),
                color,
                lim=shared_lim,
            )

        _plot_sigma_histogram(
            ax_x,
            session_data,
            x_attr,
            loaded_ids,
            colors,
            bin_edges=bin_edges,
            title="X Sigma (mm)",
            show_ylabel=(col_idx == 0),
        )
        _plot_sigma_histogram(
            ax_y,
            session_data,
            y_attr,
            loaded_ids,
            colors,
            bin_edges=bin_edges,
            title="Y Sigma (mm)",
            show_ylabel=(col_idx == 0),
        )

    for ax in hist_axes:
        ax.set_xlim(shared_lim)

    finish_view(fig, title, loaded_ids, colors, base_dir=base_dir)
