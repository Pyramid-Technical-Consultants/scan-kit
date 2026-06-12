"""Shared rendering for confidence vs IC metric density contours."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from . import (
    CELL_SQUARE,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    finish_view,
    view_grid,
)
from .timeslice_confidence import SessionConfidenceCorrelations

DENSITY_BINS = 80
CONTOUR_LEVEL_PERCENTILES = (40, 55, 68, 80, 90, 97)
CURRENT_PERCENTILE = 99.95
CONFIDENCE_PERCENTILE_LO = 0.5
CONFIDENCE_PERCENTILE_HI = 99.95

CONTOUR_FILL_ALPHA_PER_LAYER = 0.13
CONTOUR_LINE_ALPHA = 0.85
CONTOUR_LINE_WIDTH = 0.65

_PANELS = (
    ("ic1_x", "IC1 X"),
    ("ic1_y", "IC1 Y"),
    ("ic2_x", "IC2 X"),
    ("ic2_y", "IC2 Y"),
)

_ROWS = (
    ("peak", "Peak IC Current (nA)"),
    ("primary", "Primary Channel (nA)"),
)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _finite_positive(values: np.ndarray) -> np.ndarray:
    vals = values[np.isfinite(values)]
    return vals[vals > 0]


def _finite_confidence(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _confidence_xlim(
    session_data: dict[str, SessionConfidenceCorrelations],
    loaded_ids: list[str],
) -> tuple[float, float] | None:
    parts = [
        _finite_confidence(getattr(session_data[sid], f"{axis}_conf"))
        for sid in loaded_ids
        for axis, _title in _PANELS
    ]
    vals = np.concatenate([a for a in parts if a.size])
    if vals.size == 0:
        return None

    lo = float(np.percentile(vals, CONFIDENCE_PERCENTILE_LO))
    hi = float(np.percentile(vals, CONFIDENCE_PERCENTILE_HI))
    if hi <= lo:
        hi = lo + 1.0
    pad = max((hi - lo) * 0.02, 0.5)
    return lo - pad, hi + pad


def _primary_attr(axis: str) -> str:
    return f"{axis.split('_')[0]}_primary"


def _current_ylim(
    session_data: dict[str, SessionConfidenceCorrelations],
    loaded_ids: list[str],
    *,
    row_key: str,
) -> tuple[float, float] | None:
    parts: list[np.ndarray] = []
    for sid in loaded_ids:
        data = session_data[sid]
        if row_key == "peak" and not data.has_peak:
            continue
        for axis, _title in _PANELS:
            if row_key == "peak":
                parts.append(getattr(data, f"{axis}_peak"))
            else:
                parts.append(getattr(data, _primary_attr(axis)))
    vals = np.concatenate([_finite_positive(a) for a in parts if a.size])
    if vals.size == 0:
        return None
    hi = float(np.percentile(vals, CURRENT_PERCENTILE))
    if hi <= 0:
        hi = 1.0
    return 0.0, hi


def _style_axis(ax, title: str, *, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(**GRID_KW)


def _plot_density_contours(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x_lo, x_hi = xlim
    y_lo, y_hi = ylim
    x, y = _finite_xy(x, y)
    in_range = (
        (x >= x_lo)
        & (x <= x_hi)
        & (y >= y_lo)
        & (y <= y_hi)
    )
    x = x[in_range]
    y = y[in_range]
    if x.size < 20:
        return

    density, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=DENSITY_BINS,
        range=(xlim, ylim),
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


def render_confidence_correlations(
    session_data: dict[str, SessionConfidenceCorrelations],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
) -> None:
    """Render confidence vs peak-current and primary-channel density contours."""
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    confidence_xlim = _confidence_xlim(session_data, loaded_ids)
    row_ylims = {
        row_key: _current_ylim(session_data, loaded_ids, row_key=row_key)
        for row_key, _label in _ROWS
    }

    fig, axes = view_grid(
        len(_ROWS),
        len(_PANELS),
        cell_w=CELL_SQUARE,
        cell_h=CELL_SQUARE * 0.85,
    )

    for row_idx, (row_key, ylabel) in enumerate(_ROWS):
        ylim = row_ylims[row_key]
        for col_idx, (axis, panel_title) in enumerate(_PANELS):
            ax = axes[row_idx, col_idx]
            _style_axis(
                ax,
                panel_title,
                xlabel="Confidence (%)" if row_idx == len(_ROWS) - 1 else "",
                ylabel=ylabel if col_idx == 0 else "",
            )
            if ylim is None or confidence_xlim is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                )
                continue

            ax.set_xlim(confidence_xlim)
            ax.set_ylim(ylim)

            for sid, color in zip(loaded_ids, colors):
                data = session_data[sid]
                if row_key == "peak":
                    if not data.has_peak:
                        continue
                    current = getattr(data, f"{axis}_peak")
                else:
                    current = getattr(data, _primary_attr(axis))
                confidence = getattr(data, f"{axis}_conf")
                _plot_density_contours(
                    ax,
                    confidence,
                    current,
                    color,
                    xlim=confidence_xlim,
                    ylim=ylim,
                )

    finish_view(fig, title, loaded_ids, colors, base_dir=base_dir)
