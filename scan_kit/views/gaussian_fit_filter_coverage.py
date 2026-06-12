"""Gaussian fit filter coverage — spot retention vs confidence and peak-current thresholds."""

from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..common import (
    CELL_SQUARE,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    finish_view,
    view_grid,
)
from ..common.timeslice_gaussian_fit_filter_coverage import (
    GaussianFitFilterSweep,
    SessionGaussianFitFilterCoverage,
    compute_session_gaussian_fit_filter_coverage,
)

_log = logging.getLogger(__name__)

VIEW_TITLE = "Gaussian Fit Filter Coverage"

_PANELS = (
    ("ic1", "IC1"),
    ("ic2", "IC2"),
)

_ROWS = (
    ("confidence", "Confidence threshold (%)", "{:.2f}%", 10.0, 100.0),
    ("peak", "Peak IC current threshold (nA)", "{:.2f} nA", 0.10, None),
)

Y_PAD = 4.0


def _panel_xlim(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    ic_key: str,
    *,
    row_key: str,
) -> tuple[float, float]:
    _, _, _, x_pad, x_hi_fixed = next(r for r in _ROWS if r[0] == row_key)
    drop_points: list[float] = []
    hi_points: list[float] = []

    for sid in loaded_ids:
        session = filter_data.get(sid)
        if session is None:
            continue
        coverage = _row_coverage(session, row_key)
        if coverage is None:
            continue
        ic_cov = coverage.ics.get(ic_key)
        if ic_cov is None:
            continue
        hi_points.append(float(coverage.thresholds[-1]))
        breakpoint = ic_cov.full_coverage_breakpoint
        if breakpoint is not None and np.isfinite(breakpoint):
            drop_points.append(breakpoint)

    if not drop_points:
        x_hi = x_hi_fixed if x_hi_fixed is not None else (max(hi_points) if hi_points else 1.0)
        return 0.0, x_hi

    x_lo_anchor = min(drop_points)
    if row_key == "confidence":
        x_lo = max(0.0, x_lo_anchor - x_pad)
        x_hi = x_hi_fixed if x_hi_fixed is not None else 100.0
    else:
        x_lo = max(0.0, x_lo_anchor * (1.0 - x_pad))
        x_hi = max(hi_points) if hi_points else x_lo_anchor

    return x_lo, x_hi


def _panel_ylim(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    ic_key: str,
    *,
    row_key: str,
    xlim: tuple[float, float],
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    x_lo, x_hi = xlim
    for sid in loaded_ids:
        session = filter_data.get(sid)
        if session is None:
            continue
        coverage = _row_coverage(session, row_key)
        if coverage is None:
            continue
        ic_cov = coverage.ics.get(ic_key)
        if ic_cov is None:
            continue
        in_view = (coverage.thresholds >= x_lo) & (coverage.thresholds <= x_hi)
        if np.any(in_view):
            values.append(ic_cov.coverage_pct[in_view])

    if not values:
        return 0.0, 100.0 + Y_PAD

    vals = np.concatenate(values)
    ymin = max(0.0, float(np.min(vals)) - Y_PAD)
    ymax = min(100.0 + Y_PAD, float(np.max(vals)) + Y_PAD)
    if ymax <= ymin:
        ymax = min(100.0 + Y_PAD, ymin + Y_PAD)
    return ymin, ymax


def _row_coverage(
    session: SessionGaussianFitFilterCoverage,
    row_key: str,
) -> GaussianFitFilterSweep | None:
    if row_key == "confidence":
        return session.confidence
    return session.peak


def _plot_filter_row(
    axes_row,
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    colors: list,
    *,
    row_key: str,
    xlabel: str,
    breakpoint_fmt: str,
) -> None:
    for col_idx, (ax, (ic_key, panel_title)) in enumerate(zip(axes_row, _PANELS)):
        ax.set_title(panel_title)
        coverage_available = any(
            _row_coverage(filter_data[sid], row_key) is not None
            for sid in loaded_ids
            if sid in filter_data
        )
        if not coverage_available:
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

        xlim = _panel_xlim(filter_data, loaded_ids, ic_key, row_key=row_key)
        ax.set_xlim(xlim)
        ax.set_ylim(
            _panel_ylim(
                filter_data,
                loaded_ids,
                ic_key,
                row_key=row_key,
                xlim=xlim,
            )
        )
        if col_idx == 0:
            ax.set_ylabel("% spots with valid X and Y")
        ax.set_xlabel(xlabel)
        ax.grid(**GRID_KW)

        legend_handles: list = []
        legend_labels: list[str] = []
        for sid, color in zip(loaded_ids, colors):
            session = filter_data.get(sid)
            if session is None:
                continue
            coverage = _row_coverage(session, row_key)
            if coverage is None:
                continue
            ic_cov = coverage.ics.get(ic_key)
            if ic_cov is None:
                continue

            ax.plot(
                coverage.thresholds,
                ic_cov.coverage_pct,
                color=color,
                linewidth=1.5,
            )

            breakpoint = ic_cov.full_coverage_breakpoint
            if breakpoint is None or not np.isfinite(breakpoint):
                continue
            ax.axvline(breakpoint, color=color, linestyle=":", linewidth=1.5, alpha=0.9)
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle=":", linewidth=1.5)
            )
            legend_labels.append(
                f"100% breakpoint ({breakpoint_fmt.format(breakpoint)})"
            )

        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=7, loc="lower left")


def _plot_coverage(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
) -> None:
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    fig, axes = view_grid(
        len(_ROWS),
        len(_PANELS),
        cell_w=CELL_SQUARE,
        cell_h=CELL_SQUARE * 0.75,
    )

    for row_idx, (row_key, xlabel, breakpoint_fmt, _x_pad, _x_hi) in enumerate(_ROWS):
        _plot_filter_row(
            axes[row_idx],
            filter_data,
            loaded_ids,
            colors,
            row_key=row_key,
            xlabel=xlabel,
            breakpoint_fmt=breakpoint_fmt,
        )

    finish_view(fig, title, loaded_ids, colors, base_dir=base_dir)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot spot coverage versus Gaussian fit filter thresholds."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    filter_data: dict[str, SessionGaussianFitFilterCoverage] = {}
    for sid in session_ids:
        coverage = compute_session_gaussian_fit_filter_coverage(
            sid, base_dir, bg_subtract=bg_subtract
        )
        if coverage is not None:
            filter_data[sid] = coverage

    if not filter_data:
        print("No valid Gaussian fit filter coverage data found for any session")
        return

    _plot_coverage(
        filter_data,
        list(filter_data.keys()),
        title=VIEW_TITLE,
        base_dir=base_dir,
    )
