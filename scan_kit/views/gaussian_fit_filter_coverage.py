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
    PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES,
    SPOT_ERROR_CODE_AXIS_LABELS,
    SessionGaussianFitFilterCoverage,
    compute_session_gaussian_fit_filter_coverage,
)

_log = logging.getLogger(__name__)

VIEW_TITLE = "Gaussian Fit Filter Coverage"

_PANELS = (
    ("ic1", "IC1"),
    ("ic2", "IC2"),
)

_THRESHOLD_ROWS = (
    ("confidence", "Confidence threshold (%)", "{:.2f}%", 10.0, 100.0),
    ("peak", "Peak IC current threshold (nA)", "{:.2f} nA", 0.10, None),
)

Y_PAD = 4.0
PEAK_XLIM_COVERAGE_PCT = 90.0


def _threshold_where_coverage_drops_below(
    thresholds: np.ndarray,
    coverage_pct: np.ndarray,
    *,
    target_pct: float,
) -> float | None:
    below = coverage_pct < target_pct
    if not np.any(below):
        return None
    return float(thresholds[int(np.argmax(below))])


def _panel_xlim(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    ic_key: str,
    *,
    row_key: str,
) -> tuple[float, float]:
    _, _, _, x_pad, x_hi_fixed = next(r for r in _THRESHOLD_ROWS if r[0] == row_key)
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

    if row_key == "peak":
        drop_to_target: list[float] = []
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
            threshold = _threshold_where_coverage_drops_below(
                coverage.thresholds,
                ic_cov.coverage_pct,
                target_pct=PEAK_XLIM_COVERAGE_PCT,
            )
            if threshold is not None:
                drop_to_target.append(threshold)
        x_hi = max(drop_to_target) if drop_to_target else (max(hi_points) if hi_points else 1.0)
        return 0.0, x_hi

    if not drop_points:
        x_hi = x_hi_fixed if x_hi_fixed is not None else (max(hi_points) if hi_points else 1.0)
        return 0.0, x_hi

    x_lo_anchor = min(drop_points)
    x_lo = max(0.0, x_lo_anchor - x_pad)
    x_hi = x_hi_fixed if x_hi_fixed is not None else 100.0
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


def _style_error_code_axis(ax, codes: np.ndarray) -> None:
    ticks = [int(code) for code in codes]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [SPOT_ERROR_CODE_AXIS_LABELS[t] for t in ticks],
        fontsize=7,
        rotation=35,
        ha="right",
    )


def _plotted_error_confidence_counts(counts_by_code: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes = PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES
    return codes, counts_by_code[codes].astype(float)


def _row_coverage(
    session: SessionGaussianFitFilterCoverage,
    row_key: str,
) -> GaussianFitFilterSweep | None:
    if row_key == "confidence":
        return session.confidence
    if row_key == "peak":
        return session.peak
    return None


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
            legend_labels.append(f"100% breakpoint ({breakpoint_fmt.format(breakpoint)})")

        if legend_handles:
            ax.legend(legend_handles, legend_labels, fontsize=7, loc="lower left")


def _orphan_panel_error_codes(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    ic_key: str,
) -> np.ndarray:
    present: set[int] = set()
    for sid in loaded_ids:
        session = filter_data.get(sid)
        if session is None:
            continue
        ic_counts = session.orphan_spot_errors.ics.get(ic_key)
        if ic_counts is None:
            continue
        for code, count in enumerate(ic_counts.counts_by_code):
            if count > 0:
                present.add(code)
    return np.array(sorted(present), dtype=int)


def _plot_orphan_spot_error_row(
    axes_row,
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    colors: list,
) -> None:
    n_sessions = len(loaded_ids)
    group_width = 0.8
    bar_width = group_width / max(n_sessions, 1)

    for col_idx, (ax, (ic_key, panel_title)) in enumerate(zip(axes_row, _PANELS)):
        ax.set_title(panel_title)
        has_data = False
        ymax = 0.0
        panel_codes = _orphan_panel_error_codes(filter_data, loaded_ids, ic_key)

        for si, (sid, color) in enumerate(zip(loaded_ids, colors)):
            session = filter_data.get(sid)
            if session is None:
                continue
            ic_counts = session.orphan_spot_errors.ics.get(ic_key)
            if ic_counts is None or ic_counts.orphan_spots <= 0:
                continue

            all_counts = ic_counts.counts_by_code.astype(float)
            if all_counts.sum() <= 0:
                continue

            counts = all_counts[panel_codes] if panel_codes.size else all_counts
            if counts.sum() <= 0:
                continue

            has_data = True
            ymax = max(ymax, float(counts.max()))
            offsets = panel_codes + (si - (n_sessions - 1) / 2.0) * bar_width
            ax.bar(
                offsets,
                counts,
                width=bar_width,
                color=color,
                alpha=0.85,
                edgecolor="none",
                align="center",
            )

        if panel_codes.size:
            pad = 0.5
            ax.set_xlim(panel_codes[0] - pad, panel_codes[-1] + pad)
            _style_error_code_axis(ax, panel_codes)
        ax.set_xlabel("Spot error code")
        if col_idx == 0:
            ax.set_ylabel("Orphan spot rows")
        ax.grid(**GRID_KW, axis="y")

        if has_data:
            ax.set_ylim(0.0, ymax * 1.1 + 1.0)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.text(
                0.5,
                0.5,
                "No orphan spots",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )


def _plot_error_confidence_issue_row(
    axes_row,
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    colors: list,
) -> None:
    n_sessions = len(loaded_ids)
    group_width = 0.8
    bar_width = group_width / max(n_sessions, 1)

    for col_idx, (ax, (ic_key, panel_title)) in enumerate(zip(axes_row, _PANELS)):
        ax.set_title(panel_title)
        has_data = False
        ymax = 0.0

        for si, (sid, color) in enumerate(zip(loaded_ids, colors)):
            session = filter_data.get(sid)
            if session is None:
                continue
            ic_counts = session.error_confidence_issues.ics.get(ic_key)
            if ic_counts is None:
                continue

            codes, counts = _plotted_error_confidence_counts(ic_counts.counts_by_code)
            if counts.sum() <= 0:
                continue

            has_data = True
            ymax = max(ymax, float(counts.max()))
            offsets = codes + (si - (n_sessions - 1) / 2.0) * bar_width
            ax.bar(
                offsets,
                counts,
                width=bar_width,
                color=color,
                alpha=0.85,
                edgecolor="none",
                align="center",
            )

        ax.set_xlim(1.5, 5.5)
        _style_error_code_axis(ax, PLOTTED_ERROR_CONFIDENCE_ISSUE_CODES)
        ax.set_xlabel("Spot error code")
        if col_idx == 0:
            ax.set_ylabel("Beam-on rows (invalid confidence)")
        ax.grid(**GRID_KW, axis="y")

        if has_data:
            ax.set_ylim(0.0, ymax * 1.1 + 1.0)
        else:
            ax.set_ylim(0.0, 1.0)
            ax.text(
                0.5,
                0.5,
                "No issues",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )


def _plot_coverage(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
) -> None:
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    n_rows = len(_THRESHOLD_ROWS) + 2
    fig, axes = view_grid(
        n_rows,
        len(_PANELS),
        cell_w=CELL_SQUARE,
        cell_h=CELL_SQUARE * 0.65,
    )

    for row_idx, (row_key, xlabel, breakpoint_fmt, _x_pad, _x_hi) in enumerate(
        _THRESHOLD_ROWS
    ):
        _plot_filter_row(
            axes[row_idx],
            filter_data,
            loaded_ids,
            colors,
            row_key=row_key,
            xlabel=xlabel,
            breakpoint_fmt=breakpoint_fmt,
        )

    _plot_error_confidence_issue_row(
        axes[len(_THRESHOLD_ROWS)],
        filter_data,
        loaded_ids,
        colors,
    )
    _plot_orphan_spot_error_row(
        axes[len(_THRESHOLD_ROWS) + 1],
        filter_data,
        loaded_ids,
        colors,
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
