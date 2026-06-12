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
from ..common.g3_timeslice_position import G3_FIT_CONFIDENCE_MIN
from ..common.timeslice_gaussian_fit_filter_coverage import (
    GaussianFitFilterSweep,
    OrphanSpotPeakSeries,
    SessionGaussianFitFilterCoverage,
    compute_session_gaussian_fit_filter_coverage,
    spot_error_code_name,
)

_log = logging.getLogger(__name__)

VIEW_TITLE = "Gaussian Fit Filter Coverage"

_ERROR_MARKER_COLORS = {
    0: "#2ca02c",
    1: "#ff7f0e",
    2: "#d62728",
    3: "#9467bd",
    4: "#8c564b",
    5: "#e377c2",
}

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


def _orphan_peak_plot_entries(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
) -> dict[str, list[tuple[str, OrphanSpotPeakSeries]]]:
    entries = {ic_key: [] for ic_key, _ in _PANELS}
    for sid in loaded_ids:
        session = filter_data.get(sid)
        if session is None:
            continue
        for ic_key, _ in _PANELS:
            for series in session.orphan_spot_peaks.ics.get(ic_key, ()):
                entries[ic_key].append((sid, series))
    return entries


def _series_x_values(series: OrphanSpotPeakSeries) -> tuple[np.ndarray, str]:
    if series.time_ms is not None and np.any(np.isfinite(series.time_ms)):
        return series.time_ms, "Time from spot start (ms)"
    return series.beam_on_index.astype(float), "Beam-on row index"


def _error_marker_colors(codes: np.ndarray) -> list[str]:
    colors: list[str] = []
    for code in codes:
        if not np.isfinite(code):
            colors.append("#bbbbbb")
            continue
        colors.append(_ERROR_MARKER_COLORS.get(int(round(code)), "#bbbbbb"))
    return colors


def _plot_orphan_peak_series(
    ax,
    series: OrphanSpotPeakSeries,
    *,
    sid: str,
    color: str,
) -> str:
    x_vals, x_label = _series_x_values(series)
    ax.plot(x_vals, series.peak_x, color=color, linewidth=1.2, alpha=0.55, zorder=1)
    ax.plot(
        x_vals,
        series.peak_y,
        color=color,
        linewidth=1.2,
        alpha=0.55,
        linestyle="--",
        zorder=1,
    )
    ax.scatter(
        x_vals,
        series.peak_x,
        c=_error_marker_colors(series.error_x),
        s=28,
        marker="o",
        edgecolors=color,
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        x_vals,
        series.peak_y,
        c=_error_marker_colors(series.error_y),
        s=28,
        marker="s",
        edgecolors=color,
        linewidths=0.8,
        zorder=3,
    )

    layer = int(series.layer_id) if np.isfinite(series.layer_id) else series.layer_id
    spot = int(series.spot_no) if np.isfinite(series.spot_no) else series.spot_no
    ax.set_title(f"{sid}  L{layer} spot {spot}", fontsize=8)
    ax.grid(**GRID_KW)

    ymax = np.nanmax(np.concatenate([series.peak_x, series.peak_y]))
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0.0, ymax * 1.15)
    return x_label


def _orphan_peak_legend_handles() -> list[Line2D]:
    handles = [
        Line2D([0], [0], color="0.35", linewidth=1.2, label="X peak"),
        Line2D([0], [0], color="0.35", linewidth=1.2, linestyle="--", label="Y peak"),
    ]
    for code, color in sorted(_ERROR_MARKER_COLORS.items()):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markeredgecolor="0.35",
                markersize=6,
                label=f"{code} {spot_error_code_name(code)}",
            )
        )
    return handles


def _has_weighted_position_rms(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
) -> bool:
    return any(
        filter_data.get(sid) is not None
        and filter_data[sid].weighted_position_rms is not None
        for sid in loaded_ids
    )


def _plot_weighted_position_rms_row(
    axes_row,
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    colors: list,
) -> None:
    for col_idx, (ax, (ic_key, panel_title)) in enumerate(zip(axes_row, _PANELS)):
        ax.set_title(panel_title)
        has_data = False
        rms_values: list[np.ndarray] = []

        for sid, color in zip(loaded_ids, colors):
            session = filter_data.get(sid)
            if session is None or session.weighted_position_rms is None:
                continue
            ic_rms = session.weighted_position_rms.ics.get(ic_key)
            if ic_rms is None:
                continue

            thresholds = session.weighted_position_rms.thresholds
            rms_xy = ic_rms.rms_xy_mm
            finite = np.isfinite(rms_xy)
            if not np.any(finite):
                continue

            has_data = True
            rms_values.append(rms_xy[finite])
            ax.plot(
                thresholds,
                rms_xy,
                color=color,
                linewidth=1.5,
                label=sid,
            )
            ax.plot(
                thresholds,
                ic_rms.rms_x_mm,
                color=color,
                linewidth=0.9,
                alpha=0.35,
                linestyle=":",
            )
            ax.plot(
                thresholds,
                ic_rms.rms_y_mm,
                color=color,
                linewidth=0.9,
                alpha=0.35,
                linestyle="--",
            )

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No position data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
            continue

        ax.axvline(
            G3_FIT_CONFIDENCE_MIN,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        if col_idx == 0:
            ax.set_ylabel("Weighted position RMS (mm)")
        ax.set_xlabel("Confidence threshold (%)")
        ax.grid(**GRID_KW)
        ax.set_xlim(0.0, 100.0)

        vals = np.concatenate(rms_values)
        ymax = float(np.nanmax(vals))
        if ymax > 0:
            ax.set_ylim(0.0, ymax * 1.12)

        if col_idx == len(_PANELS) - 1:
            ax.legend(
                handles=[
                    Line2D([0], [0], color="0.35", linewidth=1.5, label="RMS X/Y"),
                    Line2D([0], [0], color="0.35", linewidth=0.9, linestyle=":", label="RMS X"),
                    Line2D([0], [0], color="0.35", linewidth=0.9, linestyle="--", label="RMS Y"),
                    Line2D(
                        [0],
                        [0],
                        color="0.45",
                        linestyle=":",
                        linewidth=1.0,
                        label=f"Production gate ({G3_FIT_CONFIDENCE_MIN:.0f}%)",
                    ),
                ],
                fontsize=6,
                loc="upper right",
                framealpha=0.9,
            )


def _plot_orphan_peak_rows(
    axes,
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    colors: list,
    *,
    start_row: int,
) -> None:
    entries = _orphan_peak_plot_entries(filter_data, loaded_ids)
    max_rows = max((len(entries[ic_key]) for ic_key, _ in _PANELS), default=0)
    if max_rows == 0:
        return

    sid_colors = dict(zip(loaded_ids, colors))
    x_label = "Beam-on row index"

    for row_idx in range(max_rows):
        for col_idx, (ic_key, panel_title) in enumerate(_PANELS):
            ax = axes[start_row + row_idx, col_idx]
            panel_entries = entries[ic_key]
            if row_idx >= len(panel_entries):
                ax.axis("off")
                continue

            sid, series = panel_entries[row_idx]
            x_label = _plot_orphan_peak_series(ax, series, sid=sid, color=sid_colors[sid])
            if row_idx == max_rows - 1:
                ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel("Peak amplitude (nA)")
            if row_idx == 0:
                ax.text(
                    0.02,
                    0.98,
                    panel_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                )

    axes[start_row, -1].legend(
        handles=_orphan_peak_legend_handles(),
        fontsize=6,
        loc="upper right",
        framealpha=0.9,
    )


def _plot_coverage(
    filter_data: dict[str, SessionGaussianFitFilterCoverage],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
) -> None:
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    entries = _orphan_peak_plot_entries(filter_data, loaded_ids)
    orphan_rows = max((len(entries[ic_key]) for ic_key, _ in _PANELS), default=0)
    rms_row = 1 if _has_weighted_position_rms(filter_data, loaded_ids) else 0
    n_rows = len(_THRESHOLD_ROWS) + rms_row + orphan_rows

    cell_h = CELL_SQUARE * (0.65 if orphan_rows == 0 else 0.55)
    fig, axes = view_grid(
        n_rows,
        len(_PANELS),
        cell_w=CELL_SQUARE,
        cell_h=cell_h,
    )

    row_idx = 0
    for row_key, xlabel, breakpoint_fmt, _x_pad, _x_hi in _THRESHOLD_ROWS:
        _plot_filter_row(
            axes[row_idx],
            filter_data,
            loaded_ids,
            colors,
            row_key=row_key,
            xlabel=xlabel,
            breakpoint_fmt=breakpoint_fmt,
        )
        row_idx += 1

    if rms_row:
        _plot_weighted_position_rms_row(
            axes[row_idx],
            filter_data,
            loaded_ids,
            colors,
        )
        row_idx += 1

    if orphan_rows:
        _plot_orphan_peak_rows(
            axes,
            filter_data,
            loaded_ids,
            colors,
            start_row=row_idx,
        )

    finish_view(fig, title, loaded_ids, colors, base_dir=base_dir)


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
