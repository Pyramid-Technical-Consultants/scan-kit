"""Matplotlib renderer for the universal binned summary viewer."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..common.data_filter import filter_binned_session_data
from ..common import (
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    VIEW_HEADER_SUBPLOT_TOP,
    add_binned_trend,
    add_correlation_scatter,
    add_scatter_trend,
    compute_hist_bin_range,
    draw_probability_histogram,
    link_boxplot_to_histogram,
    plot_boxplots_for_column,
    plot_density_contours,
    plot_means_for_column,
    plot_violins_for_column,
    prepare_binned_column,
    scatter_with_trend,
    set_view_header,
    style_binned_axes,
    style_linear_binned_axes,
)
from .binned_summary_catalog import (
    GLYPH_BOX,
    GLYPH_CONTOUR,
    GLYPH_MEAN,
    GLYPH_SCATTER,
    GLYPH_VIOLIN,
    X_PARAM_BY_ID,
    Y_DOSE_RATE,
    Y_GROUP_BY_ID,
    BinnedSummaryConfig,
)
from .binned_summary_data import available_series_keys

# Side panels (hist / correlation) are narrow marginals; keep wspace small so they
# sit close to the main column without eating horizontal space.
_MAIN_COL_WIDTH = 8.0
_SIDE_COL_WIDTH = 0.55
_SUMMARY_GRID_WSPACE = 0.04
_SUMMARY_GRID_HSPACE = 0.22


def _summary_width_ratios(show_hist: bool, show_corr: bool) -> list[float]:
    ratios = [_MAIN_COL_WIDTH]
    if show_hist:
        ratios.append(_SIDE_COL_WIDTH)
    if show_corr:
        ratios.append(_SIDE_COL_WIDTH)
    return ratios


def _summary_gridspec(
    fig: plt.Figure,
    n_rows: int,
    *,
    show_hist: bool,
    show_corr: bool,
) -> gridspec.GridSpec:
    n_cols = 1 + int(show_hist) + int(show_corr)
    return gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=_summary_width_ratios(show_hist, show_corr),
        hspace=_SUMMARY_GRID_HSPACE,
        wspace=_SUMMARY_GRID_WSPACE,
        top=VIEW_HEADER_SUBPLOT_TOP,
        bottom=0.08,
        left=0.07,
        right=0.98,
    )


@dataclass
class _CorrPair:
    x_key: str
    y_key: str


def _style_side_panel_axis(ax, *, row: int, keep_ylabel: bool = False) -> None:
    """Compact tick labels; y-axis label only on the top marginal panel."""
    ax.tick_params(labelsize=8)
    if row > 0 and not keep_ylabel:
        ax.set_ylabel("")


def _corr_pairs_for_series(series_keys: list[str]) -> list[_CorrPair]:
    if len(series_keys) < 2:
        return []
    pairs = []
    for i, key in enumerate(series_keys):
        other = series_keys[(i + 1) % len(series_keys)]
        pairs.append(_CorrPair(key, other))
    return pairs


def _render_correlation_panel(
    ax,
    session_data: dict[str, dict],
    pair: _CorrPair,
    loaded_ids: list[str],
    colors: list,
    labels: dict[str, str],
    *,
    row: int,
    n_rows: int,
) -> None:
    add_correlation_scatter(
        ax,
        session_data,
        pair.x_key,
        pair.y_key,
        loaded_ids,
        colors,
        xlabel=labels.get(pair.x_key, pair.x_key) if row == n_rows - 1 else None,
        ylabel=labels.get(pair.y_key, pair.y_key),
    )
    _style_side_panel_axis(ax, row=row, keep_ylabel=True)
    if row < n_rows - 1:
        ax.set_xlabel("")


def render_binned_summary(
    fig: plt.Figure,
    config: BinnedSummaryConfig,
    session_data: dict[str, dict],
    base_dir: str,
) -> None:
    """Clear *fig* and draw the binned summary layout."""
    fig.clear()
    if not session_data:
        fig.text(0.5, 0.5, "No session data loaded", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    y_group = Y_GROUP_BY_ID.get(config.y_group)
    x_param = X_PARAM_BY_ID.get(config.x_param)
    if y_group is None or x_param is None:
        fig.text(0.5, 0.5, "Invalid Y/X selection", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    series_keys = available_series_keys(session_data, config.y_group)
    column_keys = [s.key for s in y_group.series]
    session_data = filter_binned_session_data(
        session_data, column_keys, config.data_filter,
    )
    series_keys = available_series_keys(session_data, config.y_group)
    labels = {s.key: s.label for s in y_group.series}
    if not series_keys:
        fig.text(
            0.5, 0.5, f"No {y_group.label} columns available",
            ha="center", va="center",
        )
        fig.canvas.draw_idle()
        return

    if config.glyph in (GLYPH_SCATTER, GLYPH_CONTOUR):
        _render_linear_summary(
            fig, config, session_data, base_dir, y_group, x_param,
            series_keys, labels,
        )
        return

    if x_param.column not in next(iter(session_data.values()), {}):
        # Still try — prepare_binned_column will yield empty categories.
        pass

    n_bins = config.n_bins if config.n_bins is not None else x_param.n_bins
    prepared, categories = prepare_binned_column(
        session_data,
        x_param.column,
        mode=x_param.bin_mode,
        n_bins=n_bins,
        out_key="_bin",
    )
    if not categories:
        fig.text(
            0.5, 0.5, f"No finite values for X parameter: {x_param.label}",
            ha="center", va="center",
        )
        fig.canvas.draw_idle()
        return

    loaded_ids = list(prepared.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    n_rows = len(series_keys)

    set_view_header(fig, config.title, loaded_ids, colors, base_dir=base_dir)
    gs = _summary_gridspec(
        fig,
        n_rows,
        show_hist=config.show_hist,
        show_corr=config.show_corr,
    )

    main_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    hist_axes = []
    corr_axes = []
    col = 1
    if config.show_hist:
        hist_axes = [fig.add_subplot(gs[i, col]) for i in range(n_rows)]
        col += 1
    if config.show_corr:
        corr_axes = [fig.add_subplot(gs[i, col]) for i in range(n_rows)]

    corr_pairs = _corr_pairs_for_series(series_keys)
    selectors = []

    for row, key in enumerate(series_keys):
        ax = main_axes[row]
        col_data = {sid: d for sid, d in prepared.items() if key in d}
        col_colors = [colors[loaded_ids.index(sid)] for sid in col_data]
        if not col_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        if config.glyph == GLYPH_VIOLIN:
            plot_violins_for_column(
                ax, col_data, key, categories, col_colors, bin_key="_bin",
            )
        elif config.glyph == GLYPH_MEAN:
            plot_means_for_column(
                ax, col_data, key, categories, col_colors, bin_key="_bin",
                connect_lines=True,
            )
            if config.y_group == Y_DOSE_RATE and config.show_trend:
                for (_sid, data), color in zip(col_data.items(), col_colors):
                    avg = data.get("session_avg_rate")
                    if avg is None:
                        continue
                    avg_val = float(np.asarray(avg).flat[0])
                    if np.isfinite(avg_val):
                        ax.axhline(
                            avg_val,
                            color=color,
                            linestyle="--",
                            linewidth=1.4,
                            alpha=0.9,
                            zorder=1,
                        )
        else:
            plot_boxplots_for_column(
                ax, col_data, key, categories, col_colors,
                showfliers=config.show_fliers, width=0.3, bin_key="_bin",
            )

        if config.show_trend and config.y_group != Y_DOSE_RATE:
            unit = y_group.trend_unit.replace("unit", x_param.id)
            add_binned_trend(
                ax, col_data, key, categories, col_colors,
                bin_key="_bin", unit=unit, position_offset=0.35,
            )

        style_binned_axes(
            ax, categories, xlabel=x_param.xlabel if row == n_rows - 1 else "",
            ylabel=labels.get(key, key),
        )
        if row < n_rows - 1:
            ax.set_xlabel("")
        if config.y_group != Y_DOSE_RATE:
            ax.axhline(0, **REFLINE_KW)

        if config.show_hist and hist_axes:
            sels = link_boxplot_to_histogram(
                ax, hist_axes[row],
                col_data, categories, key, col_colors, list(col_data.keys()),
                hist_xlabels=labels.get(key, key),
                hist_bin_count=config.hist_bin_count,
                hist_shared_bins=config.hist_shared_bins,
                hist_ylabel="Probability (%)" if row == 0 else None,
                bin_key="_bin",
            )
            _style_side_panel_axis(hist_axes[row], row=row)
            if sels:
                selectors.extend(sels)

        if config.show_corr and corr_axes and row < len(corr_pairs):
            _render_correlation_panel(
                corr_axes[row],
                prepared,
                corr_pairs[row],
                loaded_ids,
                colors,
                labels,
                row=row,
                n_rows=n_rows,
            )

    # Keep span selectors alive on the figure.
    if selectors:
        fig._scan_kit_bin_selectors = selectors  # type: ignore[attr-defined]

    fig.canvas.draw_idle()


def _render_linear_summary(
    fig: plt.Figure,
    config: BinnedSummaryConfig,
    session_data: dict[str, dict],
    base_dir: str,
    y_group,
    x_param,
    series_keys: list[str],
    labels: dict[str, str],
) -> None:
    """Scatter or contour Y vs raw X (no binning) with a linear x-axis."""
    if x_param.column not in next(iter(session_data.values()), {}):
        fig.text(
            0.5, 0.5, f"No data for X parameter: {x_param.label}",
            ha="center", va="center",
        )
        fig.canvas.draw_idle()
        return

    loaded_ids = [
        sid for sid in session_data
        if x_param.column in session_data[sid]
        and np.isfinite(np.asarray(session_data[sid][x_param.column], dtype=float)).any()
    ]
    if not loaded_ids:
        fig.text(
            0.5, 0.5, f"No finite values for X parameter: {x_param.label}",
            ha="center", va="center",
        )
        fig.canvas.draw_idle()
        return

    scatter_data = {sid: session_data[sid] for sid in loaded_ids}
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]
    n_rows = len(series_keys)

    set_view_header(fig, config.title, loaded_ids, colors, base_dir=base_dir)
    gs = _summary_gridspec(
        fig,
        n_rows,
        show_hist=config.show_hist,
        show_corr=config.show_corr,
    )

    main_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_rows)]
    hist_axes = []
    corr_axes = []
    col = 1
    if config.show_hist:
        hist_axes = [fig.add_subplot(gs[i, col]) for i in range(n_rows)]
        col += 1
    if config.show_corr:
        corr_axes = [fig.add_subplot(gs[i, col]) for i in range(n_rows)]

    corr_pairs = _corr_pairs_for_series(series_keys)
    shared_hist_range = None
    if config.show_hist and config.hist_shared_bins:
        shared_hist_range = compute_hist_bin_range(
            scatter_data,
            series_keys,
            loaded_ids,
        )

    for row, key in enumerate(series_keys):
        ax = main_axes[row]
        col_data = {
            sid: scatter_data[sid]
            for sid in loaded_ids
            if key in scatter_data[sid]
        }
        col_colors = [colors[loaded_ids.index(sid)] for sid in col_data]
        if not col_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        for (sid, data), color in zip(col_data.items(), col_colors):
            x = np.asarray(data[x_param.column], dtype=float)
            y = np.asarray(data[key], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                continue
            xf, yf = x[mask], y[mask]
            if config.glyph == GLYPH_CONTOUR:
                plot_density_contours(
                    ax,
                    xf,
                    yf,
                    color,
                    contour_cutoff_percentile=config.contour_cutoff_percentile,
                )
                if config.show_trend:
                    add_scatter_trend(
                        ax, xf, yf, color=color, unit="", scatter=False, label=sid,
                    )
            elif config.show_trend:
                scatter_with_trend(ax, xf, yf, color=color, label=sid)
            else:
                ax.scatter(
                    xf, yf,
                    c=color, alpha=0.35, s=18, edgecolors="none", label=sid,
                )

        if config.y_group == Y_DOSE_RATE and config.show_trend:
            for (_sid, data), color in zip(col_data.items(), col_colors):
                avg = data.get("session_avg_rate")
                if avg is None:
                    continue
                avg_val = float(np.asarray(avg).flat[0])
                if np.isfinite(avg_val):
                    ax.axhline(
                        avg_val,
                        color=color,
                        linestyle="--",
                        linewidth=1.4,
                        alpha=0.9,
                        zorder=1,
                    )

        style_linear_binned_axes(
            ax,
            xlabel=x_param.xlabel if row == n_rows - 1 else "",
            ylabel=labels.get(key, key),
        )
        if row < n_rows - 1:
            ax.set_xlabel("")
        if config.y_group != Y_DOSE_RATE:
            ax.axhline(0, **REFLINE_KW)

        if config.show_hist and hist_axes:
            hist_ax = hist_axes[row]
            row_range = shared_hist_range
            draw_probability_histogram(
                hist_ax,
                scatter_data,
                key,
                loaded_ids,
                colors,
                bin_count=config.hist_bin_count,
                bin_range=row_range,
                xlabel=labels.get(key, key),
                ylabel="Probability (%)" if row == 0 else None,
            )
            hist_ax.grid(**GRID_KW)
            _style_side_panel_axis(hist_ax, row=row)

        if config.show_corr and corr_axes and row < len(corr_pairs):
            _render_correlation_panel(
                corr_axes[row],
                scatter_data,
                corr_pairs[row],
                loaded_ids,
                colors,
                labels,
                row=row,
                n_rows=n_rows,
            )

    fig.canvas.draw_idle()
