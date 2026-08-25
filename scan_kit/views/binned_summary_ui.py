"""Matplotlib renderer for the universal binned summary viewer."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from ..common.data_filter import filter_binned_session_data
from ..common import (
    DEFAULT_SESSION_COLORS,
    REFLINE_KW,
    VIEW_HEADER_SUBPLOT_TOP,
    add_binned_trend,
    add_correlation_scatter,
    link_boxplot_to_histogram,
    plot_boxplots_for_column,
    plot_means_for_column,
    plot_violins_for_column,
    prepare_binned_column,
    set_view_header,
    style_binned_axes,
)
from .binned_summary_catalog import (
    GLYPH_BOX,
    GLYPH_MEAN,
    GLYPH_VIOLIN,
    X_PARAM_BY_ID,
    Y_DOSE_RATE,
    Y_GROUP_BY_ID,
    BinnedSummaryConfig,
)
from .binned_summary_data import available_series_keys


@dataclass
class _CorrPair:
    x_key: str
    y_key: str
    title: str


def _corr_pairs_for_series(series_keys: list[str]) -> list[_CorrPair]:
    if len(series_keys) < 2:
        return []
    pairs = []
    for i, key in enumerate(series_keys):
        other = series_keys[(i + 1) % len(series_keys)]
        pairs.append(_CorrPair(key, other, f"{key} vs {other}"))
    return pairs


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
    show_side = config.show_hist or config.show_corr
    n_cols = 1 + int(config.show_hist) + int(config.show_corr)
    width_ratios = [4]
    if config.show_hist:
        width_ratios.append(1)
    if config.show_corr:
        width_ratios.append(1)

    set_view_header(fig, config.title, loaded_ids, colors, base_dir=base_dir)
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.28, wspace=0.25,
        top=VIEW_HEADER_SUBPLOT_TOP, bottom=0.08, left=0.07, right=0.98,
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
                hist_titles="Distribution",
                bin_key="_bin",
            )
            if sels:
                selectors.extend(sels)

        if config.show_corr and corr_axes and row < len(corr_pairs):
            pair = corr_pairs[row]
            add_correlation_scatter(
                corr_axes[row], prepared, pair.x_key, pair.y_key,
                loaded_ids, colors,
            )
            corr_axes[row].set_title(pair.title, fontsize=9)

    # Keep span selectors alive on the figure.
    if selectors:
        fig._scan_kit_bin_selectors = selectors  # type: ignore[attr-defined]

    fig.canvas.draw_idle()
