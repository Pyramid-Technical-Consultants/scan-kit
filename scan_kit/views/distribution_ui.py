"""Matplotlib renderer for the Distribution Explorer viewer."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..common.data_filter import filter_distribution_session_data
from ..common.confidence_correlation import render_confidence_correlations
from ..common.gaussian_fit_filter_coverage_plot import render_gaussian_fit_filter_coverage
from ..common.ic_xy_distribution import render_ic_xy_distribution
from ..common.position_error_distribution import render_position_error_distribution
from ..common.sigma_distribution import render_sigma_distribution
from .distribution_catalog import (
    MODE_BY_ID,
    MODE_CONFIDENCE_TIMESLICE,
    MODE_GAUSSIAN_FILTER,
    MODE_IC12_POS_DIFF_SPOT,
    MODE_IC12_POS_DIFF_TIMESLICE,
    MODE_POSITION_ERROR_SPOT,
    MODE_POSITION_ERROR_TIMESLICE,
    MODE_POSITION_SPOT,
    MODE_POSITION_TIMESLICE,
    MODE_SIGMA_ERROR_SPOT,
    MODE_SIGMA_ERROR_TIMESLICE,
    MODE_SIGMA_SPOT,
    MODE_SIGMA_TIMESLICE,
    DistributionConfig,
)


def _panel_kwargs(config: DistributionConfig) -> dict:
    return {
        "show_plan": config.show_plan,
        "show_ic1": config.show_ic1,
        "show_ic2": config.show_ic2,
    }


def render_distribution(
    fig: plt.Figure,
    config: DistributionConfig,
    session_data: dict,
    base_dir: str,
) -> None:
    """Clear *fig* and draw the selected distribution mode."""
    if not session_data:
        fig.clear()
        fig.text(0.5, 0.5, "No session data loaded", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    mode = MODE_BY_ID.get(config.mode)
    if mode is None:
        fig.clear()
        fig.text(0.5, 0.5, "Invalid distribution mode", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    loaded_ids = list(session_data.keys())
    title = config.title
    plot_style = config.plot_style
    contour_cutoff = config.contour_cutoff_percentile
    session_data = filter_distribution_session_data(session_data, config.data_filter)
    loaded_ids = list(session_data.keys())
    if not loaded_ids:
        fig.clear()
        fig.text(0.5, 0.5, "No data after filter", ha="center", va="center")
        fig.canvas.draw_idle()
        return

    panels = _panel_kwargs(config)

    if config.mode in (MODE_POSITION_ERROR_TIMESLICE, MODE_POSITION_ERROR_SPOT):
        render_position_error_distribution(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            plot_style=plot_style,
            contour_cutoff_percentile=contour_cutoff,
            fig=fig,
            show=False,
            **panels,
        )
        return

    if config.mode in (MODE_POSITION_TIMESLICE, MODE_POSITION_SPOT):
        render_ic_xy_distribution(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            limit_mode="square",
            plot_style=plot_style,
            contour_cutoff_percentile=contour_cutoff,
            x_hist_label="X Position (mm)",
            y_hist_label="Y Position (mm)",
            fig=fig,
            show=False,
            **panels,
        )
        return

    if config.mode in (MODE_IC12_POS_DIFF_TIMESLICE, MODE_IC12_POS_DIFF_SPOT):
        render_ic_xy_distribution(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            limit_mode="symmetric",
            plot_style=plot_style,
            contour_cutoff_percentile=contour_cutoff,
            x_hist_label="IC2−IC1 X (mm)",
            y_hist_label="IC2−IC1 Y (mm)",
            fig=fig,
            show=False,
            show_plan=False,
            show_ic1=True,
            show_ic2=False,
        )
        return

    if config.mode in (MODE_SIGMA_ERROR_TIMESLICE, MODE_SIGMA_ERROR_SPOT):
        render_ic_xy_distribution(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            limit_mode="symmetric",
            plot_style=plot_style,
            contour_cutoff_percentile=contour_cutoff,
            reference_circle=True,
            x_hist_label="X Sigma Error (mm)",
            y_hist_label="Y Sigma Error (mm)",
            fig=fig,
            show=False,
            **panels,
        )
        return

    if config.mode in (MODE_SIGMA_TIMESLICE, MODE_SIGMA_SPOT):
        render_sigma_distribution(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            plot_style=plot_style,
            contour_cutoff_percentile=contour_cutoff,
            fig=fig,
            show=False,
            **panels,
        )
        return

    if config.mode == MODE_CONFIDENCE_TIMESLICE:
        render_confidence_correlations(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            fig=fig,
            show=False,
        )
        return

    if config.mode == MODE_GAUSSIAN_FILTER:
        render_gaussian_fit_filter_coverage(
            session_data,
            loaded_ids,
            title=title,
            base_dir=base_dir,
            fig=fig,
            show=False,
        )
        return

    fig.clear()
    fig.text(
        0.5,
        0.5,
        f"Unsupported mode: {config.mode}",
        ha="center",
        va="center",
    )
    fig.canvas.draw_idle()
