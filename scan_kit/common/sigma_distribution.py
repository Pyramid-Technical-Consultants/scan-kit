"""Shared rendering for IC sigma density contours and X/Y histograms."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .ic_xy_distribution import (
    DEFAULT_CONTOUR_CUTOFF_PERCENTILE,
    PlotStyle,
    render_ic_xy_distribution,
)
from .timeslice_sigma import SessionIcSigmas


def render_sigma_distribution(
    session_data: dict[str, SessionIcSigmas],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
    plot_style: PlotStyle = "contour",
    contour_cutoff_percentile: float = DEFAULT_CONTOUR_CUTOFF_PERCENTILE,
    show_plan: bool = False,
    show_ic1: bool = True,
    show_ic2: bool = True,
    fig=None,
    show: bool = True,
) -> None:
    """Render IC1/IC2 sigma density or scatter plots and X/Y histograms."""
    render_ic_xy_distribution(
        session_data,
        loaded_ids,
        title=title,
        base_dir=base_dir,
        limit_mode="positive",
        plot_style=plot_style,
        contour_cutoff_percentile=contour_cutoff_percentile,
        show_plan=show_plan,
        show_ic1=show_ic1,
        show_ic2=show_ic2,
        reference_circle=False,
        tolerance_lines=False,
        x_hist_label="X Sigma (mm)",
        y_hist_label="Y Sigma (mm)",
        fig=fig,
        show=show,
    )
