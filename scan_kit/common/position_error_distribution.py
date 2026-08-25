"""Shared rendering for position-error density contours and X/Y histograms.

Both the timeslice view (dense beam-on samples) and the spot view (one sample
per delivered spot) load IC1/IC2 X/Y position errors into
:class:`SessionPositionErrors` and hand them to
:func:`render_position_error_distribution`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .ic_xy_distribution import (
    DEFAULT_CONTOUR_CUTOFF_PERCENTILE,
    PlotStyle,
    render_ic_xy_distribution,
)
from .timeslice_position_error import SessionPositionErrors


def render_position_error_distribution(
    session_data: dict[str, SessionPositionErrors],
    loaded_ids: list[str],
    *,
    title: str,
    base_dir: str,
    plot_style: PlotStyle = "contour",
    contour_cutoff_percentile: float = DEFAULT_CONTOUR_CUTOFF_PERCENTILE,
    fig=None,
    show: bool = True,
) -> None:
    """Render IC1/IC2 position error density or scatter plots and histograms."""
    render_ic_xy_distribution(
        session_data,
        loaded_ids,
        title=title,
        base_dir=base_dir,
        limit_mode="symmetric",
        plot_style=plot_style,
        contour_cutoff_percentile=contour_cutoff_percentile,
        show_plan=False,
        reference_circle=True,
        tolerance_lines=True,
        x_hist_label="X Error (mm)",
        y_hist_label="Y Error (mm)",
        fig=fig,
        show=show,
    )
