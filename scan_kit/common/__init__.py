"""Common utilities for scan-kit analysis scripts."""

from .io import load_csv_from_zip, load_timeslice_device_units
from .transform import remap, IC1_X_MAP, IC1_Y_MAP, IC2_X_MAP, IC2_Y_MAP
from .validation import create_valid_mask, apply_validation
from .processing import load_session_raw, process_position_data
from .plotting import (
    plot_boxplots_for_column,
    plot_scatter_energy,
    add_energy_colorbar,
    annotate_slopes,
    make_session_legend,
    style_energy_axes,
    DEFAULT_SESSION_COLORS,
    FIG_SIZE_2x2,
    FIG_SIZE_1x2,
    FIG_SIZE_SINGLE,
    SUPTITLE_KW,
    GRID_KW,
    REFLINE_KW,
    SCATTER_ALPHA,
    SCATTER_SIZE,
    SLOPE_LABEL_KW,
    SLOPE_LABEL_BOX,
)

__all__ = [
    "load_csv_from_zip",
    "load_timeslice_device_units",
    "remap",
    "IC1_X_MAP",
    "IC1_Y_MAP",
    "IC2_X_MAP",
    "IC2_Y_MAP",
    "create_valid_mask",
    "apply_validation",
    "load_session_raw",
    "process_position_data",
    "plot_boxplots_for_column",
    "plot_scatter_energy",
    "add_energy_colorbar",
    "annotate_slopes",
    "make_session_legend",
    "style_energy_axes",
    "DEFAULT_SESSION_COLORS",
    "FIG_SIZE_2x2",
    "FIG_SIZE_1x2",
    "FIG_SIZE_SINGLE",
    "SUPTITLE_KW",
    "GRID_KW",
    "REFLINE_KW",
    "SCATTER_ALPHA",
    "SCATTER_SIZE",
    "SLOPE_LABEL_KW",
    "SLOPE_LABEL_BOX",
]
