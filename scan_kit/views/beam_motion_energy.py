"""Per-energy IC1/IC2 position error motion paths from timeslice data by spill."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ..common import (
    C_LAYER_ID,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    detect_beam_on_mask,
    detect_spill_segments,
    finish_view,
    subtract_background_frames,
    view_grid,
)
from ..common.session_source import load_session_timeslice_device_units
from ..common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_error_arrays,
    resolve_session_timeslice_error_source,
)
from .timeslice_replay_common import load_energy_lookups, resolve_col, resolve_frame_energy

_log = logging.getLogger(__name__)

MAX_GRID_COLS = 10
MAX_GRID_ROWS = 8
# Dense small-multiples grid: tiny per-energy cells (overrides the standard size).
CELL_WIDTH_IN = 1.3
CELL_HEIGHT_IN = 1.1

LINE_LW = 0.7
LINE_ALPHA = 0.75
REF_CIRCLE_RADIUS_MM = 1.0


@dataclass(frozen=True)
class SpillPath:
    ic1_x_err: np.ndarray
    ic1_y_err: np.ndarray
    ic2_x_err: np.ndarray
    ic2_y_err: np.ndarray


def _spill_path_from_arrays(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    start: int,
    end: int,
) -> SpillPath | None:
    ic1_x_err, ic1_y_err, ic2_x_err, ic2_y_err = (
        arr[start:end] for arr in arrays
    )
    if not any(np.isfinite(v).any() for v in (ic1_x_err, ic1_y_err, ic2_x_err, ic2_y_err)):
        return None
    return SpillPath(
        ic1_x_err=ic1_x_err,
        ic1_y_err=ic1_y_err,
        ic2_x_err=ic2_x_err,
        ic2_y_err=ic2_y_err,
    )


def _load_session_spill_paths(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict[float, list[SpillPath]] | None:
    loaded = load_energy_lookups(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer, energy_by_idx = loaded

    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    error_source = resolve_session_timeslice_error_source(src, frames)
    if ts_layer is None or error_source is None:
        return None

    by_energy: dict[float, list[SpillPath]] = defaultdict(list)

    for frame_idx, df in enumerate(frames):
        energy = resolve_frame_energy(
            df,
            frame_idx,
            energy_by_layer=energy_by_layer,
            energy_by_idx=energy_by_idx,
            layer_col=ts_layer,
        )
        if energy is None:
            continue

        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        frame_errors = frame_timeslice_error_arrays(df, error_source)
        if frame_errors is None:
            continue

        for start, end in detect_spill_segments(beam_on):
            path = _spill_path_from_arrays(frame_errors, start, end)
            if path is not None:
                by_energy[float(energy)].append(path)

    if not by_energy:
        return None
    return dict(by_energy)


def _grid_shape(n_energies: int) -> tuple[int, int]:
    if n_energies <= 0:
        return 1, 1
    ncols = min(MAX_GRID_COLS, max(1, math.ceil(math.sqrt(n_energies * 1.25))))
    nrows = min(MAX_GRID_ROWS, math.ceil(n_energies / ncols))
    while nrows * ncols < n_energies and ncols < MAX_GRID_COLS:
        ncols += 1
        nrows = min(MAX_GRID_ROWS, math.ceil(n_energies / ncols))
    return nrows, ncols


def _plot_spill_path(ax, path: SpillPath, color: str) -> None:
    ax.plot(
        path.ic1_x_err,
        path.ic1_y_err,
        color=color,
        linestyle="-",
        linewidth=LINE_LW,
        alpha=LINE_ALPHA,
        solid_capstyle="round",
    )
    ax.plot(
        path.ic2_x_err,
        path.ic2_y_err,
        color=color,
        linestyle=":",
        linewidth=LINE_LW,
        alpha=LINE_ALPHA,
        solid_capstyle="round",
    )


def _draw_reference_circle(ax) -> None:
    ax.add_patch(
        Circle(
            (0, 0),
            REF_CIRCLE_RADIUS_MM,
            fill=False,
            edgecolor="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            zorder=0,
        )
    )


def _style_energy_axis(ax, energy: float) -> None:
    _draw_reference_circle(ax)
    ax.axhline(0, **REFLINE_KW)
    ax.axvline(0, **REFLINE_KW)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{energy:g} MeV", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(**GRID_KW)


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot per-energy IC1/IC2 position error motion paths for each session."""
    if not session_ids:
        print("No sessions selected")
        return

    bg_subtract = settings.bg_subtract if settings else False
    session_data: dict[str, dict[float, list[SpillPath]]] = {}
    for sid in session_ids:
        paths = _load_session_spill_paths(sid, base_dir, bg_subtract=bg_subtract)
        if paths is not None:
            session_data[sid] = paths

    if not session_data:
        print("No valid spill error path data found for any session")
        return

    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    all_energies: set[float] = set()
    for paths in session_data.values():
        all_energies.update(paths.keys())
    energies = sorted(all_energies)

    nrows, ncols = _grid_shape(len(energies))
    fig, axes = view_grid(
        nrows, ncols, cell_w=CELL_WIDTH_IN, cell_h=CELL_HEIGHT_IN
    )

    for idx, energy in enumerate(energies):
        ax = axes[idx // ncols][idx % ncols]
        _style_energy_axis(ax, energy)
        for sid, color in zip(loaded_ids, colors):
            for path in session_data[sid].get(energy, []):
                _plot_spill_path(ax, path, color)

    for idx in range(len(energies), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    ic_handles = [
        Line2D([0], [0], color="0.35", linestyle="-", linewidth=LINE_LW, label="IC1"),
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=LINE_LW, label="IC2"),
    ]
    axes[0][0].legend(
        handles=ic_handles,
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )

    finish_view(
        fig,
        "Beam Error Motion vs Energy (spill paths)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )
    plt.show()
