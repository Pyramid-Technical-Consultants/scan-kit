"""Per-energy IC1/IC2 position error motion paths from timeslice data by spill."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ..common import (
    C_LAYER_ID,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    apply_tight_layout,
    detect_beam_on_mask,
    detect_spill_segments,
    set_view_header,
    subtract_background_frames,
)
from ..common.g3_timeslice_position import (
    G3PositionTargetColumns,
    g3_position_error_frame_arrays,
    resolve_g3_position_target_columns,
)
from ..common.schema import resolve_column_name
from ..common.session_source import load_session_timeslice_device_units
from .timeslice_replay_common import load_energy_lookups, resolve_col, resolve_frame_energy

_log = logging.getLogger(__name__)

MAX_GRID_COLS = 10
MAX_GRID_ROWS = 8
CELL_WIDTH_IN = 1.3
CELL_HEIGHT_IN = 1.1
HEADER_HEIGHT_IN = 0.8

LINE_LW = 0.7
LINE_ALPHA = 0.75
REF_CIRCLE_RADIUS_MM = 1.0

_G2_DIRECT_ERROR = (
    ("ic1_x_err", "position_err_X"),
    ("ic1_y_err", "position_err_Y"),
    ("ic2_x_err", "position_err_X2"),
    ("ic2_y_err", "position_err_Y2"),
)

_TIMESLICE_COLS = [
    C_LAYER_ID,
    "rci_in_trigger",
    "r_beamOk",
    *(name for _, name in _G2_DIRECT_ERROR),
    "r_ic1_x_position",
    "ic1_position_x_target",
    "r_ic1_y_position",
    "ic1_position_y_target",
    "r_ic2_x_position",
    "ic2_position_x_target",
    "r_ic2_y_position",
    "ic2_position_y_target",
]


@dataclass(frozen=True)
class SpillPath:
    ic1_x_err: np.ndarray
    ic1_y_err: np.ndarray
    ic2_x_err: np.ndarray
    ic2_y_err: np.ndarray


@dataclass(frozen=True)
class _DirectErrorSource:
    mode: Literal["direct"]
    columns: dict[str, str]


@dataclass(frozen=True)
class _G3PositionTargetSource:
    mode: Literal["g3_position_target"]
    columns: G3PositionTargetColumns


_ErrorSource = _DirectErrorSource | _G3PositionTargetSource


def _resolve_named_columns(columns, pairs: tuple[tuple[str, str], ...]) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    for label, name in pairs:
        col = resolve_column_name(columns, name)
        if col is None:
            return None
        resolved[label] = col
    return resolved


def _resolve_error_source(columns) -> _ErrorSource | None:
    direct = _resolve_named_columns(columns, _G2_DIRECT_ERROR)
    if direct is not None:
        return _DirectErrorSource("direct", direct)

    g3_cols = resolve_g3_position_target_columns(columns)
    if g3_cols is not None:
        return _G3PositionTargetSource("g3_position_target", g3_cols)

    return None


def _sanitize_error(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    out[np.abs(out) > 128] = np.nan
    return out


def _frame_error_arrays(
    df,
    source: _ErrorSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if source.mode == "direct":
        return (
            _sanitize_error(df[source.columns["ic1_x_err"]].values),
            _sanitize_error(df[source.columns["ic1_y_err"]].values),
            _sanitize_error(df[source.columns["ic2_x_err"]].values),
            _sanitize_error(df[source.columns["ic2_y_err"]].values),
        )

    return g3_position_error_frame_arrays(df, source.columns)


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

    frames = load_session_timeslice_device_units(src, usecols=_TIMESLICE_COLS)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    error_source = _resolve_error_source(df0.columns)
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

        frame_errors = _frame_error_arrays(df, error_source)
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
    figsize = (CELL_WIDTH_IN * ncols, CELL_HEIGHT_IN * nrows + HEADER_HEIGHT_IN)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

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

    set_view_header(
        fig,
        "Beam Error Motion vs Energy (spill paths)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    apply_tight_layout()
    plt.show()
