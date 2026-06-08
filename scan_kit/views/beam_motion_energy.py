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

from ..common import (
    C_IC1_X_POS,
    C_IC1_Y_POS,
    C_IC2_X_POS,
    C_IC2_Y_POS,
    C_LAYER_ID,
    C_SPOT_NO,
    C_X_POSITION,
    C_Y_POSITION,
    DEFAULT_SESSION_COLORS,
    GRID_KW,
    REFLINE_KW,
    apply_tight_layout,
    detect_beam_on_mask,
    detect_spill_segments,
    resolve_concept_column,
    set_view_header,
    subtract_background_frames,
)
from ..common.schema import POSITION_KEY_G2, POSITION_KEY_G3, resolve_column_name
from ..common.session_source import load_session_csv, load_session_timeslice_device_units
from .timeslice_replay_common import load_energy_by_layer, resolve_col

_log = logging.getLogger(__name__)

MAX_GRID_COLS = 10
MAX_GRID_ROWS = 8
CELL_WIDTH_IN = 1.3
CELL_HEIGHT_IN = 1.1
HEADER_HEIGHT_IN = 0.8

LINE_LW = 0.7
LINE_ALPHA = 0.75

# G2 timeslice exports precomputed position error (mm).
_G2_DIRECT_ERROR = (
    ("ic1_x_err", "position_err_X"),
    ("ic1_y_err", "position_err_Y"),
    ("ic2_x_err", "position_err_X2"),
    ("ic2_y_err", "position_err_Y2"),
)

# G3 timeslice: processed position minus per-IC target (both mm, same frame).
_G3_MEASURED_TARGET = (
    ("ic1_x", "r_ic1_x_position", "ic1_position_x_target"),
    ("ic1_y", "r_ic1_y_position", "ic1_position_y_target"),
    ("ic2_x", "r_ic2_x_position", "ic2_position_x_target"),
    ("ic2_y", "r_ic2_y_position", "ic2_position_y_target"),
)


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
class _DeltaErrorSource:
    mode: Literal["delta"]
    measured: dict[str, str]
    nominal: dict[str, str]


_ErrorSource = _DirectErrorSource | _DeltaErrorSource


def _resolve_named_columns(columns, pairs: tuple[tuple[str, str], ...]) -> dict[str, str] | None:
    resolved: dict[str, str] = {}
    for label, name in pairs:
        col = resolve_column_name(columns, name)
        if col is None:
            return None
        resolved[label] = col
    return resolved


def _resolve_concept_positions(columns) -> dict[str, str] | None:
    pos_cols: dict[str, str] = {}
    for pos_key in (POSITION_KEY_G3, POSITION_KEY_G2):
        for concept, label in [
            (C_IC1_X_POS, "ic1_x"),
            (C_IC1_Y_POS, "ic1_y"),
            (C_IC2_X_POS, "ic2_x"),
            (C_IC2_Y_POS, "ic2_y"),
        ]:
            resolved = resolve_concept_column(columns, concept, position_key=pos_key)
            if resolved and label not in pos_cols:
                pos_cols[label] = resolved
        if len(pos_cols) == 4:
            return pos_cols
    return None


def _resolve_plan_columns(columns) -> dict[str, str] | None:
    col_x = resolve_col(columns, C_X_POSITION)
    col_y = resolve_col(columns, C_Y_POSITION)
    if col_x is None or col_y is None:
        return None
    return {"x": col_x, "y": col_y}


def _resolve_error_source(columns) -> _ErrorSource | None:
    direct = _resolve_named_columns(columns, _G2_DIRECT_ERROR)
    if direct is not None:
        return _DirectErrorSource("direct", direct)

    pairs = _G3_MEASURED_TARGET
    measured: dict[str, str] = {}
    nominal: dict[str, str] = {}
    for label, meas_name, nom_name in pairs:
        meas_col = resolve_column_name(columns, meas_name)
        nom_col = resolve_column_name(columns, nom_name)
        if meas_col is None or nom_col is None:
            measured.clear()
            break
        measured[label] = meas_col
        nominal[label] = nom_col
    if len(measured) == 4:
        return _DeltaErrorSource("delta", measured, nominal)

    pos_cols = _resolve_concept_positions(columns)
    plan_cols = _resolve_plan_columns(columns)
    if pos_cols is not None and plan_cols is not None:
        return _DeltaErrorSource("delta", pos_cols, plan_cols)

    if pos_cols is not None:
        return _DeltaErrorSource("delta", pos_cols, {})

    return None


def _load_plan_by_spot(src) -> dict[int, tuple[float, float]] | None:
    input_map = load_session_csv(src, "input_map.csv")
    if input_map is None:
        return None
    col_spot = resolve_col(input_map.columns, C_SPOT_NO)
    col_x = resolve_col(input_map.columns, C_X_POSITION)
    col_y = resolve_col(input_map.columns, C_Y_POSITION)
    if col_spot is None or col_x is None or col_y is None:
        return None
    spot_nos = input_map[col_spot].values.astype(int)
    plan_x = input_map[col_x].values.astype(float)
    plan_y = input_map[col_y].values.astype(float)
    return {
        int(sno): (float(x), float(y))
        for sno, x, y in zip(spot_nos, plan_x, plan_y, strict=True)
    }


def _lookup_plan_by_spot(
    plan_by_spot: dict[int, tuple[float, float]],
    spot_nos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    plan_x = np.empty(len(spot_nos), dtype=float)
    plan_y = np.empty(len(spot_nos), dtype=float)
    for i, sno in enumerate(spot_nos.astype(int)):
        entry = plan_by_spot.get(int(sno))
        if entry is None:
            plan_x[i] = np.nan
            plan_y[i] = np.nan
        else:
            plan_x[i], plan_y[i] = entry
    return plan_x, plan_y


def _sanitize_error(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float, copy=True)
    out[~np.isfinite(out)] = np.nan
    # G3 invalid register / G2 off-scale sentinels.
    out[np.abs(out) > 128] = np.nan
    return out


def _slice_errors(
    df,
    source: _ErrorSource,
    start: int,
    end: int,
    *,
    spot_col: str | None,
    plan_by_spot: dict[int, tuple[float, float]] | None,
) -> SpillPath | None:
    if source.mode == "direct":
        ic1_x_err = _sanitize_error(df[source.columns["ic1_x_err"]].values[start:end])
        ic1_y_err = _sanitize_error(df[source.columns["ic1_y_err"]].values[start:end])
        ic2_x_err = _sanitize_error(df[source.columns["ic2_x_err"]].values[start:end])
        ic2_y_err = _sanitize_error(df[source.columns["ic2_y_err"]].values[start:end])
    else:
        measured = source.measured
        nominal = source.nominal
        if {"x", "y"} <= nominal.keys():
            plan_x = df[nominal["x"]].values[start:end].astype(float)
            plan_y = df[nominal["y"]].values[start:end].astype(float)
        elif spot_col is not None and plan_by_spot is not None:
            plan_x, plan_y = _lookup_plan_by_spot(plan_by_spot, df[spot_col].values[start:end])
        elif all(k in nominal for k in ("ic1_x", "ic1_y", "ic2_x", "ic2_y")):
            plan_x = plan_y = None
        else:
            return None

        ic1_x = df[measured["ic1_x"]].values[start:end].astype(float)
        ic1_y = df[measured["ic1_y"]].values[start:end].astype(float)
        ic2_x = df[measured["ic2_x"]].values[start:end].astype(float)
        ic2_y = df[measured["ic2_y"]].values[start:end].astype(float)

        if plan_x is not None:
            ic1_x_err = _sanitize_error(ic1_x - plan_x)
            ic1_y_err = _sanitize_error(ic1_y - plan_y)
            ic2_x_err = _sanitize_error(ic2_x - plan_x)
            ic2_y_err = _sanitize_error(ic2_y - plan_y)
        else:
            ic1_x_err = _sanitize_error(ic1_x - df[nominal["ic1_x"]].values[start:end].astype(float))
            ic1_y_err = _sanitize_error(ic1_y - df[nominal["ic1_y"]].values[start:end].astype(float))
            ic2_x_err = _sanitize_error(ic2_x - df[nominal["ic2_x"]].values[start:end].astype(float))
            ic2_y_err = _sanitize_error(ic2_y - df[nominal["ic2_y"]].values[start:end].astype(float))

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
    loaded = load_energy_by_layer(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer = loaded

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        subtract_background_frames(frames)

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    error_source = _resolve_error_source(df0.columns)
    if ts_layer is None or error_source is None:
        return None

    spot_col = resolve_col(df0.columns, C_SPOT_NO)
    plan_by_spot = None
    if (
        error_source.mode == "delta"
        and {"x", "y"} <= error_source.nominal.keys()
    ):
        plan_by_spot = None
    elif error_source.mode == "delta" and not (
        {"ic1_x", "ic1_y", "ic2_x", "ic2_y"} <= error_source.nominal.keys()
    ):
        plan_by_spot = _load_plan_by_spot(src)
        if spot_col is None or plan_by_spot is None:
            return None

    by_energy: dict[float, list[SpillPath]] = defaultdict(list)

    for df in frames:
        layer_id = df[ts_layer].iloc[0]
        energy = energy_by_layer.get(layer_id)
        if energy is None:
            continue

        beam_on = detect_beam_on_mask(df)
        if beam_on is None:
            continue

        layer_source = _resolve_error_source(df.columns) or error_source
        layer_spot_col = resolve_col(df.columns, C_SPOT_NO) or spot_col

        for start, end in detect_spill_segments(beam_on):
            path = _slice_errors(
                df,
                layer_source,
                start,
                end,
                spot_col=layer_spot_col,
                plan_by_spot=plan_by_spot,
            )
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


def _style_energy_axis(ax, energy: float) -> None:
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
        for sid, color in zip(loaded_ids, colors):
            for path in session_data[sid].get(energy, []):
                _plot_spill_path(ax, path, color)
        _style_energy_axis(ax, energy)

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

    if nrows > 0 and ncols > 0:
        fig.supxlabel("X Error (mm)", fontsize=9)
        fig.supylabel("Y Error (mm)", fontsize=9)

    set_view_header(
        fig,
        "Beam Error Motion vs Energy (spill paths)",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    apply_tight_layout()
    plt.show()
