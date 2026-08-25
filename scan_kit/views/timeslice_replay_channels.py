"""Channel catalog, unified loader, and config builders for timeslice replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..common import (
    C_BEAM_CURRENT,
    C_IC1_X_POS_RAW,
    C_IC1_Y_POS_RAW,
    C_IC2_X_POS_RAW,
    C_IC2_Y_POS_RAW,
    C_LAYER_ID,
    C_MAG_FIELD_X,
    C_MAG_FIELD_Y,
    resolve_concept_column,
    transform,
)
from ..common.schema import POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW
from ..common.session_source import load_session_timeslice_device_units
from ..common.timeslice_energy import load_energy_lookups
from ..common.timeslice_ic_current import resolve_ic_current_columns, sum_ic3_current
from ..common.timeslice_sigma import (
    frame_timeslice_sigma_arrays,
    resolve_timeslice_sigma_source,
)
from .beam_off_rampdown import detect_beam_off_edges
from .timeslice_replay_common import (
    build_digital_signals,
    derive_current_from_dose,
    detect_digital_columns,
    resolve_col,
    resolve_frame_energy,
    resolve_ic_scan_total_dose_columns,
)
from .timeslice_replay_ui import ScatterSpec, TimesliceReplayConfig, TraceSpec

FAMILY_IC = "IC Current"
FAMILY_DDOSE = "dDose/dt"
FAMILY_SIGMA = "Sigma"
FAMILY_FIELD = "Magnetic Field"

PRESET_IC_CURRENT = "ic_current"
PRESET_DDOSE = "ddose"
PRESET_SIGMA = "sigma"
PRESET_FIELD = "field"

PRESET_LABELS: dict[str, str] = {
    PRESET_IC_CURRENT: "IC Current",
    PRESET_DDOSE: "dDose/dt",
    PRESET_SIGMA: "Sigma",
    PRESET_FIELD: "Field",
}


@dataclass(frozen=True)
class ChannelDef:
    """Selectable plot channel."""

    key: str
    label: str
    color: str
    family: str
    linewidth: float = 0.5
    beam_off_edges: bool = False


CHANNEL_DEFS: tuple[ChannelDef, ...] = (
    ChannelDef("ic1", "IC1", "#1f77b4", FAMILY_IC, beam_off_edges=True),
    ChannelDef("ic2", "IC2", "#d62728", FAMILY_IC, beam_off_edges=True),
    ChannelDef("ic3", "IC3 (A+B+C+D)", "#2ca02c", FAMILY_IC, beam_off_edges=True),
    ChannelDef(
        "ic1_ddose", "IC1 dDose/dt", "#1f77b4", FAMILY_DDOSE,
        linewidth=0.6, beam_off_edges=True,
    ),
    ChannelDef(
        "ic2_ddose", "IC2 dDose/dt", "#d62728", FAMILY_DDOSE,
        linewidth=0.6, beam_off_edges=True,
    ),
    ChannelDef(
        "ic3_ddose", "IC3 dDose/dt", "#2ca02c", FAMILY_DDOSE,
        linewidth=0.6, beam_off_edges=True,
    ),
    ChannelDef(
        "sigma_ic1_x", "IC1 σx (mm)", "#1f77b4", FAMILY_SIGMA, beam_off_edges=True,
    ),
    ChannelDef("sigma_ic1_y", "IC1 σy (mm)", "#aec7e8", FAMILY_SIGMA),
    ChannelDef("sigma_ic2_x", "IC2 σx (mm)", "#d62728", FAMILY_SIGMA),
    ChannelDef("sigma_ic2_y", "IC2 σy (mm)", "#ff9896", FAMILY_SIGMA),
    ChannelDef("bx", "Bx (G)", "#1f77b4", FAMILY_FIELD),
    ChannelDef("by", "By (G)", "#d62728", FAMILY_FIELD),
)

CHANNEL_BY_KEY: dict[str, ChannelDef] = {c.key: c for c in CHANNEL_DEFS}

PRESET_CHANNELS: dict[str, tuple[str, ...]] = {
    PRESET_IC_CURRENT: ("ic1", "ic2", "ic3"),
    PRESET_DDOSE: ("ic1_ddose", "ic2_ddose", "ic3_ddose"),
    PRESET_SIGMA: ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y"),
    PRESET_FIELD: ("bx", "by"),
}

_ANALOG_SIGNAL_KEYS = frozenset(CHANNEL_BY_KEY)
_SIGMA_KEYS = ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y")
_POS_KEYS = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")


def channel_defs_by_family() -> list[tuple[str, list[ChannelDef]]]:
    """Return catalog grouped in display order."""
    order = (FAMILY_IC, FAMILY_DDOSE, FAMILY_SIGMA, FAMILY_FIELD)
    grouped: dict[str, list[ChannelDef]] = {name: [] for name in order}
    for channel in CHANNEL_DEFS:
        grouped[channel.family].append(channel)
    return [(name, grouped[name]) for name in order if grouped[name]]


def available_channel_keys(session_data: dict[str, dict]) -> set[str]:
    """Keys present with data in at least one loaded session."""
    available: set[str] = set()
    for data in session_data.values():
        for key in _ANALOG_SIGNAL_KEYS:
            arr = data.get(key)
            if arr is not None and len(arr):
                available.add(key)
    return available


def filter_available_keys(
    keys: Sequence[str],
    available: Iterable[str],
) -> list[str]:
    avail = set(available)
    return [key for key in keys if key in avail]


def default_selected_keys(available: set[str]) -> list[str]:
    """Pick the first preset that has any available channels."""
    for preset_id in (
        PRESET_IC_CURRENT, PRESET_DDOSE, PRESET_SIGMA, PRESET_FIELD,
    ):
        selected = filter_available_keys(PRESET_CHANNELS[preset_id], available)
        if selected:
            return selected
    return []


def load_session_timeline_catalog(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict | None:
    """Load all available timeslice channel families into one session dict."""
    loaded = load_energy_lookups(session_id, base_dir)
    if loaded is None:
        return None
    src, energy_by_layer, energy_by_idx = loaded

    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    if bg_subtract:
        from ..common import subtract_background_frames

        subtract_background_frames(frames)

    df0 = frames[0]
    ts_layer = resolve_col(df0.columns, C_LAYER_ID)
    if ts_layer is None:
        return None

    ic_cols = resolve_ic_current_columns(df0.columns)
    has_ic = ic_cols is not None
    has_ic3 = bool(ic_cols and ic_cols.ic3_parts)

    dose_cols = resolve_ic_scan_total_dose_columns(df0.columns)
    ts_dose1 = dose_cols["ic1"]
    ts_dose2 = dose_cols["ic2"]
    ts_dose3 = dose_cols["ic3"]
    has_ddose = bool(ts_dose1 and ts_dose2)
    has_ddose3 = has_ddose and ts_dose3 is not None

    sigma_source = resolve_timeslice_sigma_source(df0.columns)
    has_sigma = sigma_source is not None

    ts_bx = resolve_col(df0.columns, C_MAG_FIELD_X)
    ts_by = resolve_col(df0.columns, C_MAG_FIELD_Y)
    has_field = bool(ts_bx and ts_by)

    if not any((has_ic, has_ddose, has_sigma, has_field)):
        return None

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    has_beam = ts_beam is not None

    pos_cols: dict[str, str] = {}
    for pos_key in (POSITION_KEY_G3_RAW, POSITION_KEY_G2_RAW):
        for concept, label in (
            (C_IC1_X_POS_RAW, "ic1_x"),
            (C_IC1_Y_POS_RAW, "ic1_y"),
            (C_IC2_X_POS_RAW, "ic2_x"),
            (C_IC2_Y_POS_RAW, "ic2_y"),
        ):
            resolved = resolve_concept_column(
                df0.columns, concept, position_key=pos_key,
            )
            if resolved and label not in pos_cols:
                pos_cols[label] = resolved
        if len(pos_cols) == 4:
            break
    has_positions = len(pos_cols) == 4

    digital_cols = detect_digital_columns(df0.columns)
    digital_parts: dict[str, list[np.ndarray]] = {col: [] for col, _ in digital_cols}

    parts: dict[str, list[np.ndarray]] = {
        "ic1": [],
        "ic2": [],
        "ic3": [],
        "ic1_ddose": [],
        "ic2_ddose": [],
        "ic3_ddose": [],
        **{k: [] for k in _SIGMA_KEYS},
        "bx": [],
        "by": [],
        "beam": [],
        **{k: [] for k in _POS_KEYS},
    }
    energy_parts: list[np.ndarray] = []
    layer_boundaries: list[tuple[int, float]] = []
    edge_indices: dict[str, list[int]] = {
        "ic1": [],
        "ic2": [],
        "ic3": [],
        "ic1_ddose": [],
        "ic2_ddose": [],
        "ic3_ddose": [],
        "sigma_ic1_x": [],
    }
    offset = 0

    for frame_i, df in enumerate(frames):
        n = len(df)
        energy = resolve_frame_energy(
            df,
            frame_i,
            energy_by_layer=energy_by_layer,
            energy_by_idx=energy_by_idx,
            layer_col=ts_layer,
        )
        if energy is None:
            energy = 0.0

        if has_ic:
            ic1_vals = df[ic_cols.ic1].values.astype(float)
            ic2_vals = df[ic_cols.ic2].values.astype(float)
            parts["ic1"].append(ic1_vals)
            parts["ic2"].append(ic2_vals)
            for key, vals in (("ic1", ic1_vals), ("ic2", ic2_vals)):
                edges = detect_beam_off_edges(vals)
                edge_indices[key].extend((edges + offset).tolist())
            if has_ic3:
                ic3_vals = sum_ic3_current(df, ic_cols.ic3_parts)
                parts["ic3"].append(ic3_vals)
                edges = detect_beam_off_edges(ic3_vals)
                edge_indices["ic3"].extend((edges + offset).tolist())

        if has_ddose:
            d1 = derive_current_from_dose(df[ts_dose1].values.astype(float))
            d2 = derive_current_from_dose(df[ts_dose2].values.astype(float))
            parts["ic1_ddose"].append(d1)
            parts["ic2_ddose"].append(d2)
            for key, vals in (("ic1_ddose", d1), ("ic2_ddose", d2)):
                edges = detect_beam_off_edges(vals)
                edge_indices[key].extend((edges + offset).tolist())
            if has_ddose3:
                d3 = derive_current_from_dose(df[ts_dose3].values.astype(float))
                parts["ic3_ddose"].append(d3)
                edges = detect_beam_off_edges(d3)
                edge_indices["ic3_ddose"].extend((edges + offset).tolist())

        if has_sigma:
            frame_sigmas = frame_timeslice_sigma_arrays(df, sigma_source)
            if frame_sigmas is None:
                nan = np.full(n, np.nan)
                for key in _SIGMA_KEYS:
                    parts[key].append(nan)
            else:
                s_ic1_x, s_ic1_y, s_ic2_x, s_ic2_y = frame_sigmas
                parts["sigma_ic1_x"].append(s_ic1_x)
                parts["sigma_ic1_y"].append(s_ic1_y)
                parts["sigma_ic2_x"].append(s_ic2_x)
                parts["sigma_ic2_y"].append(s_ic2_y)
            if has_ic:
                edges = detect_beam_off_edges(df[ic_cols.ic1].values.astype(float))
                edge_indices["sigma_ic1_x"].extend((edges + offset).tolist())

        if has_field:
            parts["bx"].append(df[ts_bx].values.astype(float))
            parts["by"].append(df[ts_by].values.astype(float))

        if has_beam:
            parts["beam"].append(df[ts_beam].values.astype(float))
        for col, _ in digital_cols:
            if col in df.columns:
                digital_parts[col].append(df[col].values.astype(float))
            else:
                digital_parts[col].append(np.zeros(n))
        if has_positions:
            for label, col in pos_cols.items():
                parts[label].append(df[col].values.astype(float))

        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    result: dict = {
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_ic3": has_ic3,
        "has_beam": has_beam,
        "has_positions": has_positions,
        "has_sigma": has_sigma,
        "has_field": has_field,
        "has_ddose": has_ddose,
        "energy": np.concatenate(energy_parts),
        "beam_off_edges": {
            k: np.asarray(v, dtype=int) for k, v in edge_indices.items() if v
        },
        "digital": build_digital_signals(digital_parts, digital_cols),
    }

    def _store(key: str) -> None:
        if parts[key]:
            result[key] = np.concatenate(parts[key])

    if has_ic:
        _store("ic1")
        _store("ic2")
        if has_ic3:
            _store("ic3")
    if has_ddose:
        _store("ic1_ddose")
        _store("ic2_ddose")
        if has_ddose3:
            _store("ic3_ddose")
    if has_sigma:
        for key in _SIGMA_KEYS:
            _store(key)
    if has_field:
        _store("bx")
        _store("by")
        result["b_mag"] = np.hypot(result["bx"], result["by"])
    if has_beam:
        _store("beam")
    if has_positions:
        result["ic1_x"] = transform.remap(
            np.concatenate(parts["ic1_x"]), *transform.IC1_X_MAP,
        )
        result["ic1_y"] = transform.remap(
            np.concatenate(parts["ic1_y"]), *transform.IC1_Y_MAP,
        )
        result["ic2_x"] = transform.remap(
            np.concatenate(parts["ic2_x"]), *transform.IC2_X_MAP,
        )
        result["ic2_y"] = transform.remap(
            np.concatenate(parts["ic2_y"]), *transform.IC2_Y_MAP,
        )
        pos_limit = transform.IC_MM_MAX
        for key in _POS_KEYS:
            arr = result[key]
            arr[np.abs(arr) > pos_limit] = np.nan

    return result


def load_sessions_catalog(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    bg_subtract: bool = False,
) -> dict[str, dict]:
    """Load catalog data for each session that has usable timeslice channels."""
    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = load_session_timeline_catalog(
            sid, base_dir, bg_subtract=bg_subtract,
        )
        if data is not None:
            session_data[sid] = data
    return session_data


def _families_of(keys: Sequence[str]) -> set[str]:
    return {CHANNEL_BY_KEY[k].family for k in keys if k in CHANNEL_BY_KEY}


def _timeline_for_selection(
    selected: Sequence[str],
    session_data: dict[str, dict],
) -> tuple[str, str]:
    if not selected:
        return "ic1", "Signal"

    families = _families_of(selected)
    if families == {FAMILY_FIELD}:
        if any(d.get("b_mag") is not None for d in session_data.values()):
            return "b_mag", "|B| (G)"
        first = CHANNEL_BY_KEY[selected[0]]
        return first.key, first.label

    first = CHANNEL_BY_KEY[selected[0]]
    return first.key, first.label


def _scatter_for_selection(
    selected: Sequence[str],
    session_data: dict[str, dict],
) -> ScatterSpec:
    if not selected:
        return ScatterSpec(mode="none")

    families = _families_of(selected)
    if len(families) != 1:
        return ScatterSpec(mode="none")

    family = next(iter(families))
    if family == FAMILY_FIELD:
        return ScatterSpec(
            mode="single",
            x_key="bx",
            y_key="by",
            title="B vector (G)",
            xlabel="Bx (G)",
            ylabel="By (G)",
        )

    if family == FAMILY_SIGMA:
        return ScatterSpec(
            mode="per_trace",
            per_trace_xy={
                "sigma_ic1_x": ("sigma_ic1_x", "sigma_ic1_y"),
                "sigma_ic1_y": ("sigma_ic1_x", "sigma_ic1_y"),
                "sigma_ic2_x": ("sigma_ic2_x", "sigma_ic2_y"),
                "sigma_ic2_y": ("sigma_ic2_x", "sigma_ic2_y"),
            },
            per_trace_title_suffix=" σ (mm)",
            missing_label="No sigma data",
        )

    if family in {FAMILY_IC, FAMILY_DDOSE}:
        if not any(d.get("has_positions") for d in session_data.values()):
            return ScatterSpec(mode="none")
        per_trace: dict[str, tuple[str, str]] = {}
        for key in selected:
            if key in {"ic1", "ic1_ddose"}:
                per_trace[key] = ("ic1_x", "ic1_y")
            elif key in {"ic2", "ic2_ddose"}:
                per_trace[key] = ("ic2_x", "ic2_y")
        if not per_trace:
            return ScatterSpec(mode="none")
        return ScatterSpec(mode="per_trace", per_trace_xy=per_trace)

    return ScatterSpec(mode="none")


def build_replay_config(
    selected_keys: Sequence[str],
    session_data: dict[str, dict],
    *,
    peer_overlay: bool = False,
    show_digital: bool = True,
    show_beam_twin: bool = True,
    beam_off_edges: bool = True,
    title: str = "Timeslice Replay",
) -> TimesliceReplayConfig:
    """Build a renderer config from the current channel selection."""
    available = available_channel_keys(session_data)
    selected = filter_available_keys(selected_keys, available)
    traces = tuple(
        TraceSpec(
            key=CHANNEL_BY_KEY[key].key,
            label=CHANNEL_BY_KEY[key].label,
            color=CHANNEL_BY_KEY[key].color,
            linewidth=CHANNEL_BY_KEY[key].linewidth,
            beam_off_edges=beam_off_edges and CHANNEL_BY_KEY[key].beam_off_edges,
        )
        for key in selected
    )
    timeline_key, timeline_ylabel = _timeline_for_selection(selected, session_data)
    scatter = _scatter_for_selection(selected, session_data)
    show_scatter = scatter.mode != "none"
    n_traces = max(1, len(traces))
    fig_h = 8 + 1.0 * max(0, n_traces - 2)

    return TimesliceReplayConfig(
        title=title,
        no_data_message="No valid timeslice data found for any session",
        traces=traces,
        timeline_key=timeline_key,
        timeline_ylabel=timeline_ylabel,
        figsize=(22 if show_scatter else 18, fig_h),
        scatter=scatter,
        peer_overlay=peer_overlay and len(session_data) <= 1,
        show_digital=show_digital,
        show_beam_twin=show_beam_twin,
    )
