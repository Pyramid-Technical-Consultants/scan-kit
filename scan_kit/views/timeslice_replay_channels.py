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
from ..data.timeline_channels import (
    FAMILY_DDOSE,
    FAMILY_FIELD,
    FAMILY_IC,
    FAMILY_IC12_POS_DIFF,
    FAMILY_POSITION,
    FAMILY_POSITION_ERROR,
    FAMILY_SIGMA,
    FAMILY_SIGMA_ERROR,
    REPLAY_CHANNEL_SPECS,
    TIMELINE_CHANNEL_BY_KEY,
    available_channel_keys as _shared_available_channel_keys,
)
from ..data.types import (
    DATA_SOURCE_TIMESLICE_ISO,
    DataSourceKind,
    data_source_is_timeslice,
    option_key,
)
from .timeslice_replay_catalog import (
    METRIC_BY_ID,
    METRIC_DDOSE,
    METRIC_MAG_FIELD,
    PRESET_CHANNELS,
    PRESET_DDOSE,
    PRESET_FIELD,
    PRESET_IC_CURRENT,
    PRESET_LABELS,
    PRESET_SIGMA,
    REPLAY_REGISTRY_SOURCE_IDS,
    VIEW_OPTIONS,
    channel_keys_for_metric,
    custom_option_keys,
    metric_for_option,
    reference_frame_for_source,
)
from .timeslice_replay_common import (
    build_digital_signals,
    derive_current_from_dose,
    detect_digital_columns,
    resolve_col,
    resolve_frame_energy,
    resolve_ic_scan_total_dose_columns,
)
from .timeslice_replay_ui import ScatterSpec, TimesliceReplayConfig, TraceSpec
from .unified_catalog import is_option_available


@dataclass(frozen=True)
class ChannelDef:
    """Selectable plot channel."""

    key: str
    label: str
    color: str
    family: str
    linewidth: float = 0.5
    beam_off_edges: bool = False


CHANNEL_DEFS: tuple[ChannelDef, ...] = tuple(
    ChannelDef(
        key=spec.key,
        label=spec.label,
        color=spec.replay_color or "#1f77b4",
        family=spec.family,
        linewidth=spec.replay_linewidth,
        beam_off_edges=spec.beam_off_edges,
    )
    for spec in REPLAY_CHANNEL_SPECS
)

CHANNEL_BY_KEY: dict[str, ChannelDef] = {c.key: c for c in CHANNEL_DEFS}

_ANALOG_SIGNAL_KEYS = frozenset(TIMELINE_CHANNEL_BY_KEY)
_SIGMA_KEYS = ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y")
_SIGMA_ERR_KEYS = (
    "sigma_ic1_x_err",
    "sigma_ic1_y_err",
    "sigma_ic2_x_err",
    "sigma_ic2_y_err",
)
_POS_KEYS = ("ic1_x", "ic1_y", "ic2_x", "ic2_y")
_POS_ERR_KEYS = ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err")
_IC12_DIFF_KEYS = ("ic12_x_diff", "ic12_y_diff")
_CATALOG_IC_KEYS = frozenset({"ic1", "ic2", "ic3"})
_CATALOG_DDOSE_KEYS = frozenset({"ic1_ddose", "ic2_ddose", "ic3_ddose"})
_CATALOG_SIGMA_KEYS = frozenset(_SIGMA_KEYS)
_CATALOG_SIGMA_ERR_KEYS = frozenset(_SIGMA_ERR_KEYS)
_CATALOG_FIELD_KEYS = frozenset({"bx", "by", "b_mag"})
_CATALOG_BEAM_KEYS = frozenset({"beam"})
_CATALOG_POS_KEYS = frozenset(_POS_KEYS)
_CATALOG_POS_ERR_KEYS = frozenset(_POS_ERR_KEYS)
_CATALOG_IC12_DIFF_KEYS = frozenset(_IC12_DIFF_KEYS)


def _catalog_wants_family(
    channel_keys: frozenset[str] | None,
    family_keys: frozenset[str],
) -> bool:
    if channel_keys is None:
        return True
    return bool(family_keys & channel_keys)


def probe_session_timeline_flags(session_id: str, base_dir: str) -> dict[str, bool] | None:
    """Return file-level timeslice channel flags from the first frame (no array load)."""
    from ..common.session_source import resolve_session_source

    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    frames = load_session_timeslice_device_units(src, max_frames=1)
    if not frames:
        return None
    df0 = frames[0]

    ic_cols = resolve_ic_current_columns(df0.columns)
    has_ic = ic_cols is not None
    file_has_ic3 = bool(ic_cols and ic_cols.ic3_parts)

    dose_cols = resolve_ic_scan_total_dose_columns(df0.columns)
    ts_dose1 = dose_cols["ic1"]
    ts_dose2 = dose_cols["ic2"]
    ts_dose3 = dose_cols["ic3"]
    file_has_ddose = bool(ts_dose1 and ts_dose2)
    file_has_ddose3 = file_has_ddose and ts_dose3 is not None

    sigma_source = resolve_timeslice_sigma_source(df0.columns)
    file_has_sigma = sigma_source is not None

    ts_bx = resolve_col(df0.columns, C_MAG_FIELD_X)
    ts_by = resolve_col(df0.columns, C_MAG_FIELD_Y)
    file_has_field = bool(ts_bx and ts_by)

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    file_has_beam = ts_beam is not None

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
    file_has_positions = len(pos_cols) == 4

    return {
        "has_ic": has_ic,
        "has_ic3": file_has_ic3,
        "has_ddose": file_has_ddose,
        "has_ddose3": file_has_ddose3,
        "has_sigma": file_has_sigma,
        "has_field": file_has_field,
        "has_beam": file_has_beam,
        "has_positions": file_has_positions,
    }


def channel_defs_by_family() -> list[tuple[str, list[ChannelDef]]]:
    """Return catalog grouped in display order."""
    order = (FAMILY_IC, FAMILY_DDOSE, FAMILY_SIGMA, FAMILY_FIELD)
    grouped: dict[str, list[ChannelDef]] = {name: [] for name in order}
    for channel in CHANNEL_DEFS:
        grouped[channel.family].append(channel)
    return [(name, grouped[name]) for name in order if grouped[name]]


def available_channel_keys(session_data: dict[str, dict]) -> set[str]:
    """Keys present with data in at least one loaded session."""
    return _shared_available_channel_keys(session_data)


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


def probe_replay_option_availability(
    session_ids: Sequence[str],
    base_dir: str,
) -> dict[str, bool]:
    """Return availability for every unified timeslice-replay option."""
    from ..data.availability import probe_sessions

    availability = {
        key: value
        for key, value in probe_sessions(
            session_ids,
            base_dir,
            source_ids=REPLAY_REGISTRY_SOURCE_IDS,
        ).items()
        if data_source_is_timeslice(key.split(":", 1)[0])
    }
    for key in custom_option_keys():
        availability.setdefault(key, False)

    for session_id in session_ids:
        flags = probe_session_timeline_flags(session_id, base_dir)
        if not flags:
            continue
        if flags.get("has_ddose"):
            availability[option_key(DATA_SOURCE_TIMESLICE_ISO, METRIC_DDOSE)] = True
        if flags.get("has_field"):
            availability[option_key(DATA_SOURCE_TIMESLICE_ISO, METRIC_MAG_FIELD)] = True

    for opt in VIEW_OPTIONS:
        availability.setdefault(option_key(opt.source, opt.id), False)
    return availability


def default_metric_selection(
    availability: dict[str, bool],
) -> tuple[str, DataSourceKind, tuple[str, ...]]:
    """Pick the first available metric/source and its default channels."""
    for opt in VIEW_OPTIONS:
        if not is_option_available(availability, opt):
            continue
        metric = METRIC_BY_ID.get(opt.id)
        if metric is None:
            continue
        return opt.id, opt.source, metric.default_channel_keys
    return PRESET_IC_CURRENT, DATA_SOURCE_TIMESLICE_ISO, ("ic1", "ic2")


def load_session_timeline_catalog(
    session_id: str,
    base_dir: str,
    *,
    bg_subtract: bool = False,
    opened: tuple | None = None,
    channel_keys: frozenset[str] | None = None,
    position_reference_frame: str | None = None,
) -> dict | None:
    """Load timeslice channel families into one session dict.

    When *opened* is provided (from :func:`load_session_timeslice_frames`),
    skip re-reading timeslice CSVs.

    When *channel_keys* is set, only extract those catalog keys (frames are
    still read, but unused families are not concatenated).
    """
    session_src = None
    if opened is not None:
        session_src, frames, energy_by_layer, energy_by_idx, ts_layer = opened
        if not frames:
            return None
    else:
        loaded = load_energy_lookups(session_id, base_dir)
        if loaded is None:
            return None
        session_src, energy_by_layer, energy_by_idx = loaded

        frames = load_session_timeslice_device_units(session_src)
        if not frames:
            return None
        if bg_subtract:
            from ..common import subtract_background_frames

            subtract_background_frames(frames)

        df0 = frames[0]
        ts_layer = resolve_col(df0.columns, C_LAYER_ID)
        if ts_layer is None:
            return None

    df0 = frames[0]

    ic_cols = resolve_ic_current_columns(df0.columns)
    file_has_ic = ic_cols is not None
    file_has_ic3 = bool(ic_cols and ic_cols.ic3_parts)

    dose_cols = resolve_ic_scan_total_dose_columns(df0.columns)
    ts_dose1 = dose_cols["ic1"]
    ts_dose2 = dose_cols["ic2"]
    ts_dose3 = dose_cols["ic3"]
    file_has_ddose = bool(ts_dose1 and ts_dose2)
    file_has_ddose3 = file_has_ddose and ts_dose3 is not None

    sigma_source = resolve_timeslice_sigma_source(df0.columns)
    file_has_sigma = sigma_source is not None

    ts_bx = resolve_col(df0.columns, C_MAG_FIELD_X)
    ts_by = resolve_col(df0.columns, C_MAG_FIELD_Y)
    file_has_field = bool(ts_bx and ts_by)

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
    file_has_positions = len(pos_cols) == 4

    from ..data.types import REFERENCE_CHAMBER, REFERENCE_ISO

    ts_beam = resolve_col(df0.columns, C_BEAM_CURRENT)
    file_has_beam = ts_beam is not None

    error_source = None
    from ..common.timeslice_position_error import (
        frame_timeslice_chamber_position_arrays,
        frame_timeslice_error_arrays,
        frame_timeslice_iso_position_arrays,
        resolve_session_timeslice_chamber_position_source,
        resolve_session_timeslice_error_source,
        resolve_session_timeslice_iso_position_source,
    )

    if session_src is not None:
        error_source = resolve_session_timeslice_error_source(session_src, frames)
        iso_position_source = resolve_session_timeslice_iso_position_source(
            session_src, frames,
        )
        chamber_position_source = resolve_session_timeslice_chamber_position_source(
            session_src, frames,
        )
    else:
        iso_position_source = None
        chamber_position_source = None

    file_has_position_error = error_source is not None
    file_has_iso_positions = iso_position_source is not None
    file_has_chamber_positions = chamber_position_source is not None or file_has_positions

    sigma_target_cols = None
    file_has_sigma_error = False
    if file_has_sigma:
        from ..common.timeslice_sigma import _resolve_sigma_target_columns

        sigma_target_cols = _resolve_sigma_target_columns(df0.columns)
        file_has_sigma_error = sigma_target_cols is not None

    file_has_ic12_diff = file_has_iso_positions or file_has_chamber_positions

    if channel_keys is None:
        if not any((file_has_ic, file_has_ddose, file_has_sigma, file_has_field)):
            return None
    else:
        wanted = channel_keys
        has_wanted = (
            (file_has_ic and bool(_CATALOG_IC_KEYS & wanted))
            or (file_has_ddose and bool(_CATALOG_DDOSE_KEYS & wanted))
            or (file_has_sigma and bool(_CATALOG_SIGMA_KEYS & wanted))
            or (file_has_field and bool(_CATALOG_FIELD_KEYS & wanted))
            or (
                (file_has_iso_positions or file_has_chamber_positions or file_has_positions)
                and bool(_CATALOG_POS_KEYS & wanted)
            )
            or (file_has_position_error and bool(_CATALOG_POS_ERR_KEYS & wanted))
            or (file_has_sigma_error and bool(_CATALOG_SIGMA_ERR_KEYS & wanted))
            or (file_has_ic12_diff and bool(_CATALOG_IC12_DIFF_KEYS & wanted))
        )
        if not has_wanted:
            return None

    wants_ic = _catalog_wants_family(channel_keys, _CATALOG_IC_KEYS)
    wants_ddose = _catalog_wants_family(channel_keys, _CATALOG_DDOSE_KEYS)
    wants_sigma = _catalog_wants_family(channel_keys, _CATALOG_SIGMA_KEYS)
    wants_field = _catalog_wants_family(channel_keys, _CATALOG_FIELD_KEYS)
    wants_beam = _catalog_wants_family(channel_keys, _CATALOG_BEAM_KEYS)
    wants_positions = _catalog_wants_family(channel_keys, _CATALOG_POS_KEYS)
    wants_position_error = _catalog_wants_family(channel_keys, _CATALOG_POS_ERR_KEYS)
    wants_sigma_error = _catalog_wants_family(channel_keys, _CATALOG_SIGMA_ERR_KEYS)
    wants_ic12_diff = _catalog_wants_family(channel_keys, _CATALOG_IC12_DIFF_KEYS)

    has_ic = file_has_ic and wants_ic
    has_ic3 = file_has_ic3 and (
        channel_keys is None or "ic3" in channel_keys
    )
    has_ddose = file_has_ddose and wants_ddose
    has_ddose3 = file_has_ddose3 and (
        channel_keys is None or "ic3_ddose" in channel_keys
    )
    has_sigma = file_has_sigma and wants_sigma
    has_field = file_has_field and wants_field
    has_beam = file_has_beam and wants_beam
    has_positions = wants_positions and (
        (position_reference_frame == REFERENCE_ISO and file_has_iso_positions)
        or (position_reference_frame == REFERENCE_CHAMBER and file_has_chamber_positions)
        or (
            position_reference_frame is None
            and (file_has_positions or file_has_iso_positions)
        )
    )
    has_position_error = file_has_position_error and wants_position_error
    has_sigma_error = file_has_sigma_error and wants_sigma_error and has_sigma
    has_ic12_diff = file_has_ic12_diff and wants_ic12_diff

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
        **{k: [] for k in _POS_ERR_KEYS},
        **{k: [] for k in _SIGMA_ERR_KEYS},
        **{k: [] for k in _IC12_DIFF_KEYS},
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
        frame_positions = None
        if has_positions or has_ic12_diff:
            if position_reference_frame == REFERENCE_ISO and iso_position_source is not None:
                frame_positions = frame_timeslice_iso_position_arrays(
                    df, iso_position_source,
                )
            elif (
                position_reference_frame == REFERENCE_CHAMBER
                and chamber_position_source is not None
            ):
                frame_positions = frame_timeslice_chamber_position_arrays(
                    df, chamber_position_source,
                )
            elif file_has_positions:
                frame_positions = tuple(
                    df[pos_cols[label]].values.astype(float) for label in _POS_KEYS
                )
            if frame_positions is None:
                nan = np.full(n, np.nan)
                frame_positions = (nan, nan, nan, nan)
            if has_positions:
                for key, arr in zip(_POS_KEYS, frame_positions):
                    parts[key].append(arr)
            if has_ic12_diff:
                ic1_x, ic1_y, ic2_x, ic2_y = frame_positions
                parts["ic12_x_diff"].append(ic2_x - ic1_x)
                parts["ic12_y_diff"].append(ic2_y - ic1_y)

        if has_position_error and error_source is not None:
            frame_errors = frame_timeslice_error_arrays(df, error_source)
            if frame_errors is None:
                nan = np.full(n, np.nan)
                for key in _POS_ERR_KEYS:
                    parts[key].append(nan)
            else:
                for key, arr in zip(_POS_ERR_KEYS, frame_errors):
                    parts[key].append(arr)

        if has_sigma_error and sigma_target_cols is not None:
            from ..common.timeslice_sigma import frame_timeslice_sigma_error_arrays

            frame_errors = frame_timeslice_sigma_error_arrays(
                df, sigma_source, sigma_target_cols,
            )
            if frame_errors is None:
                nan = np.full(n, np.nan)
                for key in _SIGMA_ERR_KEYS:
                    parts[key].append(nan)
            else:
                for key, arr in zip(_SIGMA_ERR_KEYS, frame_errors):
                    parts[key].append(arr)

        energy_parts.append(np.full(n, energy))
        layer_boundaries.append((offset, energy))
        offset += n

    if offset == 0:
        return None

    result: dict = {
        "layer_boundaries": layer_boundaries,
        "n_samples": offset,
        "has_ic3": file_has_ic3,
        "has_beam": file_has_beam,
        "has_positions": has_positions or has_ic12_diff,
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
        use_chamber_remap = (
            position_reference_frame == REFERENCE_CHAMBER
            or (
                position_reference_frame is None
                and file_has_positions
                and not file_has_iso_positions
            )
        )
        if use_chamber_remap:
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
        else:
            for key in _POS_KEYS:
                _store(key)
        pos_limit = transform.IC_MM_MAX
        for key in _POS_KEYS:
            arr = result[key]
            arr[np.abs(arr) > pos_limit] = np.nan
    if has_position_error:
        for key in _POS_ERR_KEYS:
            _store(key)
    if has_sigma_error:
        for key in _SIGMA_ERR_KEYS:
            _store(key)
    if has_ic12_diff:
        for key in _IC12_DIFF_KEYS:
            _store(key)

    return result


def load_sessions_catalog(
    session_ids: Sequence[str],
    base_dir: str,
    *,
    bg_subtract: bool = False,
    metric_id: str | None = None,
    data_source: DataSourceKind | None = None,
) -> dict[str, dict]:
    """Load catalog data for each session that has usable timeslice channels."""
    channel_keys: frozenset[str] | None = None
    position_reference_frame = None
    if metric_id is not None and data_source is not None:
        if metric_for_option(metric_id, data_source) is None:
            return {}
        channel_keys = channel_keys_for_metric(metric_id)
        position_reference_frame = reference_frame_for_source(data_source)

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = load_session_timeline_catalog(
            sid,
            base_dir,
            bg_subtract=bg_subtract,
            channel_keys=channel_keys,
            position_reference_frame=position_reference_frame,
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

    if family == FAMILY_POSITION:
        return ScatterSpec(
            mode="per_trace",
            per_trace_xy={
                "ic1_x": ("ic1_x", "ic1_y"),
                "ic1_y": ("ic1_x", "ic1_y"),
                "ic2_x": ("ic2_x", "ic2_y"),
                "ic2_y": ("ic2_x", "ic2_y"),
            },
            per_trace_title_suffix=" position (mm)",
            missing_label="No position data",
        )

    if family == FAMILY_SIGMA_ERROR:
        return ScatterSpec(
            mode="per_trace",
            per_trace_xy={
                "sigma_ic1_x_err": ("sigma_ic1_x_err", "sigma_ic1_y_err"),
                "sigma_ic1_y_err": ("sigma_ic1_x_err", "sigma_ic1_y_err"),
                "sigma_ic2_x_err": ("sigma_ic2_x_err", "sigma_ic2_y_err"),
                "sigma_ic2_y_err": ("sigma_ic2_x_err", "sigma_ic2_y_err"),
            },
            per_trace_title_suffix=" σ error (mm)",
            missing_label="No sigma error data",
        )

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
