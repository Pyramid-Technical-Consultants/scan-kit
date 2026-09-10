"""Signal metrics and unified data-source options for Timeslice Replay."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from ..data.sources.ic_current import SOURCE_IC_CURRENT
from ..data.sources.position import SOURCE_POSITION
from ..data.sources.position_error import SOURCE_POSITION_ERROR
from ..data.sources.sigma import SOURCE_SIGMA
from ..data.sources.sigma_error import SOURCE_SIGMA_ERROR
from ..data.types import (
    DATA_SOURCE_TIMESLICE_CHAMBER,
    DATA_SOURCE_TIMESLICE_ISO,
    DataSourceKind,
    data_source_is_timeslice,
    data_source_reference_frame,
    option_key,
)
from .unified_catalog import (
    UnifiedViewOption,
    format_view_option_label,
    registry_data_sources,
)

METRIC_IC_CURRENT = "ic_current"
METRIC_DDOSE = "ddose"
METRIC_SIGMA = "sigma"
METRIC_SIGMA_ERROR = "sigma_error"
METRIC_POSITION = "position"
METRIC_POSITION_ERROR = "position_error"
METRIC_IC12_POS_DIFF = "ic12_pos_diff"
METRIC_MAG_FIELD = "mag_field"

PRESET_IC_CURRENT = METRIC_IC_CURRENT
PRESET_DDOSE = METRIC_DDOSE
PRESET_SIGMA = METRIC_SIGMA
PRESET_FIELD = METRIC_MAG_FIELD
PRESET_POSITION_ISO = "position_iso"


@dataclass(frozen=True)
class ReplayMetricDef:
    id: str
    label: str
    channel_keys: tuple[str, ...]
    default_channel_keys: tuple[str, ...]
    registry_source_id: str | None = None
    custom_sources: tuple[DataSourceKind, ...] = ()
    peer_overlay_default: bool = False


def _registry_timeslice_sources(source_id: str) -> tuple[DataSourceKind, ...]:
    return tuple(
        source
        for source in registry_data_sources(source_id)
        if data_source_is_timeslice(source)
    )


REPLAY_METRICS: tuple[ReplayMetricDef, ...] = (
    ReplayMetricDef(
        METRIC_IC_CURRENT,
        "IC Current",
        ("ic1", "ic2", "ic3"),
        ("ic1", "ic2", "ic3"),
        registry_source_id=SOURCE_IC_CURRENT,
    ),
    ReplayMetricDef(
        METRIC_DDOSE,
        "dDose/dt",
        ("ic1_ddose", "ic2_ddose", "ic3_ddose"),
        ("ic1_ddose", "ic2_ddose", "ic3_ddose"),
        custom_sources=(DATA_SOURCE_TIMESLICE_ISO,),
        peer_overlay_default=True,
    ),
    ReplayMetricDef(
        METRIC_SIGMA,
        "Sigma (mm)",
        ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y"),
        ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y"),
        registry_source_id=SOURCE_SIGMA,
    ),
    ReplayMetricDef(
        METRIC_SIGMA_ERROR,
        "Sigma Error (mm)",
        (
            "sigma_ic1_x_err",
            "sigma_ic1_y_err",
            "sigma_ic2_x_err",
            "sigma_ic2_y_err",
        ),
        (
            "sigma_ic1_x_err",
            "sigma_ic1_y_err",
            "sigma_ic2_x_err",
            "sigma_ic2_y_err",
        ),
        registry_source_id=SOURCE_SIGMA_ERROR,
    ),
    ReplayMetricDef(
        METRIC_POSITION,
        "Position (mm)",
        ("ic1_x", "ic1_y", "ic2_x", "ic2_y"),
        ("ic1_x", "ic1_y", "ic2_x", "ic2_y"),
        registry_source_id=SOURCE_POSITION,
    ),
    ReplayMetricDef(
        METRIC_POSITION_ERROR,
        "Position Error (mm)",
        ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
        ("ic1_x_err", "ic1_y_err", "ic2_x_err", "ic2_y_err"),
        registry_source_id=SOURCE_POSITION_ERROR,
    ),
    ReplayMetricDef(
        METRIC_IC12_POS_DIFF,
        "IC2−IC1 Position (mm)",
        ("ic12_x_diff", "ic12_y_diff"),
        ("ic12_x_diff", "ic12_y_diff"),
        registry_source_id=SOURCE_IC12_POS_DIFF,
    ),
    ReplayMetricDef(
        METRIC_MAG_FIELD,
        "Magnetic Field",
        ("bx", "by"),
        ("bx", "by"),
        custom_sources=(DATA_SOURCE_TIMESLICE_ISO,),
    ),
)

METRIC_BY_ID = {metric.id: metric for metric in REPLAY_METRICS}

REPLAY_REGISTRY_SOURCE_IDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        metric.registry_source_id
        for metric in REPLAY_METRICS
        if metric.registry_source_id is not None
    )
)


def metric_data_sources(metric: ReplayMetricDef) -> tuple[DataSourceKind, ...]:
    if metric.custom_sources:
        return metric.custom_sources
    if metric.registry_source_id is not None:
        return _registry_timeslice_sources(metric.registry_source_id)
    return ()


_ALL_VIEW_OPTIONS: tuple[UnifiedViewOption, ...] = tuple(
    UnifiedViewOption(
        metric.id,
        format_view_option_label(
            metric.label,
            source,
            sibling_sources=metric_data_sources(metric),
        ),
        source,
    )
    for metric in REPLAY_METRICS
    for source in metric_data_sources(metric)
)

VIEW_OPTIONS: tuple[UnifiedViewOption, ...] = _ALL_VIEW_OPTIONS


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    metric_id: str
    source: DataSourceKind
    channels: tuple[str, ...]


PRESETS: tuple[PresetDef, ...] = (
    PresetDef(
        PRESET_IC_CURRENT,
        "IC Current",
        METRIC_IC_CURRENT,
        DATA_SOURCE_TIMESLICE_ISO,
        ("ic1", "ic2", "ic3"),
    ),
    PresetDef(
        PRESET_DDOSE,
        "dDose/dt",
        METRIC_DDOSE,
        DATA_SOURCE_TIMESLICE_ISO,
        ("ic1_ddose", "ic2_ddose", "ic3_ddose"),
    ),
    PresetDef(
        PRESET_SIGMA,
        "Sigma",
        METRIC_SIGMA,
        DATA_SOURCE_TIMESLICE_ISO,
        ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y"),
    ),
    PresetDef(
        PRESET_FIELD,
        "Magnetic Field",
        METRIC_MAG_FIELD,
        DATA_SOURCE_TIMESLICE_ISO,
        ("bx", "by"),
    ),
    PresetDef(
        PRESET_POSITION_ISO,
        "Position (Isocenter)",
        METRIC_POSITION,
        DATA_SOURCE_TIMESLICE_ISO,
        ("ic1_x", "ic1_y", "ic2_x", "ic2_y"),
    ),
)

PRESET_BY_ID = {preset.id: preset for preset in PRESETS}

# Backward-compatible preset channel map for tests and doc screenshots.
PRESET_CHANNELS: dict[str, tuple[str, ...]] = {
    preset.id: preset.channels for preset in PRESETS
}
PRESET_LABELS: dict[str, str] = {preset.id: preset.label for preset in PRESETS}


def metric_for_option(metric_id: str, source: DataSourceKind) -> ReplayMetricDef | None:
    metric = METRIC_BY_ID.get(metric_id)
    if metric is None:
        return None
    if source not in metric_data_sources(metric):
        return None
    return metric


def channel_keys_for_metric(metric_id: str) -> frozenset[str]:
    metric = METRIC_BY_ID.get(metric_id)
    if metric is None:
        return frozenset()
    return frozenset(metric.channel_keys)


def reference_frame_for_source(source: DataSourceKind):
    from ..data.types import REFERENCE_CHAMBER, REFERENCE_ISO

    return data_source_reference_frame(source)


def custom_option_keys() -> tuple[str, ...]:
    return tuple(
        option_key(source, metric.id)
        for metric in REPLAY_METRICS
        if metric.custom_sources
        for source in metric.custom_sources
    )
