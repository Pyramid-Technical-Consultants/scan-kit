"""Distribution modes and presets for the Distribution Explorer viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..data.registry import get_spec
from ..data.sources.confidence import SOURCE_CONFIDENCE
from ..data.sources.gaussian_fit_filter import SOURCE_GAUSSIAN_FIT_FILTER
from ..data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from ..data.sources.position import SOURCE_POSITION
from ..data.sources.position_error import SOURCE_POSITION_ERROR
from ..data.sources.sigma import SOURCE_SIGMA
from ..data.sources.sigma_error import SOURCE_SIGMA_ERROR
from ..data.types import coarse_data_source, DataSourceKind, option_key
from .unified_catalog import (
    COARSE_SOURCE_SPOT,
    COARSE_SOURCE_TIMESLICE,
    CoarseDataSourceKind,
    UnifiedViewOption,
    format_view_option_label,
)

METRIC_POSITION = "position"
METRIC_POSITION_ERROR = "position_error"
METRIC_SIGMA = "sigma"
METRIC_SIGMA_ERROR = "sigma_error"
METRIC_IC12_POS_DIFF = "ic12_pos_diff"
METRIC_CONFIDENCE = "confidence"
METRIC_GAUSSIAN_FILTER = "gaussian_fit_filter"

MODE_POSITION_SPOT = "position_spot"
MODE_POSITION_TIMESLICE = "position_timeslice"
MODE_POSITION_ERROR_SPOT = "position_error_spot"
MODE_POSITION_ERROR_TIMESLICE = "position_error_timeslice"
MODE_SIGMA_SPOT = "sigma_spot"
MODE_SIGMA_TIMESLICE = "sigma_timeslice"
MODE_SIGMA_ERROR_SPOT = "sigma_error_spot"
MODE_SIGMA_ERROR_TIMESLICE = "sigma_error_timeslice"
MODE_IC12_POS_DIFF_SPOT = "ic12_pos_diff_spot"
MODE_IC12_POS_DIFF_TIMESLICE = "ic12_pos_diff_timeslice"
MODE_CONFIDENCE_TIMESLICE = "confidence_timeslice"
MODE_GAUSSIAN_FILTER = "gaussian_fit_filter"

PRESET_POSITION_SPOT = MODE_POSITION_SPOT
PRESET_POSITION_TIMESLICE = MODE_POSITION_TIMESLICE
PRESET_POSITION_ERROR_SPOT = MODE_POSITION_ERROR_SPOT
PRESET_POSITION_ERROR_TIMESLICE = MODE_POSITION_ERROR_TIMESLICE
PRESET_SIGMA_SPOT = MODE_SIGMA_SPOT
PRESET_SIGMA_TIMESLICE = MODE_SIGMA_TIMESLICE
PRESET_SIGMA_ERROR_TIMESLICE = MODE_SIGMA_ERROR_TIMESLICE
PRESET_CONFIDENCE_TIMESLICE = MODE_CONFIDENCE_TIMESLICE
PRESET_GAUSSIAN_FILTER = MODE_GAUSSIAN_FILTER

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, DataFilterSelection

PlotStyle = Literal["contour", "scatter"]


@dataclass(frozen=True)
class DistributionModeDef:
    id: str
    metric_id: str
    label: str
    title: str
    source: str
    uses_bg_subtract: bool = True
    supports_plot_style: bool = True
    supports_data_filter: bool = True


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    mode: str


MODES: tuple[DistributionModeDef, ...] = (
    DistributionModeDef(
        MODE_POSITION_SPOT,
        METRIC_POSITION,
        "Position",
        "Position distribution (spot)",
        COARSE_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_POSITION_TIMESLICE,
        METRIC_POSITION,
        "Position",
        "Position distribution (timeslice)",
        COARSE_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_POSITION_ERROR_SPOT,
        METRIC_POSITION_ERROR,
        "Position Error",
        "Position error distribution (spot)",
        COARSE_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_SIGMA_SPOT,
        METRIC_SIGMA,
        "Sigma",
        "Sigma distribution (spot)",
        COARSE_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_SIGMA_ERROR_SPOT,
        METRIC_SIGMA_ERROR,
        "Sigma Error",
        "Sigma error distribution (spot)",
        COARSE_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_POSITION_ERROR_TIMESLICE,
        METRIC_POSITION_ERROR,
        "Position Error",
        "Position error distribution (timeslice)",
        COARSE_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_SIGMA_TIMESLICE,
        METRIC_SIGMA,
        "Sigma",
        "Sigma distribution (timeslice)",
        COARSE_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_SIGMA_ERROR_TIMESLICE,
        METRIC_SIGMA_ERROR,
        "Sigma Error",
        "Sigma error distribution (timeslice)",
        COARSE_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_IC12_POS_DIFF_SPOT,
        METRIC_IC12_POS_DIFF,
        "IC2−IC1 Position",
        "IC2−IC1 position difference (spot)",
        COARSE_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_IC12_POS_DIFF_TIMESLICE,
        METRIC_IC12_POS_DIFF,
        "IC2−IC1 Position",
        "IC2−IC1 position difference (timeslice)",
        COARSE_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_CONFIDENCE_TIMESLICE,
        METRIC_CONFIDENCE,
        "Confidence Correlations",
        "Confidence correlations (timeslice)",
        COARSE_SOURCE_TIMESLICE,
        supports_plot_style=False,
        supports_data_filter=False,
    ),
    DistributionModeDef(
        MODE_GAUSSIAN_FILTER,
        METRIC_GAUSSIAN_FILTER,
        "Gaussian Fit Filter Coverage",
        "Gaussian fit filter coverage",
        COARSE_SOURCE_TIMESLICE,
        supports_plot_style=False,
        supports_data_filter=False,
    ),
)

METRIC_SOURCE_IDS: dict[str, str] = {
    METRIC_POSITION: SOURCE_POSITION,
    METRIC_POSITION_ERROR: SOURCE_POSITION_ERROR,
    METRIC_SIGMA: SOURCE_SIGMA,
    METRIC_SIGMA_ERROR: SOURCE_SIGMA_ERROR,
    METRIC_IC12_POS_DIFF: SOURCE_IC12_POS_DIFF,
    METRIC_CONFIDENCE: SOURCE_CONFIDENCE,
    METRIC_GAUSSIAN_FILTER: SOURCE_GAUSSIAN_FIT_FILTER,
}


def _data_sources_for_mode(mode: DistributionModeDef) -> tuple[DataSourceKind, ...]:
    source_id = METRIC_SOURCE_IDS.get(mode.metric_id)
    if source_id is None:
        return ()
    spec = get_spec(source_id)
    return tuple(
        ds for ds in sorted(spec.data_sources)
        if coarse_data_source(ds) == mode.source
    )


VIEW_OPTIONS: tuple[UnifiedViewOption, ...] = tuple(
    UnifiedViewOption(
        mode.metric_id,
        format_view_option_label(
            mode.label,
            data_source,
            sibling_sources=_data_sources_for_mode(mode),
        ),
        data_source,
    )
    for mode in MODES
    for data_source in _data_sources_for_mode(mode)
)

_SOURCE_LABEL = {
    COARSE_SOURCE_SPOT: "Spot",
    COARSE_SOURCE_TIMESLICE: "Timeslice",
}

PRESETS: tuple[PresetDef, ...] = tuple(
    PresetDef(
        mode.id,
        f"{mode.label} ({_SOURCE_LABEL.get(mode.source, mode.source)})",
        mode.id,
    )
    for mode in MODES
)

MODE_BY_ID = {mode.id: mode for mode in MODES}
PRESET_BY_ID = {preset.id: preset for preset in PRESETS}
METRIC_MODE_BY_SOURCE: dict[tuple[str, CoarseDataSourceKind], str] = {
    (mode.metric_id, mode.source): mode.id  # type: ignore[misc]
    for mode in MODES
}


def resolve_mode_id(metric_id: str, source: DataSourceKind) -> str | None:
    return METRIC_MODE_BY_SOURCE.get((metric_id, coarse_data_source(source)))


def metric_source_for_mode(mode_id: str) -> tuple[str, CoarseDataSourceKind] | None:
    mode = MODE_BY_ID.get(mode_id)
    if mode is None:
        return None
    return mode.metric_id, mode.source  # type: ignore[return-value]


def default_data_source_for_mode(
    mode_id: str,
    availability: dict[str, bool],
) -> DataSourceKind | None:
    mapping = metric_source_for_mode(mode_id)
    if mapping is None:
        return None
    metric_id, coarse = mapping
    fallback: DataSourceKind | None = None
    for opt in VIEW_OPTIONS:
        if opt.id != metric_id or coarse_data_source(opt.source) != coarse:
            continue
        fallback = opt.source
        if availability.get(option_key(opt.source, opt.id), False):
            return opt.source
    return fallback


@dataclass
class DistributionConfig:
    mode: str = MODE_POSITION_ERROR_SPOT
    plot_style: PlotStyle = "contour"
    contour_cutoff_percentile: float = 5.0
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH
    show_plan: bool = False
    show_ic1: bool = True
    show_ic2: bool = True

    @property
    def data_filter(self) -> DataFilterSelection:
        return DataFilterSelection(
            domain_filter=self.domain_filter,
            beam_state_filter=self.beam_state_filter,
        )

    @property
    def title(self) -> str:
        mode = MODE_BY_ID.get(self.mode)
        return mode.title if mode else self.mode

    @property
    def supports_plot_style(self) -> bool:
        mode = MODE_BY_ID.get(self.mode)
        return mode.supports_plot_style if mode else False

    @property
    def supports_data_filter(self) -> bool:
        mode = MODE_BY_ID.get(self.mode)
        return mode.supports_data_filter if mode else False
