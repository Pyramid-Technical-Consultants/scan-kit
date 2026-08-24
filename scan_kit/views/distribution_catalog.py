"""Distribution modes and presets for the Distribution Explorer viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    DATA_SOURCES,
    DataSourceKind,
    UnifiedViewOption,
)

METRIC_POSITION = "position"
METRIC_POSITION_ERROR = "position_error"
METRIC_SIGMA = "sigma"
METRIC_SIGMA_ERROR = "sigma_error"
METRIC_CONFIDENCE = "confidence"
METRIC_GAUSSIAN_FILTER = "gaussian_fit_filter"

MODE_POSITION_SPOT = "position_spot"
MODE_POSITION_TIMESLICE = "position_timeslice"
MODE_POSITION_ERROR_SPOT = "position_error_spot"
MODE_POSITION_ERROR_TIMESLICE = "position_error_timeslice"
MODE_SIGMA_SPOT = "sigma_spot"
MODE_SIGMA_TIMESLICE = "sigma_timeslice"
MODE_SIGMA_ERROR_TIMESLICE = "sigma_error_timeslice"
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
        DATA_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_POSITION_TIMESLICE,
        METRIC_POSITION,
        "Position",
        "Position distribution (timeslice, beam-on)",
        DATA_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_POSITION_ERROR_SPOT,
        METRIC_POSITION_ERROR,
        "Position Error",
        "Position error distribution (spot)",
        DATA_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_SIGMA_SPOT,
        METRIC_SIGMA,
        "Sigma",
        "Sigma distribution (spot)",
        DATA_SOURCE_SPOT,
        uses_bg_subtract=False,
    ),
    DistributionModeDef(
        MODE_POSITION_ERROR_TIMESLICE,
        METRIC_POSITION_ERROR,
        "Position Error",
        "Position error distribution (timeslice, beam-on)",
        DATA_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_SIGMA_TIMESLICE,
        METRIC_SIGMA,
        "Sigma",
        "Sigma distribution (timeslice, beam-on)",
        DATA_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_SIGMA_ERROR_TIMESLICE,
        METRIC_SIGMA_ERROR,
        "Sigma Error",
        "Sigma error distribution (timeslice, beam-on)",
        DATA_SOURCE_TIMESLICE,
    ),
    DistributionModeDef(
        MODE_CONFIDENCE_TIMESLICE,
        METRIC_CONFIDENCE,
        "Confidence Correlations",
        "Confidence correlations (timeslice, beam-on)",
        DATA_SOURCE_TIMESLICE,
        supports_plot_style=False,
    ),
    DistributionModeDef(
        MODE_GAUSSIAN_FILTER,
        METRIC_GAUSSIAN_FILTER,
        "Gaussian Fit Filter Coverage",
        "Gaussian fit filter coverage",
        DATA_SOURCE_TIMESLICE,
        supports_plot_style=False,
    ),
)

VIEW_OPTIONS: tuple[UnifiedViewOption, ...] = tuple(
    UnifiedViewOption(mode.metric_id, mode.label, mode.source)  # type: ignore[arg-type]
    for mode in MODES
)

_SOURCE_LABEL = dict(DATA_SOURCES)

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
METRIC_MODE_BY_SOURCE: dict[tuple[str, DataSourceKind], str] = {
    (mode.metric_id, mode.source): mode.id  # type: ignore[misc]
    for mode in MODES
}


def resolve_mode_id(metric_id: str, source: DataSourceKind) -> str | None:
    return METRIC_MODE_BY_SOURCE.get((metric_id, source))


def metric_source_for_mode(mode_id: str) -> tuple[str, DataSourceKind] | None:
    mode = MODE_BY_ID.get(mode_id)
    if mode is None:
        return None
    return mode.metric_id, mode.source  # type: ignore[return-value]


@dataclass
class DistributionConfig:
    mode: str = MODE_POSITION_ERROR_SPOT
    plot_style: PlotStyle = "contour"

    @property
    def title(self) -> str:
        mode = MODE_BY_ID.get(self.mode)
        return mode.title if mode else self.mode

    @property
    def supports_plot_style(self) -> bool:
        mode = MODE_BY_ID.get(self.mode)
        return mode.supports_plot_style if mode else False
