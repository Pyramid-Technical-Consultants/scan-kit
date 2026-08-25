"""Y-metric and X-parameter catalogs for the universal binned summary viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, DataFilterSelection
from .unified_catalog import (
    DATA_SOURCE_SPOT,
    DATA_SOURCE_TIMESLICE,
    UnifiedViewOption,
    option_key,
)

Glyph = Literal["box", "violin", "mean"]
BinMode = Literal["unique", "quantile"]

Y_DOSE_ERROR = "dose_error"
Y_DOSE_RATIO = "dose_ratio"
Y_DOSE_RATE = "dose_rate"
Y_CURRENT_RATIO = "current_ratio"
Y_IC_CURRENT = "ic_current"
Y_POSITION_ERROR = "position_error"
Y_SIGMA = "sigma"
Y_SPOT_TIME = "spot_time"

X_ENERGY = "energy"
X_TARGET_MU = "target_mu"
X_SPOT_TIME = "spot_time"
X_RADIUS = "radius"

GLYPH_BOX = "box"
GLYPH_VIOLIN = "violin"
GLYPH_MEAN = "mean"

PRESET_DOSE_ERROR_ENERGY = "dose_error_energy"
PRESET_DOSE_ERROR_ENERGY_MEAN = "dose_error_energy_mean"
PRESET_DOSE_ERROR_MU = "dose_error_mu"
PRESET_DOSE_RATIO_ENERGY = "dose_ratio_energy"
PRESET_DOSE_RATIO_SPOT_TIME = "dose_ratio_spot_time"
PRESET_DOSE_RATIO_RADIUS = "dose_ratio_radius"
PRESET_DOSE_RATE_ENERGY = "dose_rate_energy"
PRESET_CURRENT_RATIO_ENERGY = "current_ratio_energy"
PRESET_IC_CURRENT_ENERGY = "ic_current_energy"
PRESET_POSITION_ERROR_ENERGY = "position_error_energy"
PRESET_SIGMA_ENERGY = "sigma_energy"
PRESET_SPOT_TIME_ENERGY = "spot_time_energy"


@dataclass(frozen=True)
class SeriesDef:
    key: str
    label: str


@dataclass(frozen=True)
class YGroupDef:
    id: str
    label: str
    series: tuple[SeriesDef, ...]
    default_glyph: Glyph
    trend_unit: str
    sources: tuple[str, ...] = (DATA_SOURCE_SPOT,)
    supports_data_filter: bool = True


@dataclass(frozen=True)
class XParamDef:
    id: str
    label: str
    column: str
    bin_mode: BinMode
    xlabel: str
    n_bins: int = 8


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    y_group: str
    x_param: str
    glyph: Glyph
    show_trend: bool = True
    show_hist: bool = False
    show_corr: bool = False


Y_GROUPS: tuple[YGroupDef, ...] = (
    YGroupDef(
        Y_DOSE_ERROR,
        "Dose Error (%)",
        (
            SeriesDef("ic1_dose_err_pct", "IC1"),
            SeriesDef("ic2_dose_err_pct", "IC2"),
            SeriesDef("ic3_dose_err_pct", "IC3"),
        ),
        GLYPH_BOX,
        "%/unit",
    ),
    YGroupDef(
        Y_DOSE_RATIO,
        "Dose Ratios",
        (
            SeriesDef("ic21_ratio", "IC2/IC1 (%)"),
            SeriesDef("ic31_ratio", "IC3/IC1 (%)"),
            SeriesDef("ic32_ratio", "IC3/IC2 (%)"),
        ),
        GLYPH_BOX,
        "%/unit",
    ),
    YGroupDef(
        Y_DOSE_RATE,
        "Dose Rate (MU/s)",
        (SeriesDef("mu_rate", "Delivery Rate (MU/s)"),),
        GLYPH_MEAN,
        "MU/s/unit",
        supports_data_filter=False,
    ),
    YGroupDef(
        Y_CURRENT_RATIO,
        "Current Ratios (%)",
        (
            SeriesDef("ic21_ratio", "IC2/IC1 (%)"),
            SeriesDef("ic31_ratio", "IC3/IC1 (%)"),
            SeriesDef("ic32_ratio", "IC3/IC2 (%)"),
        ),
        GLYPH_MEAN,
        "%/unit",
        (DATA_SOURCE_TIMESLICE,),
        supports_data_filter=False,
    ),
    YGroupDef(
        Y_IC_CURRENT,
        "IC Current (nA)",
        (
            SeriesDef("ic1_current", "IC1"),
            SeriesDef("ic2_current", "IC2"),
            SeriesDef("ic3_current", "IC3 (sum A+B+C+D)"),
        ),
        GLYPH_BOX,
        "nA/unit",
        (DATA_SOURCE_TIMESLICE,),
    ),
    YGroupDef(
        Y_POSITION_ERROR,
        "Position Error (mm)",
        (
            SeriesDef("ic1_x_err", "IC1 X"),
            SeriesDef("ic1_y_err", "IC1 Y"),
            SeriesDef("ic2_x_err", "IC2 X"),
            SeriesDef("ic2_y_err", "IC2 Y"),
        ),
        GLYPH_VIOLIN,
        "mm/unit",
        (DATA_SOURCE_SPOT, DATA_SOURCE_TIMESLICE),
    ),
    YGroupDef(
        Y_SIGMA,
        "Sigma (mm)",
        (
            SeriesDef("ic1_sig_x", "IC1 σx"),
            SeriesDef("ic1_sig_y", "IC1 σy"),
            SeriesDef("ic2_sig_x", "IC2 σx"),
            SeriesDef("ic2_sig_y", "IC2 σy"),
        ),
        GLYPH_VIOLIN,
        "mm/unit",
        (DATA_SOURCE_SPOT, DATA_SOURCE_TIMESLICE),
    ),
    YGroupDef(
        Y_SPOT_TIME,
        "Spot Delivery Time",
        (
            SeriesDef("spot_time", "Total (ms)"),
            SeriesDef("beam_on_time", "Beam-On (ms)"),
            SeriesDef("overhead_time", "Overhead (ms)"),
        ),
        GLYPH_BOX,
        "ms/unit",
    ),
)

X_PARAMS: tuple[XParamDef, ...] = (
    XParamDef(X_ENERGY, "Energy", "energy", "unique", "Energy (MeV)"),
    XParamDef(X_TARGET_MU, "Target MU", "target_mu", "quantile", "Target MU", n_bins=8),
    XParamDef(X_SPOT_TIME, "Spot Time", "spot_time", "quantile", "Spot time (ms)", n_bins=8),
    XParamDef(X_RADIUS, "Beam Radius", "radius", "quantile", "Radius (mm)", n_bins=8),
)

PRESETS: tuple[PresetDef, ...] = (
    PresetDef(
        PRESET_DOSE_ERROR_ENERGY, "Dose error vs Energy",
        Y_DOSE_ERROR, X_ENERGY, GLYPH_BOX, show_hist=True, show_corr=True,
    ),
    PresetDef(
        PRESET_DOSE_ERROR_ENERGY_MEAN, "Dose error mean vs Energy",
        Y_DOSE_ERROR, X_ENERGY, GLYPH_MEAN, show_hist=True, show_corr=True,
    ),
    PresetDef(
        PRESET_DOSE_RATIO_ENERGY, "Dose ratios vs Energy",
        Y_DOSE_RATIO, X_ENERGY, GLYPH_BOX, show_corr=True,
    ),
    PresetDef(
        PRESET_DOSE_RATE_ENERGY, "Dose rate vs Energy",
        Y_DOSE_RATE, X_ENERGY, GLYPH_MEAN,
    ),
    PresetDef(
        PRESET_CURRENT_RATIO_ENERGY, "Current ratios vs Energy",
        Y_CURRENT_RATIO, X_ENERGY, GLYPH_MEAN,
    ),
    PresetDef(
        PRESET_IC_CURRENT_ENERGY, "IC current vs Energy",
        Y_IC_CURRENT, X_ENERGY, GLYPH_BOX,
    ),
    PresetDef(
        PRESET_POSITION_ERROR_ENERGY, "Position error vs Energy",
        Y_POSITION_ERROR, X_ENERGY, GLYPH_VIOLIN, show_trend=False,
    ),
    PresetDef(
        PRESET_SIGMA_ENERGY, "Sigma vs Energy",
        Y_SIGMA, X_ENERGY, GLYPH_VIOLIN, show_trend=False,
    ),
    PresetDef(
        PRESET_SPOT_TIME_ENERGY, "Spot time vs Energy",
        Y_SPOT_TIME, X_ENERGY, GLYPH_BOX,
    ),
    PresetDef(
        PRESET_DOSE_ERROR_MU, "Dose error vs Target MU",
        Y_DOSE_ERROR, X_TARGET_MU, GLYPH_BOX, show_hist=True, show_corr=True,
    ),
    PresetDef(
        PRESET_DOSE_RATIO_SPOT_TIME, "Dose ratios vs Spot time",
        Y_DOSE_RATIO, X_SPOT_TIME, GLYPH_BOX, show_corr=True,
    ),
    PresetDef(
        PRESET_DOSE_RATIO_RADIUS, "Dose ratios vs Beam radius",
        Y_DOSE_RATIO, X_RADIUS, GLYPH_BOX, show_corr=True,
    ),
)

Y_GROUP_BY_ID = {g.id: g for g in Y_GROUPS}
X_PARAM_BY_ID = {x.id: x for x in X_PARAMS}
PRESET_BY_ID = {p.id: p for p in PRESETS}

VIEW_OPTIONS: tuple[UnifiedViewOption, ...] = tuple(
    UnifiedViewOption(group.id, group.label, source)  # type: ignore[arg-type]
    for group in Y_GROUPS
    for source in group.sources
)


@dataclass
class BinnedSummaryConfig:
    y_group: str = Y_DOSE_RATIO
    source: str = DATA_SOURCE_SPOT
    x_param: str = X_ENERGY
    glyph: Glyph = GLYPH_VIOLIN
    show_trend: bool = True
    show_hist: bool = False
    show_corr: bool = False
    show_fliers: bool = False
    n_bins: int | None = None
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH

    @property
    def data_filter(self) -> DataFilterSelection:
        return DataFilterSelection(
            domain_filter=self.domain_filter,
            beam_state_filter=self.beam_state_filter,
        )

    @property
    def title(self) -> str:
        y = Y_GROUP_BY_ID.get(self.y_group)
        x = X_PARAM_BY_ID.get(self.x_param)
        y_label = y.label if y else self.y_group
        x_label = x.label if x else self.x_param
        return f"{y_label} vs {x_label}"
