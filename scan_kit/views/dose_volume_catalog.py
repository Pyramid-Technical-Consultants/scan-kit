"""Presets and display options for the 3D dose-volume viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GRAIN_SPOT = "spot"
GRAIN_TIMESLICE = "timeslice"
GrainKind = Literal["spot", "timeslice"]

XY_IC1 = "ic1"
XY_IC2 = "ic2"
XY_ISO_RAY = "iso_ray"
XY_PLAN = "plan"
XyMode = Literal["ic1", "ic2", "iso_ray", "plan"]

DEFAULT_SPOT_CAP = 1_000_000
DEFAULT_GAIN = 1.0
DEFAULT_SMEAR_MEV = 0.5
# 0 means autoscale so the energy span matches the XY span.
DEFAULT_MM_PER_MEV = 0.0
# vispy +Z is up; range is −R so gantry 0° puts high energy at the bottom.
# 90° about X then lays that depth along Y.
DEFAULT_GANTRY_DEG = 90.0

MEDIUM_WATER = "water"
MEDIUM_COPPER = "copper"
MediumKind = Literal["water", "copper"]
DEFAULT_MEDIUM = MEDIUM_WATER

AGREE_TRANSPARENT = "transparent"
AGREE_WHITE = "white"
AgreementKind = Literal["transparent", "white"]
DEFAULT_AGREEMENT = AGREE_TRANSPARENT

ERROR_PERCENT = "percent"
ERROR_ABSOLUTE = "absolute"
ErrorKind = Literal["percent", "absolute"]
DEFAULT_ERROR_MODE = ERROR_ABSOLUTE
DEFAULT_ERROR_PCT = 10.0
DEFAULT_ERROR_MU = 0.2
# Native units for the default (absolute) mode.
DEFAULT_ERROR_SCALE = DEFAULT_ERROR_MU

WEIGHT_MU = "mu"
WEIGHT_PROTONS = "protons"
WeightKind = Literal["mu", "protons"]
DEFAULT_WEIGHT = WEIGHT_MU

# How a viewing ray combines the voxels it crosses.
RAY_INTEGRAL = "integral"
RAY_MAXIMUM = "maximum"
RAY_TRANSPARENT = "transparent"
RayKind = Literal["integral", "maximum", "transparent"]
DEFAULT_RAY = RAY_INTEGRAL
DEFAULT_IC_GAP_MM = 10.0

# One color scale paints the number the ray produced.
# Dose uses a sequential map. A difference uses a divergent map.
SCALE_VIRIDIS = "viridis"
SCALE_MAGMA = "magma"
SCALE_TURBO = "turbo"
SCALE_COOLWARM = "coolwarm"
SCALE_SEISMIC = "seismic"
SCALE_BWR = "bwr"
SEQUENTIAL_SCALES = (
    (SCALE_VIRIDIS, "Viridis"),
    (SCALE_MAGMA, "Magma"),
    (SCALE_TURBO, "Turbo"),
)
DIVERGENT_SCALES = (
    (SCALE_COOLWARM, "Coolwarm"),
    (SCALE_SEISMIC, "Seismic"),
    (SCALE_BWR, "Blue–white–red"),
)
DEFAULT_SCALE = SCALE_VIRIDIS
DEFAULT_DIVERGENT_SCALE = SCALE_COOLWARM


def scales_for(compare: bool) -> tuple[tuple[str, str], ...]:
    return DIVERGENT_SCALES if compare else SEQUENTIAL_SCALES


def active_scale(compare: bool, scale: str) -> str:
    """The scale to use. A difference cannot keep a sequential map."""
    allowed = {name for name, _label in scales_for(compare)}
    if scale in allowed:
        return scale
    return DEFAULT_DIVERGENT_SCALE if compare else DEFAULT_SCALE

PRESET_SPOT_IC1 = "spot_ic1"
PRESET_SPOT_ISO_RAY = "spot_iso_ray"
PRESET_TIMESLICE_IC1 = "timeslice_ic1"
PRESET_SPOT_VS_PLAN = "spot_vs_plan"


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    grain: GrainKind = GRAIN_SPOT
    xy_mode: XyMode = XY_IC1
    overlay_plan: bool = False


PRESETS: tuple[PresetDef, ...] = (
    PresetDef(PRESET_SPOT_IC1, "Spot · IC1"),
    PresetDef(PRESET_SPOT_ISO_RAY, "Spot · ISO ray", xy_mode=XY_ISO_RAY),
    PresetDef(PRESET_TIMESLICE_IC1, "Timeslice · IC1", grain=GRAIN_TIMESLICE),
    PresetDef(
        PRESET_SPOT_VS_PLAN,
        "Spot · IC1 − plan",
        overlay_plan=True,
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}


@dataclass
class DoseVolumeConfig:
    grain: GrainKind = GRAIN_SPOT
    xy_mode: XyMode = XY_IC1
    overlay_plan: bool = False
    gain: float = DEFAULT_GAIN
    smear_axis_units: float = DEFAULT_SMEAR_MEV
    mm_per_mev: float = DEFAULT_MM_PER_MEV
    splat_cap: int = DEFAULT_SPOT_CAP
    gantry_deg: float = DEFAULT_GANTRY_DEG
    medium: MediumKind = DEFAULT_MEDIUM
    agreement: AgreementKind = DEFAULT_AGREEMENT
    error_mode: ErrorKind = DEFAULT_ERROR_MODE
    error_scale: float = DEFAULT_ERROR_SCALE
    weight_mode: WeightKind = DEFAULT_WEIGHT
    ic_gap_mm: float = DEFAULT_IC_GAP_MM
    ray_mode: RayKind = DEFAULT_RAY
    scale: str = DEFAULT_SCALE
    auto_scale: bool = True

    @property
    def title(self) -> str:
        return "Dose Volume (3D)"


# Loader still speaks this name.
SplatConfig = DoseVolumeConfig
