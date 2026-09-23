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
# Beam energy spread σE in % of E; range straggling is added on top.
DEFAULT_ENERGY_SPREAD_PCT = 1.0
# 0 means autoscale so the energy span matches the XY span.
DEFAULT_MM_PER_MEV = 0.0
# vispy +Z is up; range is −R so gantry 0° puts high energy at the bottom.
# 90° about X then lays that depth along Y.
DEFAULT_GANTRY_DEG = 90.0

MEDIUM_WATER = "water"
MEDIUM_COPPER = "copper"
MediumKind = Literal["water", "copper"]
DEFAULT_MEDIUM = MEDIUM_WATER
# Phantom thickness along the beam; 0 means auto (deep enough that nothing exits).
DEFAULT_PHANTOM_MM = 0.0
# How far past the deepest range an Auto phantom reaches, in range-spread σ.
# The depth-dose tables carry 5σ, so that is also the most Auto can hold.
DEFAULT_AUTO_MARGIN_SIGMA = 5.0
# Water-equivalent material between nozzle and phantom surface (tank wall, buildup, range shifter).
DEFAULT_ENTRANCE_WET_MM = 0.0

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

# Dose: Bragg curve and depth-grown scatter, in Gy. Stops: where protons end, in MU or protons.
WEIGHT_DOSE = "dose"
WEIGHT_MU = "mu"
WEIGHT_PROTONS = "protons"
WeightKind = Literal["dose", "mu", "protons"]
DEFAULT_WEIGHT = WEIGHT_DOSE

# Global 3D gamma (AAPM TG-218 defaults): measured is the reference, plan is searched.
DEFAULT_GAMMA_DOSE_PCT = 3.0
DEFAULT_GAMMA_DTA_MM = 2.0
DEFAULT_GAMMA_CUTOFF_PCT = 10.0
GAMMA_TOLERANCE_PCT = 95.0
GAMMA_ACTION_PCT = 90.0

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
SCALE_INFERNO = "inferno"
SCALE_PLASMA = "plasma"
SCALE_CIVIDIS = "cividis"
SCALE_TURBO = "turbo"
SCALE_CUBEHELIX = "cubehelix"
SCALE_HEAT = "afmhot"
SCALE_DEEP = "YlGnBu_r"
SCALE_COOLWARM = "coolwarm"
SCALE_RDYLBU = "RdYlBu_r"
SCALE_SPECTRAL = "Spectral_r"
SCALE_BERLIN = "berlin"
SCALE_MANAGUA = "managua_r"
SCALE_PUOR = "PuOr_r"
# Every sequential map starts dark at 0 (the view background) and brightens with
# dose; all but Turbo and Heat are perceptually uniform, and Cividis is color-vision safe.
SEQUENTIAL_SCALES = (
    (SCALE_VIRIDIS, "Viridis"),
    (SCALE_MAGMA, "Magma"),
    (SCALE_INFERNO, "Inferno"),
    (SCALE_PLASMA, "Plasma"),
    (SCALE_CIVIDIS, "Cividis"),
    (SCALE_DEEP, "Deep"),
    (SCALE_CUBEHELIX, "Cubehelix"),
    (SCALE_HEAT, "Heat"),
    (SCALE_TURBO, "Turbo"),
)
# Every divergent map runs cool (plan extra) to warm (measured extra).
DIVERGENT_SCALES = (
    (SCALE_MANAGUA, "Managua"),
    (SCALE_BERLIN, "Berlin"),
    (SCALE_COOLWARM, "Coolwarm"),
    (SCALE_RDYLBU, "Red–yellow–blue"),
    (SCALE_SPECTRAL, "Spectral"),
    (SCALE_PUOR, "Purple–orange"),
)
DEFAULT_SCALE = SCALE_VIRIDIS
DEFAULT_DIVERGENT_SCALE = SCALE_MANAGUA


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
    smear_axis_units: float = DEFAULT_ENERGY_SPREAD_PCT
    mm_per_mev: float = DEFAULT_MM_PER_MEV
    splat_cap: int = DEFAULT_SPOT_CAP
    gantry_deg: float = DEFAULT_GANTRY_DEG
    medium: MediumKind = DEFAULT_MEDIUM
    phantom_mm: float = DEFAULT_PHANTOM_MM
    auto_margin_sigma: float = DEFAULT_AUTO_MARGIN_SIGMA
    entrance_wet_mm: float = DEFAULT_ENTRANCE_WET_MM
    agreement: AgreementKind = DEFAULT_AGREEMENT
    error_mode: ErrorKind = DEFAULT_ERROR_MODE
    error_scale: float = DEFAULT_ERROR_SCALE
    weight_mode: WeightKind = DEFAULT_WEIGHT
    ic_gap_mm: float = DEFAULT_IC_GAP_MM
    ray_mode: RayKind = DEFAULT_RAY
    scale: str = DEFAULT_SCALE
    auto_scale: bool = True
    voxel_mm: float = 1.0
    smooth: bool = True
    gamma: bool = False
    gamma_dose_pct: float = DEFAULT_GAMMA_DOSE_PCT
    gamma_dta_mm: float = DEFAULT_GAMMA_DTA_MM
    gamma_cutoff_pct: float = DEFAULT_GAMMA_CUTOFF_PCT

    @property
    def title(self) -> str:
        return "Dose Volume (3D)"


# Loader still speaks this name.
SplatConfig = DoseVolumeConfig
