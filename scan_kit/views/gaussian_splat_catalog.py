"""Presets and display options for the 3D Gaussian splat viewer."""

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

DEFAULT_SPLAT_CAP = 1_000_000
DEFAULT_GAIN = 1.0
DEFAULT_SMEAR_MEV = 0.5
# 0 means autoscale so the energy span matches the XY span.
DEFAULT_MM_PER_MEV = 0.0
# vispy Z is up; 90° about X puts energy along Y (gantry 90 / Y↔Z).
DEFAULT_GANTRY_DEG = 90.0

MEDIUM_WATER = "water"
MEDIUM_AIR = "air"
MediumKind = Literal["water", "air"]
DEFAULT_MEDIUM = MEDIUM_WATER

AGREE_TRANSPARENT = "transparent"
AGREE_WHITE = "white"
AgreementKind = Literal["transparent", "white"]
DEFAULT_AGREEMENT = AGREE_TRANSPARENT

PLAN_RGB = (1.0, 0.55, 0.15)

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
class SplatConfig:
    grain: GrainKind = GRAIN_SPOT
    xy_mode: XyMode = XY_IC1
    overlay_plan: bool = False
    gain: float = DEFAULT_GAIN
    smear_axis_units: float = DEFAULT_SMEAR_MEV
    mm_per_mev: float = DEFAULT_MM_PER_MEV
    splat_cap: int = DEFAULT_SPLAT_CAP
    gantry_deg: float = DEFAULT_GANTRY_DEG
    medium: MediumKind = DEFAULT_MEDIUM
    agreement: AgreementKind = DEFAULT_AGREEMENT

    @property
    def title(self) -> str:
        return "Gaussian Splats (3D)"
