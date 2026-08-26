"""Presets and display options for the 3D IC beam trajectory viewer."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_DEFAULT = "default"
PRESET_PLAN_FOCUS = "plan_focus"
PRESET_UPSTREAM = "upstream_zoom"

DEFAULT_EXTEND_UPSTREAM_MM = 2000.0
DEFAULT_EXTEND_DOWNSTREAM_MM = 2000.0
DEFAULT_PIVOT_MARGIN_MM = 100.0


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    show_spots: bool = True
    show_plan: bool = True
    show_pivot_markers: bool = True
    show_iso_planes: bool = True
    show_ic_planes: bool = True
    extend_upstream_mm: float = DEFAULT_EXTEND_UPSTREAM_MM
    extend_downstream_mm: float = DEFAULT_EXTEND_DOWNSTREAM_MM


PRESETS: tuple[PresetDef, ...] = (
    PresetDef(PRESET_DEFAULT, "Full trajectory (3D)"),
    PresetDef(
        PRESET_PLAN_FOCUS,
        "Plan overlay focus",
        show_plan=True,
        show_pivot_markers=True,
        show_iso_planes=True,
        extend_upstream_mm=1500.0,
        extend_downstream_mm=2500.0,
    ),
    PresetDef(
        PRESET_UPSTREAM,
        "Upstream magnet region",
        show_plan=False,
        show_pivot_markers=True,
        show_iso_planes=False,
        extend_upstream_mm=3000.0,
        extend_downstream_mm=500.0,
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}


@dataclass
class TrajectoryConfig:
    show_spots: bool = True
    show_plan: bool = True
    show_pivot_markers: bool = True
    show_iso_planes: bool = True
    show_ic_planes: bool = True
    extend_upstream_mm: float = DEFAULT_EXTEND_UPSTREAM_MM
    extend_downstream_mm: float = DEFAULT_EXTEND_DOWNSTREAM_MM
    pivot_margin_mm: float = DEFAULT_PIVOT_MARGIN_MM

    @property
    def title(self) -> str:
        return "IC Beam Trajectory (3D)"
