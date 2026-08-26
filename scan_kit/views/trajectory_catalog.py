"""Presets and display options for the 3D IC beam trajectory viewer."""

from __future__ import annotations

from dataclasses import dataclass

PRESET_DEFAULT = "default"
PRESET_PLAN_FOCUS = "plan_focus"
PRESET_UPSTREAM = "upstream_zoom"

DEFAULT_EXTEND_UPSTREAM_MM = 2000.0
DEFAULT_EXTEND_DOWNSTREAM_MM = 2000.0
DEFAULT_PIVOT_MARGIN_MM = 100.0

# Ray / marker opacity in the 3D scene.
SPOT_RAY_ALPHA = 0.07
PLAN_RAY_ALPHA = 0.22
ENERGY_RAY_ALPHA = 0.28
PLANE_SPOT_MARKER_SIZE = 12.0
# Nudge markers slightly downstream of plane meshes to avoid z-fighting halos.
PLANE_SPOT_MARKER_Z_EPSILON_MM = 0.3

# Treatment isocenter plane outline (30 cm × 40 cm); height is lateral Y (vertical).
ISO_PLANE_WIDTH_MM = 300.0
ISO_PLANE_HEIGHT_MM = 400.0
ISO_PLANE_HALF_WIDTH_MM = ISO_PLANE_WIDTH_MM / 2.0
ISO_PLANE_HALF_HEIGHT_MM = ISO_PLANE_HEIGHT_MM / 2.0

# Isocenter plane grid (1 cm minor lines, 10 cm major lines).
ISO_GRID_STEP_MM = 10.0
ISO_GRID_MAJOR_STEP_MM = 100.0


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    show_spot_lines: bool = True
    show_spot_markers: bool = True
    show_plan_lines: bool = False
    show_plan_markers: bool = False
    show_pivot_markers: bool = True
    show_iso_planes: bool = True
    show_magnet_gaps: bool = True
    show_ic_planes: bool = True
    extend_upstream_mm: float = DEFAULT_EXTEND_UPSTREAM_MM
    extend_downstream_mm: float = DEFAULT_EXTEND_DOWNSTREAM_MM


PRESETS: tuple[PresetDef, ...] = (
    PresetDef(PRESET_DEFAULT, "Full trajectory (3D)"),
    PresetDef(
        PRESET_PLAN_FOCUS,
        "Plan overlay focus",
        show_plan_lines=True,
        show_plan_markers=True,
        show_pivot_markers=True,
        show_iso_planes=True,
        extend_upstream_mm=1500.0,
        extend_downstream_mm=2500.0,
    ),
    PresetDef(
        PRESET_UPSTREAM,
        "Upstream magnet region",
        show_plan_lines=False,
        show_plan_markers=False,
        show_pivot_markers=True,
        show_iso_planes=False,
        extend_upstream_mm=3000.0,
        extend_downstream_mm=500.0,
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}


@dataclass
class TrajectoryConfig:
    show_spot_lines: bool = True
    show_spot_markers: bool = True
    show_plan_lines: bool = False
    show_plan_markers: bool = False
    show_pivot_markers: bool = True
    show_iso_planes: bool = True
    show_magnet_gaps: bool = True
    show_ic_planes: bool = True
    extend_upstream_mm: float = DEFAULT_EXTEND_UPSTREAM_MM
    extend_downstream_mm: float = DEFAULT_EXTEND_DOWNSTREAM_MM
    pivot_margin_mm: float = DEFAULT_PIVOT_MARGIN_MM

    @property
    def title(self) -> str:
        return "IC Beam Trajectory (3D)"
