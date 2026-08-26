"""D2-650-style dual dipole scan magnet geometry for pencil-beam trajectories.

Beam coordinates use IC2 as ``z = 0`` with downstream ``+z`` (see ``ic_trajectory``).

Physical layout (D2-650-IE datasheet):

* **Magnet 1 — X axis** — further upstream (beam entrance).
* **Magnet 2 — Y axis** — downstream of magnet 1 along the beam.
* **One isocenter plane** — all spots share a single treatment isocenter (SAD).
* Per-axis **SAD** on the datasheet is the distance from each magnet *geometric
  center* to that same isocenter plane.  Therefore axial magnet separation is
  ``SAD_X - SAD_Y`` for a given scan-field configuration.

Each dipole is a straight deflection magnet: small-angle ``sin θ ≈ L_eff / ρ``
with ``ρ = p/(qB)``.  IC spot fans back-project to a *virtual pivot* upstream of
the magnet.  For visualization and SAD, the magnet **geometric center** (midpoint
of the pole length along the beam) sits ``pole_length / 2`` downstream of that
fitted pivot so the upstream pole face meets the virtual source plane.  ``L_eff``
(from the datasheet field integral) is retained for angle physics but is longer
than the mechanical pole length.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np

from .ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from .trajectory_fits import project_plan_to_z

if TYPE_CHECKING:
    from .trajectory_fits import IsoFit, MagnetFit

# D2-650-IE nominal effective lengths (field integral / central gap field at 400 A).
D2_650_L_EFF_X_MM = 0.17 / 0.599 * 1000.0  # axis-1 (X) ≈ 283 mm
D2_650_L_EFF_Y_MM = 0.246 / 0.70 * 1000.0  # axis-2 (Y) ≈ 351 mm

# Example 35 cm × 40 cm field SADs to the *same* isocenter (datasheet).
D2_650_SAD_35X40_MM = {"x": 2280.0, "y": 1880.0}

# Pole geometry from D2-650-IE datasheet (axis 1 = X, axis 2 = Y).
D2_650_POLE_LENGTH_X_MM = 225.0
D2_650_POLE_GAP_X_MM = 60.0
D2_650_POLE_LENGTH_Y_MM = 275.0
D2_650_POLE_GAP_Y_MM = 72.0

# Visual pole-block depth (face thickness; not specified — schematic only).
D2_650_POLE_BLOCK_DEPTH_MM = 48.0
D2_650_POLE_CROSS_WIDTH_MM = 140.0


def dipole_virtual_pivot_upstream_mm(effective_length_mm: float) -> float:
    """Hard-edge dipole: virtual pivot is ``L_eff / 2`` upstream of field center."""
    return effective_length_mm / 2.0


def virtual_pivot_to_magnet_center_z(
    z_virtual_pivot: float,
    pole_length_mm: float,
) -> float:
    """Geometric magnet center (pole midplane) downstream of the virtual pivot."""
    return z_virtual_pivot + pole_length_mm / 2.0


def magnet_center_to_isocenter_sad_mm(z_magnet_center: float, z_isocenter: float) -> float:
    """Source-to-axis distance from magnet center to isocenter (downstream ``+z``)."""
    return z_isocenter - z_magnet_center


def datasheet_magnet_separation_mm(sad_x_mm: float, sad_y_mm: float) -> float:
    """Axial center-to-center spacing from per-axis SADs to a common isocenter."""
    return sad_x_mm - sad_y_mm


def combined_isocenter_z(
    iso_x: IsoFit | None,
    iso_y: IsoFit | None,
) -> float:
    """Single isocenter depth — median of per-axis IC/plan crossing estimates."""
    vals: list[float] = []
    if iso_x is not None and iso_x.is_valid:
        vals.append(iso_x.z_iso)
    if iso_y is not None and iso_y.is_valid:
        vals.append(iso_y.z_iso)
    if not vals:
        return float("nan")
    return float(np.median(vals))


@dataclass(frozen=True)
class DualDipoleGeometry:
    """Fitted dual-dipole layout with one shared isocenter plane."""

    z_virtual_pivot_x: float
    z_virtual_pivot_y: float
    z_isocenter: float
    z_magnet_center_x: float
    z_magnet_center_y: float
    iso_x_sigma_mm: float = float("nan")
    iso_y_sigma_mm: float = float("nan")

    @property
    def is_valid(self) -> bool:
        return (
            np.isfinite(self.z_virtual_pivot_x)
            and np.isfinite(self.z_virtual_pivot_y)
            and np.isfinite(self.z_isocenter)
            and self.z_isocenter > IC2_Z_MM
        )

    @property
    def sad_x_mm(self) -> float:
        return magnet_center_to_isocenter_sad_mm(self.z_magnet_center_x, self.z_isocenter)

    @property
    def sad_y_mm(self) -> float:
        return magnet_center_to_isocenter_sad_mm(self.z_magnet_center_y, self.z_isocenter)

    @property
    def magnet_separation_mm(self) -> float:
        return self.z_magnet_center_y - self.z_magnet_center_x

    @property
    def x_magnet_upstream_of_y(self) -> bool:
        """X dipole (magnet 1) should sit upstream of Y (magnet 2)."""
        return self.z_magnet_center_x < self.z_magnet_center_y


def build_dual_dipole_geometry(
    magnet_x: MagnetFit,
    magnet_y: MagnetFit,
    iso_x: IsoFit | None,
    iso_y: IsoFit | None,
    *,
    pole_length_x_mm: float = D2_650_POLE_LENGTH_X_MM,
    pole_length_y_mm: float = D2_650_POLE_LENGTH_Y_MM,
) -> DualDipoleGeometry | None:
    """Construct physics layout from IC trajectory fits."""
    if not magnet_x.is_valid or not magnet_y.is_valid:
        return None
    z_iso = combined_isocenter_z(iso_x, iso_y)
    if not np.isfinite(z_iso):
        return None

    z_cx = virtual_pivot_to_magnet_center_z(magnet_x.z_pivot, pole_length_x_mm)
    z_cy = virtual_pivot_to_magnet_center_z(magnet_y.z_pivot, pole_length_y_mm)
    sig_x = iso_x.downstream_sigma_mm if iso_x is not None and iso_x.is_valid else float("nan")
    sig_y = iso_y.downstream_sigma_mm if iso_y is not None and iso_y.is_valid else float("nan")

    return DualDipoleGeometry(
        z_virtual_pivot_x=magnet_x.z_pivot,
        z_virtual_pivot_y=magnet_y.z_pivot,
        z_isocenter=z_iso,
        z_magnet_center_x=z_cx,
        z_magnet_center_y=z_cy,
        iso_x_sigma_mm=sig_x,
        iso_y_sigma_mm=sig_y,
    )


def beam_lateral_mm(
    p2: np.ndarray,
    p1: np.ndarray,
    z_mm: float | np.ndarray,
    *,
    ic_sep_mm: float = IC_SEP_MM,
) -> np.ndarray:
    """Lateral position along a measured IC2→IC1 ray at downstream *z* (mm from IC2)."""
    p2 = np.asarray(p2, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    slope = (p1 - p2) / ic_sep_mm
    z = np.asarray(z_mm, dtype=float)
    return p2 + slope * (z - IC2_Z_MM)


def segment_z_bounds(
    magnet_x: MagnetFit | None,
    magnet_y: MagnetFit | None,
    iso_x: IsoFit | None,
    iso_y: IsoFit | None,
    *,
    extend_upstream_mm: float,
    extend_downstream_mm: float,
) -> tuple[float, float]:
    """Clip range: furthest upstream virtual pivot → shared isocenter plane."""
    z_start = IC2_Z_MM - extend_upstream_mm
    z_end = IC1_Z_MM + extend_downstream_mm

    pivots: list[float] = []
    if magnet_x is not None and magnet_x.is_valid:
        pivots.append(magnet_x.z_pivot)
    if magnet_y is not None and magnet_y.is_valid:
        pivots.append(magnet_y.z_pivot)
    if pivots:
        z_start = min(pivots)

    z_iso = combined_isocenter_z(iso_x, iso_y)
    if np.isfinite(z_iso):
        z_end = z_iso

    if z_end < z_start:
        z_end = z_start
    return z_start, z_end


def plan_lateral_at_z(
    plan_mm: np.ndarray,
    z_pivot: float,
    z_isocenter: float,
    z_mm: float,
) -> np.ndarray:
    """Plan nominal lateral position at *z* for one scan axis (pivot → isocenter)."""
    if not np.isfinite(z_pivot) or not np.isfinite(z_isocenter) or abs(z_isocenter - z_pivot) < 1e-3:
        return np.full(plan_mm.shape, np.nan, dtype=float)
    return project_plan_to_z(plan_mm, z_pivot, z_isocenter, z_mm)


@dataclass(frozen=True)
class PoleBoxSpec:
    """Axis-aligned box in scene coordinates (beam X, lateral Y, lateral Z)."""

    center_x: float
    center_y: float
    center_z: float
    width_x: float
    width_y: float
    width_z: float


def x_dipole_pole_boxes(
    z_magnet_center: float,
    *,
    pole_length_mm: float = D2_650_POLE_LENGTH_X_MM,
    gap_mm: float = D2_650_POLE_GAP_X_MM,
    pole_depth_mm: float = D2_650_POLE_BLOCK_DEPTH_MM,
    cross_width_mm: float = D2_650_POLE_CROSS_WIDTH_MM,
) -> tuple[PoleBoxSpec, PoleBoxSpec]:
    """Two pole blocks flanking the X-dipole gap (opening along lateral Y)."""
    half_gap = gap_mm / 2.0
    half_depth = pole_depth_mm / 2.0
    offset = half_gap + half_depth
    size = (pole_length_mm, pole_depth_mm, cross_width_mm)
    return (
        PoleBoxSpec(z_magnet_center, -offset, 0.0, *size),
        PoleBoxSpec(z_magnet_center, offset, 0.0, *size),
    )


def y_dipole_pole_boxes(
    z_magnet_center: float,
    *,
    pole_length_mm: float = D2_650_POLE_LENGTH_Y_MM,
    gap_mm: float = D2_650_POLE_GAP_Y_MM,
    pole_depth_mm: float = D2_650_POLE_BLOCK_DEPTH_MM,
    cross_width_mm: float = D2_650_POLE_CROSS_WIDTH_MM,
) -> tuple[PoleBoxSpec, PoleBoxSpec]:
    """Two pole blocks flanking the Y-dipole gap (opening along lateral Z)."""
    half_gap = gap_mm / 2.0
    half_depth = pole_depth_mm / 2.0
    offset = half_gap + half_depth
    size = (pole_length_mm, cross_width_mm, pole_depth_mm)
    return (
        PoleBoxSpec(z_magnet_center, 0.0, -offset, *size),
        PoleBoxSpec(z_magnet_center, 0.0, offset, *size),
    )


def dual_dipole_pole_boxes(geom: DualDipoleGeometry) -> tuple[PoleBoxSpec, ...]:
    """Four pole blocks (two per dipole) at fitted magnet centers."""
    return (
        *x_dipole_pole_boxes(geom.z_magnet_center_x),
        *y_dipole_pole_boxes(geom.z_magnet_center_y),
    )


def magnet_field_bounds_z(
    z_magnet_center: float,
    pole_length_mm: float,
) -> tuple[float, float]:
    """Approximate field extent along the beam (pole face to pole face)."""
    half = pole_length_mm / 2.0
    return z_magnet_center - half, z_magnet_center + half


def trace_knots_z(
    z_start: float,
    z_end: float,
    geom: DualDipoleGeometry,
) -> np.ndarray:
    """Ordered ``z`` samples for piecewise beam polylines (clipped to clip range)."""
    z_x_lo, z_x_hi = magnet_field_bounds_z(
        geom.z_magnet_center_x,
        D2_650_POLE_LENGTH_X_MM,
    )
    z_y_lo, z_y_hi = magnet_field_bounds_z(
        geom.z_magnet_center_y,
        D2_650_POLE_LENGTH_Y_MM,
    )
    candidates = [
        z_start,
        z_end,
        geom.z_virtual_pivot_x,
        geom.z_virtual_pivot_y,
        geom.z_magnet_center_x,
        geom.z_magnet_center_y,
        z_x_lo,
        z_x_hi,
        z_y_lo,
        z_y_hi,
        geom.z_isocenter,
    ]
    knots = sorted(
        {float(z) for z in candidates if np.isfinite(z) and z_start <= z <= z_end},
    )
    if not knots:
        return np.array([z_start, z_end], dtype=float)
    if knots[0] > z_start:
        knots.insert(0, z_start)
    if knots[-1] < z_end:
        knots.append(z_end)
    return np.asarray(knots, dtype=float)


def measured_lateral_xy(
    z_mm: np.ndarray,
    ic2_x: float,
    ic1_x: float,
    ic2_y: float,
    ic1_y: float,
    z_pivot_x: float,
    z_pivot_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential dual-dipole trace: X deflection from X pivot, Y from Y pivot.

    Matches orthogonal dipoles: no Y steering until the Y magnet region; no X
  steering before the X magnet.  Uses IC2→IC1 slopes downstream of each pivot
    (alignment-corrected positions).
    """
    z = np.asarray(z_mm, dtype=float)
    sx = (ic1_x - ic2_x) / IC_SEP_MM
    sy = (ic1_y - ic2_y) / IC_SEP_MM
    x = np.where(
        z < z_pivot_x,
        0.0,
        ic2_x + sx * (z - IC2_Z_MM),
    )
    y = np.where(
        z < z_pivot_y,
        0.0,
        ic2_y + sy * (z - IC2_Z_MM),
    )
    return x, y


def plan_lateral_xy(
    z_mm: np.ndarray,
    plan_x: float,
    plan_y: float,
    z_pivot_x: float,
    z_pivot_y: float,
    z_isocenter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan ray with the same sequential pivot activation as measured spots."""
    z = np.asarray(z_mm, dtype=float)
    denom_x = z_isocenter - z_pivot_x
    denom_y = z_isocenter - z_pivot_y
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    if np.isfinite(z_pivot_x) and np.isfinite(z_isocenter) and abs(denom_x) > 1e-3:
        active = z >= z_pivot_x
        x[active] = plan_x * (z[active] - z_pivot_x) / denom_x
    if np.isfinite(z_pivot_y) and np.isfinite(z_isocenter) and abs(denom_y) > 1e-3:
        active = z >= z_pivot_y
        y[active] = plan_y * (z[active] - z_pivot_y) / denom_y
    return x, y


def reference_on_axis_trace(
    z_start: float,
    z_end: float,
    geom: DualDipoleGeometry,
) -> np.ndarray:
    """On-axis ``(x, y, z)`` polyline through magnet field envelopes (no spot angle)."""
    knots_z = trace_knots_z(z_start, z_end, geom)
    n = knots_z.size
    return np.column_stack(
        [np.zeros(n, dtype=float), np.zeros(n, dtype=float), knots_z],
    )


def spot_traces_3d_arrays(
    ic2_x: np.ndarray,
    ic1_x: np.ndarray,
    ic2_y: np.ndarray,
    ic1_y: np.ndarray,
    geom: DualDipoleGeometry,
    knots_z: np.ndarray,
) -> np.ndarray:
    """Build ``(n_spots, n_knots, 3)`` beam-coordinate polylines."""
    n = ic2_x.size
    k = knots_z.size
    out = np.empty((n, k, 3), dtype=float)
    z_px = geom.z_virtual_pivot_x
    z_py = geom.z_virtual_pivot_y
    for i in range(n):
        x, y = measured_lateral_xy(
            knots_z,
            float(ic2_x[i]),
            float(ic1_x[i]),
            float(ic2_y[i]),
            float(ic1_y[i]),
            z_px,
            z_py,
        )
        out[i, :, 0] = x
        out[i, :, 1] = y
        out[i, :, 2] = knots_z
    return out
