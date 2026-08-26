"""Tests for dual-dipole scan magnet geometry helpers."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.common.scan_magnet_model import (
    beam_lateral_mm,
    dipole_virtual_pivot_upstream_mm,
    plan_lateral_at_z,
    segment_z_bounds,
)
from scan_kit.common.trajectory_fits import IsoFit, MagnetFit


def test_dipole_virtual_pivot_is_half_effective_length() -> None:
    np.testing.assert_allclose(dipole_virtual_pivot_upstream_mm(240.0), 120.0)


def test_segment_z_bounds_uses_separate_x_y_pivots_and_isos() -> None:
    magnet_x = MagnetFit(-1200.0, IC2_Z_MM - (-1200.0), 5.0)
    magnet_y = MagnetFit(-1800.0, IC2_Z_MM - (-1800.0), 8.0)
    iso_x = IsoFit(600.0, 600.0, 10.0)
    iso_y = IsoFit(750.0, 750.0, 12.0)

    z_start, z_end = segment_z_bounds(
        magnet_x,
        magnet_y,
        iso_x,
        iso_y,
        extend_upstream_mm=500.0,
        extend_downstream_mm=500.0,
    )
    assert z_start == -1800.0
    assert z_end == 750.0


def test_plan_lateral_at_z_uses_per_axis_pivots() -> None:
    plan = np.array([3.0, -2.0])
    z_px, z_ix = -1000.0, 500.0
    z_py, z_iy = -1500.0, 650.0
    x_at_x_pivot = plan_lateral_at_z(plan, z_px, z_ix, z_px)
    y_at_y_pivot = plan_lateral_at_z(plan, z_py, z_iy, z_py)
    np.testing.assert_allclose(x_at_x_pivot, 0.0)
    np.testing.assert_allclose(y_at_y_pivot, 0.0)


def test_beam_lateral_mm_matches_ic_extrapolation() -> None:
    z_pivot = IC2_Z_MM - 1000.0
    angles = np.linspace(-0.02, 0.02, 20)
    ic2 = angles * (IC2_Z_MM - z_pivot)
    ic1 = angles * (IC1_Z_MM - z_pivot)
    z = IC2_Z_MM - 500.0
    expected = ic2 + (ic1 - ic2) / IC_SEP_MM * (z - IC2_Z_MM)
    np.testing.assert_allclose(beam_lateral_mm(ic2, ic1, z), expected)
