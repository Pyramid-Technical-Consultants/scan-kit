"""Tests for dual-dipole scan magnet geometry helpers."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.common.scan_magnet_model import (
    beam_lateral_mm,
    build_dual_dipole_geometry,
    combined_isocenter_z,
    datasheet_magnet_separation_mm,
    dipole_virtual_pivot_upstream_mm,
    dual_dipole_pole_boxes,
    D2_650_POLE_GAP_X_MM,
    D2_650_POLE_GAP_Y_MM,
    D2_650_SAD_35X40_MM,
    plan_lateral_at_z,
    segment_z_bounds,
    virtual_pivot_to_magnet_center_z,
    x_dipole_pole_boxes,
    y_dipole_pole_boxes,
)
from scan_kit.common.scan_magnet_model import (
    build_dual_dipole_geometry,
    measured_lateral_xy,
    plan_lateral_xy,
    trace_knots_z,
)
from scan_kit.common.trajectory_fits import IsoFit, MagnetFit


def test_dipole_virtual_pivot_is_half_effective_length() -> None:
    np.testing.assert_allclose(dipole_virtual_pivot_upstream_mm(240.0), 120.0)


def test_combined_isocenter_is_single_plane() -> None:
    iso_x = IsoFit(600.0, 600.0, 10.0)
    iso_y = IsoFit(618.0, 618.0, 12.0)
    np.testing.assert_allclose(combined_isocenter_z(iso_x, iso_y), 609.0)


def test_segment_z_bounds_uses_upstream_pivots_and_shared_isocenter() -> None:
    magnet_x = MagnetFit(-1800.0, IC2_Z_MM - (-1800.0), 5.0)
    magnet_y = MagnetFit(-1400.0, IC2_Z_MM - (-1400.0), 8.0)
    iso_x = IsoFit(600.0, 600.0, 10.0)
    iso_y = IsoFit(620.0, 620.0, 12.0)

    z_start, z_end = segment_z_bounds(
        magnet_x,
        magnet_y,
        iso_x,
        iso_y,
        extend_upstream_mm=500.0,
        extend_downstream_mm=500.0,
    )
    assert z_start == -1800.0
    assert z_end == 610.0


def test_build_dual_dipole_geometry_sad_and_separation() -> None:
    magnet_x = MagnetFit(-2000.0, IC2_Z_MM - (-2000.0), 5.0)
    magnet_y = MagnetFit(-1600.0, IC2_Z_MM - (-1600.0), 8.0)
    z_iso = 2500.0
    iso_x = IsoFit(z_iso, z_iso - IC2_Z_MM, 10.0)
    iso_y = IsoFit(z_iso, z_iso - IC2_Z_MM, 12.0)
    geom = build_dual_dipole_geometry(magnet_x, magnet_y, iso_x, iso_y)
    assert geom is not None
    assert geom.x_magnet_upstream_of_y
    np.testing.assert_allclose(geom.z_isocenter, z_iso)
    np.testing.assert_allclose(
        geom.magnet_separation_mm,
        geom.sad_x_mm - geom.sad_y_mm,
    )
    np.testing.assert_allclose(
        datasheet_magnet_separation_mm(
            D2_650_SAD_35X40_MM["x"],
            D2_650_SAD_35X40_MM["y"],
        ),
        400.0,
    )


def test_plan_lateral_at_z_uses_per_axis_pivots_to_shared_iso() -> None:
    plan = np.array([3.0, -2.0])
    z_px, z_py, z_iso = -1000.0, -1500.0, 500.0
    x_at_x_pivot = plan_lateral_at_z(plan, z_px, z_iso, z_px)
    y_at_y_pivot = plan_lateral_at_z(plan, z_py, z_iso, z_py)
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


def test_virtual_pivot_to_center_is_downstream() -> None:
    z_pivot = -1000.0
    center = virtual_pivot_to_magnet_center_z(z_pivot, 240.0)
    assert center > z_pivot
    np.testing.assert_allclose(center - z_pivot, 120.0)


def test_fitted_pivot_meets_upstream_pole_face() -> None:
    magnet_x = MagnetFit(-2000.0, IC2_Z_MM - (-2000.0), 5.0)
    magnet_y = MagnetFit(-1600.0, IC2_Z_MM - (-1600.0), 8.0)
    iso = IsoFit(2500.0, 2500.0, 10.0)
    geom = build_dual_dipole_geometry(magnet_x, magnet_y, iso, iso)
    assert geom is not None
    x_boxes = x_dipole_pole_boxes(geom.z_magnet_center_x)
    upstream_face = min(b.center_x - b.width_x / 2.0 for b in x_boxes)
    np.testing.assert_allclose(upstream_face, magnet_x.z_pivot, rtol=0, atol=1e-9)


def test_dipole_pole_boxes_flank_gap() -> None:
    z_c = -500.0
    left, right = x_dipole_pole_boxes(z_c)
    left_inner_y = left.center_y + left.width_y / 2.0
    right_inner_y = right.center_y - right.width_y / 2.0
    np.testing.assert_allclose(right_inner_y - left_inner_y, D2_650_POLE_GAP_X_MM)
    assert left.center_x == z_c
    below, above = y_dipole_pole_boxes(z_c)
    below_inner_z = below.center_z + below.width_z / 2.0
    above_inner_z = above.center_z - above.width_z / 2.0
    np.testing.assert_allclose(above_inner_z - below_inner_z, D2_650_POLE_GAP_Y_MM)


def test_dual_dipole_pole_boxes_returns_four_blocks() -> None:
    magnet_x = MagnetFit(-2000.0, IC2_Z_MM - (-2000.0), 5.0)
    magnet_y = MagnetFit(-1600.0, IC2_Z_MM - (-1600.0), 8.0)
    iso = IsoFit(2500.0, 2500.0, 10.0)
    geom = build_dual_dipole_geometry(magnet_x, magnet_y, iso, iso)
    assert geom is not None
    assert len(dual_dipole_pole_boxes(geom)) == 4


def test_sequential_trace_has_no_y_deflection_before_y_pivot() -> None:
    geom = build_dual_dipole_geometry(
        MagnetFit(-2000.0, IC2_Z_MM - (-2000.0), 5.0),
        MagnetFit(-1500.0, IC2_Z_MM - (-1500.0), 5.0),
        IsoFit(2000.0, 2000.0, 5.0),
        IsoFit(2000.0, 2000.0, 5.0),
    )
    assert geom is not None
    z_mid = (geom.z_virtual_pivot_x + geom.z_virtual_pivot_y) / 2.0
    if geom.z_virtual_pivot_x < z_mid < geom.z_virtual_pivot_y:
        x, y = measured_lateral_xy(
            np.array([z_mid]),
            0.0,
            10.0,
            0.0,
            8.0,
            geom.z_virtual_pivot_x,
            geom.z_virtual_pivot_y,
        )
        assert abs(float(y[0])) < 1e-9
        assert abs(float(x[0])) > 0.0


def test_plan_lateral_xy_is_zero_at_respective_pivots() -> None:
    z_iso = 2000.0
    z_px, z_py = -1800.0, -1400.0
    x, y = plan_lateral_xy(
        np.array([z_px, z_py]),
        5.0,
        -3.0,
        z_px,
        z_py,
        z_iso,
    )
    np.testing.assert_allclose(x[0], 0.0)
    np.testing.assert_allclose(y[1], 0.0)
