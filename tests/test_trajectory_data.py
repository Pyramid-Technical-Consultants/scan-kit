"""Tests for 3D trajectory segment builders."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.common.trajectory_fits import fit_iso_plane, fit_magnet_pivot
from scan_kit.views.trajectory_data import (
    build_trajectory_session,
    plan_segments_3d,
    segment_z_extent,
    spot_segments_3d,
    TrajectorySession,
)


def _synthetic_session() -> TrajectorySession:
    z_pivot = IC2_Z_MM - 1000.0
    z_iso = IC2_Z_MM + 600.0
    angles = np.linspace(-0.02, 0.02, 50)
    ic2 = angles * (IC2_Z_MM - z_pivot)
    ic1 = angles * (IC1_Z_MM - z_pivot)
    plan = ic2 + (z_iso - IC2_Z_MM) * (ic1 - ic2) / IC_SEP_MM
    magnet_x = fit_magnet_pivot(ic2, ic1)
    magnet_y = fit_magnet_pivot(ic2, ic1)
    iso_x = fit_iso_plane(ic2, ic1, plan, magnet_x.z_pivot)
    iso_y = fit_iso_plane(ic2, ic1, plan, magnet_y.z_pivot)
    return TrajectorySession(
        session_id="test",
        ic2_x=ic2,
        ic2_y=ic2,
        ic1_x=ic1,
        ic1_y=ic1,
        plan_x=plan,
        plan_y=plan,
        energy=np.linspace(70, 80, ic2.size),
        align_ic2_x=0.0,
        align_ic1_x=0.0,
        align_ic2_y=0.0,
        align_ic1_y=0.0,
        magnet_x=magnet_x,
        magnet_y=magnet_y,
        iso_x=iso_x,
        iso_y=iso_y,
    )


def test_spot_segments_3d_endpoint_geometry() -> None:
    session = _synthetic_session()
    extend_up = 50.0
    extend_dn = 75.0
    segments = spot_segments_3d(
        session,
        extend_upstream_mm=extend_up,
        extend_downstream_mm=extend_dn,
    )
    assert segments.shape == (session.n_spots, 2, 3)
    z_start, z_end = segment_z_extent(
        session,
        extend_upstream_mm=extend_up,
        extend_downstream_mm=extend_dn,
    )
    np.testing.assert_allclose(segments[:, 0, 2], z_start)
    np.testing.assert_allclose(segments[:, 1, 2], z_end)
    assert z_start == session.magnet_x.z_pivot
    assert z_end == session.iso_x.z_iso
    sx = (session.ic1_x - session.ic2_x) / IC_SEP_MM
    sy = (session.ic1_y - session.ic2_y) / IC_SEP_MM
    np.testing.assert_allclose(
        segments[:, 0, 0],
        session.ic2_x + sx * (z_start - IC2_Z_MM),
    )
    np.testing.assert_allclose(
        segments[:, 0, 1],
        session.ic2_y + sy * (z_start - IC2_Z_MM),
    )


def test_plan_segments_3d_matches_spot_extents() -> None:
    session = _synthetic_session()
    extend_up = 40.0
    extend_dn = 60.0
    plan_segs = plan_segments_3d(
        session,
        extend_upstream_mm=extend_up,
        extend_downstream_mm=extend_dn,
    )
    assert plan_segs is not None
    z_start, z_end = segment_z_extent(
        session,
        extend_upstream_mm=extend_up,
        extend_downstream_mm=extend_dn,
    )
    np.testing.assert_allclose(plan_segs[:, 0, 2], z_start)
    np.testing.assert_allclose(plan_segs[:, 1, 2], z_end)
    np.testing.assert_allclose(plan_segs[:, 0, 0], 0.0)
    np.testing.assert_allclose(plan_segs[:, 0, 1], 0.0)
    np.testing.assert_allclose(plan_segs[:, 1, 0], session.plan_x)
    np.testing.assert_allclose(plan_segs[:, 1, 1], session.plan_y)


def _dual_dipole_session() -> TrajectorySession:
    z_pivot_x = IC2_Z_MM - 1200.0
    z_pivot_y = IC2_Z_MM - 1800.0
    z_iso_x = IC2_Z_MM + 600.0
    z_iso_y = IC2_Z_MM + 720.0
    angles_x = np.linspace(-0.02, 0.02, 40)
    angles_y = np.linspace(-0.015, 0.015, 40)
    ic2_x = angles_x * (IC2_Z_MM - z_pivot_x)
    ic1_x = angles_x * (IC1_Z_MM - z_pivot_x)
    ic2_y = angles_y * (IC2_Z_MM - z_pivot_y)
    ic1_y = angles_y * (IC1_Z_MM - z_pivot_y)
    plan_x = ic2_x + (z_iso_x - IC2_Z_MM) * (ic1_x - ic2_x) / IC_SEP_MM
    plan_y = ic2_y + (z_iso_y - IC2_Z_MM) * (ic1_y - ic2_y) / IC_SEP_MM
    magnet_x = fit_magnet_pivot(ic2_x, ic1_x)
    magnet_y = fit_magnet_pivot(ic2_y, ic1_y)
    iso_x = fit_iso_plane(ic2_x, ic1_x, plan_x, magnet_x.z_pivot)
    iso_y = fit_iso_plane(ic2_y, ic1_y, plan_y, magnet_y.z_pivot)
    return TrajectorySession(
        session_id="dual",
        ic2_x=ic2_x,
        ic2_y=ic2_y,
        ic1_x=ic1_x,
        ic1_y=ic1_y,
        plan_x=plan_x,
        plan_y=plan_y,
        energy=np.linspace(70, 80, ic2_x.size),
        align_ic2_x=0.0,
        align_ic1_x=0.0,
        align_ic2_y=0.0,
        align_ic1_y=0.0,
        magnet_x=magnet_x,
        magnet_y=magnet_y,
        iso_x=iso_x,
        iso_y=iso_y,
    )


def test_dual_dipole_segment_extent_uses_extreme_pivots_and_isos() -> None:
    session = _dual_dipole_session()
    z_start, z_end = segment_z_extent(session, extend_upstream_mm=100.0, extend_downstream_mm=100.0)
    assert z_start == session.magnet_y.z_pivot
    assert z_end == session.iso_y.z_iso
    plan_segs = plan_segments_3d(session, extend_upstream_mm=100.0, extend_downstream_mm=100.0)
    assert plan_segs is not None
    np.testing.assert_allclose(plan_segs[:, 0, 1], 0.0)
    np.testing.assert_allclose(plan_segs[:, 1, 0], session.plan_x)
    np.testing.assert_allclose(plan_segs[:, 1, 1], session.plan_y)


def test_build_trajectory_session_from_g3(test_data_dir: str, g3_session_id: str) -> None:
    from scan_kit.common import try_load_position_data
    from scan_kit.views.trajectory_data import _process_session

    data = try_load_position_data(
        g3_session_id,
        test_data_dir,
        lambda sid, key, base: _process_session(sid, key, base),
        raw=True,
    )
    assert data is not None
    session = build_trajectory_session(g3_session_id, data)
    assert session is not None
    assert session.n_spots > 0
    segs = spot_segments_3d(session, extend_upstream_mm=100.0, extend_downstream_mm=100.0)
    assert segs.shape[0] == session.n_spots
