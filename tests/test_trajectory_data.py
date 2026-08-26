"""Tests for 3D trajectory segment builders."""

from __future__ import annotations

import numpy as np

from scan_kit.common.ic_trajectory import IC1_Z_MM, IC2_Z_MM, IC_SEP_MM
from scan_kit.common.trajectory_fits import fit_iso_plane, fit_magnet_pivot
from scan_kit.views.trajectory_data import (
    build_trajectory_session,
    plan_segments_3d,
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
    z_up = IC2_Z_MM - extend_up
    z_dn = IC1_Z_MM + extend_dn
    np.testing.assert_allclose(segments[:, 0, 2], z_up)
    np.testing.assert_allclose(segments[:, 1, 2], z_dn)
    sx = (session.ic1_x - session.ic2_x) / IC_SEP_MM
    sy = (session.ic1_y - session.ic2_y) / IC_SEP_MM
    np.testing.assert_allclose(
        segments[:, 0, 0],
        session.ic2_x + sx * (z_up - IC2_Z_MM),
    )
    np.testing.assert_allclose(
        segments[:, 0, 1],
        session.ic2_y + sy * (z_up - IC2_Z_MM),
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
    z_up = IC2_Z_MM - extend_up
    z_dn = IC1_Z_MM + extend_dn
    np.testing.assert_allclose(plan_segs[:, 0, 2], z_up)
    np.testing.assert_allclose(plan_segs[:, 1, 2], z_dn)


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
