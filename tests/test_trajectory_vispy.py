"""Tests for trajectory visPy scene behavior."""

from __future__ import annotations

import numpy as np

from unittest.mock import MagicMock, patch

from scan_kit.views.trajectory_vispy import (
    TrajectoryScene,
    _beam_to_scene,
    _energy_colors,
    _energy_marker_colors,
    _grid_major_coord,
    _iso_plane_grid_line_sets,
    _marker_face_rgba,
    _plane_spot_marker_positions,
    _segments_to_line_pos,
    _trace_vertex_colors,
)
from scan_kit.views.trajectory_catalog import (
    ENERGY_RAY_ALPHA,
    ISO_GRID_MAJOR_STEP_MM,
    ISO_GRID_STEP_MM,
    SPOT_RAY_ALPHA,
)


def test_trajectory_scene_clear_preserves_beam_axis() -> None:
    view = MagicMock(scene=MagicMock())
    canvas = MagicMock()
    canvas.central_widget.add_view.return_value = view

    import vispy.scene

    with (
        patch.object(vispy.scene.cameras, "TurntableCamera", MagicMock),
        patch.object(vispy.scene.visuals, "Line", MagicMock),
    ):
        scene = TrajectoryScene(canvas)

    assert scene._axis is not None
    scene._nodes.append(MagicMock())
    scene.clear()
    assert scene._nodes == []
    assert scene._axis is not None


def test_energy_colors_accepts_read_only_energy() -> None:
    energy = np.array([70.0, 71.0, np.nan], dtype=np.float64)
    energy.setflags(write=False)
    colors = _energy_colors(energy)
    assert colors.shape == (3, 4)
    assert colors.dtype == np.float32


def test_marker_face_rgba_keeps_rgb_and_is_opaque() -> None:
    rgba = _marker_face_rgba((0.2, 0.4, 0.6, SPOT_RAY_ALPHA))
    assert rgba == (0.2, 0.4, 0.6, 1.0)


def test_energy_marker_colors_are_opaque() -> None:
    energy = np.array([70.0, 71.0], dtype=np.float64)
    line_colors = _energy_colors(energy)
    marker_colors = _energy_marker_colors(energy)
    np.testing.assert_allclose(marker_colors[:, :3], line_colors[:, :3])
    np.testing.assert_allclose(marker_colors[:, 3], 1.0)
    np.testing.assert_allclose(line_colors[:, 3], ENERGY_RAY_ALPHA)


def test_trace_vertex_colors_matches_polyline_vertices() -> None:
    energy = np.array([70.0, 71.0, np.nan], dtype=np.float64)
    colors = _trace_vertex_colors(energy, knots_per_trace=5)
    assert colors.shape == (15, 4)


def test_segments_to_line_pos_supports_polyline_knots() -> None:
    polylines = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 100.0], [2.0, 1.0, 200.0]],
        ],
        dtype=float,
    )
    pos, connect = _segments_to_line_pos(polylines)
    assert pos.shape == (3, 3)
    assert connect.shape == (2, 2)
    assert connect[0].tolist() == [0, 1]
    assert connect[1].tolist() == [1, 2]


def test_plane_spot_marker_positions_maps_to_scene_coords() -> None:
    lx = np.array([1.0, np.nan, -2.0])
    ly = np.array([3.0, 4.0, -5.0])
    pos = _plane_spot_marker_positions(lx, ly, 100.0)
    assert pos.shape == (2, 3)
    np.testing.assert_allclose(pos[:, 0], 100.0)
    np.testing.assert_allclose(pos[0], [100.0, 1.0, 3.0])
    np.testing.assert_allclose(pos[1], [100.0, -2.0, -5.0])


def test_iso_plane_grid_has_1cm_minor_and_10cm_major_lines() -> None:
    hw, hh = 150.0, 200.0
    minor_pos, minor_conn, major_pos, major_conn = _iso_plane_grid_line_sets(
        500.0,
        hw,
        hh,
        step_mm=ISO_GRID_STEP_MM,
        major_step_mm=ISO_GRID_MAJOR_STEP_MM,
    )
    assert minor_conn.shape[0] == 64
    assert major_conn.shape[0] == 8
    assert minor_conn.shape[0] + major_conn.shape[0] == 72
    assert _grid_major_coord(0.0, ISO_GRID_MAJOR_STEP_MM)
    assert _grid_major_coord(100.0, ISO_GRID_MAJOR_STEP_MM)
    assert not _grid_major_coord(50.0, ISO_GRID_MAJOR_STEP_MM)


def test_beam_to_scene_maps_downstream_to_vispy_x() -> None:
    segments = np.array(
        [
            [[1.0, 2.0, 100.0], [3.0, 4.0, 200.0]],
        ],
        dtype=float,
    )
    scene = _beam_to_scene(segments)
    np.testing.assert_allclose(scene[0, 0], [100.0, 1.0, 2.0])
    np.testing.assert_allclose(scene[0, 1], [200.0, 3.0, 4.0])
