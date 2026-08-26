"""Tests for trajectory visPy scene behavior."""

from __future__ import annotations

import numpy as np

from unittest.mock import MagicMock, patch

from scan_kit.views.trajectory_vispy import TrajectoryScene, _beam_to_scene, _energy_colors


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
