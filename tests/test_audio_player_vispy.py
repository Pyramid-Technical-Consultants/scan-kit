"""Tests for IC audio player visPy scene."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from scan_kit.views.audio_player_data import WaveformRenderChannel
from scan_kit.views.audio_player_vispy import AudioWaveformScene, _envelope_polygon


def _sample_render_channel() -> WaveformRenderChannel:
    signal = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    envelope_poly = np.array(
        [[0.0, -1.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    return WaveformRenderChannel(
        label="IC1",
        channel_id="ic1",
        signal=signal,
        color="#1f77b4",
        envelope_poly=envelope_poly,
        y_lo=-1.1,
        y_hi=1.1,
    )


def test_envelope_polygon_is_closed_ring() -> None:
    t = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    y_min = np.array([-1.0, -0.5, -0.2])
    y_max = np.array([1.0, 0.5, 0.2])
    poly = _envelope_polygon(t, y_min, y_max)
    assert poly.shape == (6, 2)
    np.testing.assert_allclose(poly[0], [0.0, 1.0])
    np.testing.assert_allclose(poly[-1], [0.0, -1.0])


def test_audio_waveform_scene_clear_and_status() -> None:
    canvas = MagicMock()
    grid = MagicMock()
    grid.children = []
    canvas.central_widget.add_grid.return_value = grid
    view = MagicMock()
    grid.add_view.return_value = view

    import vispy.scene

    with (
        patch.object(vispy.scene.cameras, "PanZoomCamera", MagicMock),
        patch.object(vispy.scene.visuals, "Polygon", MagicMock),
        patch.object(vispy.scene.visuals, "Line", MagicMock),
        patch.object(vispy.scene, "Text", MagicMock),
    ):
        scene = AudioWaveformScene(canvas)
        scene.show_status("Loading")
        assert scene._status_node is not None
        scene.clear()
        assert scene._rows == []


def test_audio_waveform_scene_set_render_channels() -> None:
    canvas = MagicMock()
    grid = MagicMock()
    grid.children = []
    canvas.central_widget.add_grid.return_value = grid
    view = MagicMock()
    view.scene = MagicMock()
    grid.add_view.return_value = view

    import vispy.scene

    polygon_mock = MagicMock()
    line_mock = MagicMock()
    with (
        patch.object(vispy.scene.cameras, "PanZoomCamera", MagicMock),
        patch.object(
            vispy.scene.visuals, "Polygon", MagicMock(return_value=polygon_mock),
        ),
        patch.object(vispy.scene.visuals, "Line", MagicMock(return_value=line_mock)),
        patch.object(vispy.scene, "Text", MagicMock),
        patch.object(vispy.scene, "Axis", MagicMock),
    ):
        scene = AudioWaveformScene(canvas)
        scene.set_render_channels(
            [_sample_render_channel()],
            selected_index=0,
            cursor_time=0.1,
        )
        assert len(scene._rows) == 1
        assert scene.duration > 0.0
        polygon_mock.set_gl_state.assert_called_once()
