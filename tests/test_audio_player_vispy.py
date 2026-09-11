"""Tests for IC audio player visPy scene."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scan_kit.views.audio_player_data import WaveformRenderChannel
from scan_kit.views.audio_player_vispy import (
    AudioSpectrumScene,
    AudioWaveformScene,
    _envelope_polygon,
)
from scan_kit.views.vispy_plot import envelope_mesh_geometry


def _sample_render_channel() -> WaveformRenderChannel:
    signal = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    envelope_poly = np.array(
        [[0.0, -1.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    line_pos = np.array([[0.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    return WaveformRenderChannel(
        label="IC1",
        channel_id="ic1",
        signal=signal,
        color="#1f77b4",
        envelope_poly=envelope_poly,
        line_pos=line_pos,
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
    pos, faces = envelope_mesh_geometry(poly)
    assert pos.shape == (6, 3)
    assert faces.shape == (4, 3)
    assert faces.max() < len(pos)


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
        patch.object(vispy.scene.visuals, "Mesh", MagicMock),
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
    mesh_mock = MagicMock()
    line_mock = MagicMock()
    with (
        patch.object(vispy.scene.cameras, "PanZoomCamera", MagicMock),
        patch.object(
            vispy.scene.visuals, "Polygon", MagicMock(return_value=polygon_mock),
        ),
        patch.object(
            vispy.scene.visuals, "Mesh", MagicMock(return_value=mesh_mock),
        ),
        patch.object(vispy.scene.visuals, "Line", MagicMock(return_value=line_mock)),
        patch.object(vispy.scene, "Text", MagicMock),
        patch.object(vispy.scene, "AxisWidget", MagicMock),
    ):
        scene = AudioWaveformScene(canvas)
        scene.set_render_channels(
            [_sample_render_channel()],
            selected_index=0,
            cursor_time=0.1,
        )
        assert len(scene._rows) == 1
        assert scene.duration > 0.0
        mesh_mock.set_gl_state.assert_called_once_with(
            "translucent", depth_test=False, depth_mask=False,
        )
        assert line_mock.order == 1


def test_time_at_canvas_pos_maps_x_linearly() -> None:
    canvas = MagicMock()
    canvas.size = (1000, 400)
    grid = MagicMock()
    grid.children = []
    canvas.central_widget.add_grid.return_value = grid
    view = MagicMock()
    view.scene = MagicMock()
    view.pos = (0.0, 0.0)
    view.size = (1000.0, 400.0)
    grid.add_view.return_value = view

    import vispy.scene

    with (
        patch.object(vispy.scene.cameras, "PanZoomCamera", MagicMock),
        patch.object(vispy.scene.visuals, "Polygon", MagicMock()),
        patch.object(vispy.scene.visuals, "Mesh", MagicMock()),
        patch.object(vispy.scene.visuals, "Line", MagicMock()),
        patch.object(vispy.scene, "Text", MagicMock),
        patch.object(vispy.scene, "AxisWidget", MagicMock),
    ):
        scene = AudioWaveformScene(canvas)
        scene.set_render_channels([_sample_render_channel()])
        assert scene.time_at_canvas_pos((0.0, 10.0)) == 0.0
        assert scene.time_at_canvas_pos((500.0, 10.0)) == pytest.approx(
            scene.duration * 0.5
        )
        assert scene.time_at_canvas_pos((1000.0, 10.0)) == pytest.approx(scene.duration)
        assert scene.time_at_canvas_pos((-50.0, 10.0)) == 0.0
        assert scene.time_at_canvas_pos((2000.0, 10.0)) == pytest.approx(scene.duration)
        view.pos = (80.0, 0.0)
        view.size = (920.0, 400.0)
        assert scene.time_at_canvas_pos((80.0, 10.0)) == 0.0
        assert scene.time_at_canvas_pos((80.0 + 460.0, 10.0)) == pytest.approx(
            scene.duration * 0.5
        )


def test_audio_spectrum_scene_set_spectrum() -> None:
    canvas = MagicMock()
    grid = MagicMock()
    canvas.central_widget.add_grid.return_value = grid
    view = MagicMock()
    view.scene = MagicMock()
    grid.add_view.return_value = view
    spec_line = MagicMock()
    peak_line = MagicMock()
    axis_widget = MagicMock()

    import vispy.scene

    with (
        patch.object(vispy.scene, "PanZoomCamera", MagicMock),
        patch.object(vispy.scene.cameras, "PanZoomCamera", MagicMock),
        patch.object(
            vispy.scene.visuals, "Line",
            MagicMock(side_effect=[spec_line, peak_line]),
        ),
        patch.object(vispy.scene, "AxisWidget", MagicMock(return_value=axis_widget)),
    ):
        scene = AudioSpectrumScene(canvas)
        assert spec_line.visible is False
        assert peak_line.visible is False
        freqs = np.array([0.0, 100.0, 200.0])
        db = np.array([-80.0, 0.0, -20.0])
        scene.set_spectrum(
            freqs, db, color="#1f77b4", peaks=np.array([100.0]),
        )
        spec_line.set_data.assert_called()
        assert spec_line.visible is True
        assert peak_line.visible is True
        assert axis_widget.link_view.call_count == 2
        peak_pos = peak_line.set_data.call_args.kwargs.get("pos")
        if peak_pos is None:
            peak_pos = peak_line.set_data.call_args[1].get("pos")
        assert peak_pos is not None
        assert peak_pos.shape == (2, 2)
        assert float(peak_pos[0, 0]) == pytest.approx(100.0)
        scene.set_spectrum(np.zeros(0), np.zeros(0), peaks=np.zeros(0))
        assert spec_line.visible is False
        assert peak_line.visible is False
        scene.set_spectrum(
            np.array([0.0, 100.0, 200.0]),
            np.array([-80.0, 0.0]),
            peaks=None,
        )
        assert spec_line.visible is True
        assert peak_line.visible is False
