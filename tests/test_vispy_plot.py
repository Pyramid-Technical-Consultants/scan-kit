"""Tests for shared visPy 2D plot primitives and the visPy view shell."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scan_kit.views.vispy_plot import (
    envelope_mesh_geometry,
    hex_to_rgba,
    map_canvas_x_to_data,
    set_data_range,
    vertical_segments,
)


def test_hex_to_rgba_parses_hex_and_alpha() -> None:
    assert hex_to_rgba("#ff0000", 0.5) == (1.0, 0.0, 0.0, 0.5)
    assert hex_to_rgba("bad") == (0.7, 0.7, 0.7, 1.0)


def test_set_data_range_uses_zero_margin() -> None:
    view = MagicMock()
    set_data_range(view, (0.0, 500.0), (-80.0, 5.0))
    view.camera.set_range.assert_called_once_with(
        x=(0.0, 500.0), y=(-80.0, 5.0), margin=0.0,
    )


def test_vertical_segments_pairs_floor_and_top() -> None:
    pos = vertical_segments([10.0, 20.0], -80.0, 5.0)
    assert pos.shape == (4, 2)
    assert pos[0].tolist() == [10.0, -80.0]
    assert pos[1].tolist() == [10.0, 5.0]
    assert pos[2].tolist() == [20.0, -80.0]
    assert vertical_segments([], -80.0, 5.0).shape == (0, 2)


def test_map_canvas_x_to_data_uses_view_rect() -> None:
    view = MagicMock()
    view.pos = (80.0, 0.0)
    view.size = (920.0, 400.0)
    assert map_canvas_x_to_data(view, 80.0, 0.0, 10.0) == 0.0
    assert map_canvas_x_to_data(view, 80.0 + 460.0, 0.0, 10.0) == pytest.approx(5.0)
    assert map_canvas_x_to_data(view, 2000.0, 0.0, 10.0) == 10.0


def test_envelope_mesh_geometry_covers_bins() -> None:
    poly = np.array(
        [[0.0, 1.0], [1.0, 0.5], [1.0, -0.5], [0.0, -1.0]],
        dtype=np.float32,
    )
    pos, faces = envelope_mesh_geometry(poly)
    assert pos.shape == (4, 3)
    assert faces.shape == (2, 3)
    assert faces.max() < len(pos)


def test_add_locked_xy_plot_gutter_does_not_pass_width_kwargs() -> None:
    from scan_kit.views.vispy_plot import add_locked_xy_plot

    grid = MagicMock()
    view = MagicMock()
    view.scene = MagicMock()
    grid.add_view.return_value = view
    spacer = MagicMock()
    axis = MagicMock()

    import vispy.scene
    import vispy.scene.widgets

    with (
        patch.object(vispy.scene, "PanZoomCamera", MagicMock),
        patch.object(vispy.scene, "AxisWidget", MagicMock(return_value=axis)),
        patch.object(
            vispy.scene.widgets, "Widget", MagicMock(return_value=spacer),
        ),
    ):
        add_locked_xy_plot(
            grid,
            x_range=(0.0, 500.0),
            y_range=(-80.0, 5.0),
            x_axis=True,
            right_gutter=True,
        )

    for call in grid.add_widget.call_args_list:
        assert "width_min" not in call.kwargs
        assert "width_max" not in call.kwargs
    assert spacer.width_min == 8
    assert spacer.width_max == 12


def test_vispy_view_window_side_panel_and_footer(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    from scan_kit.views.plot_view_shell import (
        VispyViewWindow,
        make_side_panel_column,
    )

    window = VispyViewWindow(title="VisPy Shell")
    panel, layout = make_side_panel_column()
    layout.addWidget(QLabel("Controls"))
    window.set_side_panel(panel)
    sizes = window._splitter.sizes()
    assert len(sizes) == 2
    assert sizes[1] >= window._side_min_width
    footer = QLabel("Transport")
    window.set_footer(footer)
    assert window._footer_host is not None
    assert not window._footer_host.isHidden()
    window.close()
