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


def test_add_line_uses_agg_without_depth() -> None:
    import vispy.scene

    line = MagicMock()
    with patch.object(
        vispy.scene.visuals, "Line", MagicMock(return_value=line),
    ) as ctor:
        from scan_kit.views.vispy_plot import add_line

        add_line(MagicMock(), width=2.0, order=1)
    assert ctor.call_args.kwargs.get("parent") is None
    assert ctor.call_args.kwargs["method"] == "agg"
    line.set_gl_state.assert_called_once_with(
        "translucent",
        depth_test=True,
        depth_mask=True,
        cull_face=False,
        polygon_offset_fill=True,
        polygon_offset=(1.0, 1.0),
    )
    assert line.order == 1
    assert line.parent is not None


def test_lock_panzoom_relocks_camera_on_resize() -> None:
    from scan_kit.views.vispy_plot import lock_panzoom

    view = MagicMock()
    view.size = (800, 200)
    camera = MagicMock()
    lock_panzoom(view, camera)
    view.events.resize.connect.assert_called()
    relock = view.events.resize.connect.call_args[0][0]
    relock()
    camera.view_changed.assert_called()
    view._update_scene_clipper.assert_called()
    camera.view_changed.reset_mock()
    view.size = (0, 200)
    relock()
    camera.view_changed.assert_not_called()


def test_line_segments_mesh_makes_quads() -> None:
    from scan_kit.views.vispy_plot import line_segments_mesh

    pos = np.array([[10.0, 0.0], [10.0, 8.0]], dtype=np.float32)
    verts, faces = line_segments_mesh(pos, width=2.0)
    assert verts.shape == (4, 3)
    assert faces.shape == (2, 3)
    assert faces.max() < len(verts)
    # Vertical tick: offset is along X.
    xs = verts[:, 0]
    assert xs.max() > 10.0
    assert xs.min() < 10.0


def test_empty_grid_cell_does_not_steal_axis_row() -> None:
    from scan_kit.views.vispy_plot import _empty_grid_cell

    spacer = _empty_grid_cell()
    assert tuple(spacer.stretch) == (0.1, 0.1)
    gutter = _empty_grid_cell(width_min=8, width_max=12, stretch=(0.1, 1))
    assert tuple(gutter.stretch) == (0.1, 1)
    assert gutter.width_min == 8
    from scan_kit.views.vispy_plot import axis_widget

    import vispy.scene

    widget = MagicMock()
    with patch.object(vispy.scene, "AxisWidget", MagicMock(return_value=widget)):
        axis_widget("bottom")
        assert widget.stretch == (1, 0.1)
        axis_widget("left")
        assert widget.stretch == (0.1, 1)


def test_pin_axis_domain_keeps_fft_range() -> None:
    from scan_kit.views.vispy_plot import _pin_axis_domain

    axis_w = MagicMock()
    axis_w._linked_view = None
    _pin_axis_domain(axis_w, (0.0, 500.0))
    assert axis_w.axis.domain == (0.0, 500.0)
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
