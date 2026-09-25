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
    from scan_kit.views.vispy_plot import _LOCKED_RANGES

    view = MagicMock()
    set_data_range(view, (0.0, 500.0), (-80.0, 5.0))
    view.camera.set_range.assert_called_once_with(
        x=(0.0, 500.0), y=(-80.0, 5.0), margin=0.0,
    )
    assert _LOCKED_RANGES[view] == ((0.0, 500.0), (-80.0, 5.0), 0.0)


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
        depth_test=False,
        depth_mask=False,
        cull_face=False,
        blend=True,
    )
    assert line.order == 1
    assert line.parent is not None


def test_add_fill_mesh_uses_same_2d_gl_state() -> None:
    import vispy.scene

    mesh = MagicMock()
    verts = np.zeros((4, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    with patch.object(vispy.scene.visuals, "Mesh", MagicMock(return_value=mesh)):
        from scan_kit.views.vispy_plot import add_fill_mesh

        add_fill_mesh(MagicMock(), verts, faces, (1.0, 0.0, 0.0, 1.0))
    mesh.set_gl_state.assert_called_once_with(
        "translucent",
        depth_test=False,
        depth_mask=False,
        cull_face=False,
        blend=True,
    )
    assert mesh.order == 0


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


def test_lock_panzoom_reapplies_stored_range_after_zero_size() -> None:
    from scan_kit.views.vispy_plot import lock_panzoom, set_data_range

    view = MagicMock()
    view.size = (0, 200)
    camera = MagicMock()
    lock_panzoom(view, camera)
    set_data_range(view, (0.0, 10.0), (-1.0, 1.0))
    relock = view.events.resize.connect.call_args[0][0]
    camera.set_range.reset_mock()
    relock()
    camera.set_range.assert_not_called()
    view.size = (400, 120)
    relock()
    camera.set_range.assert_called_with(x=(0.0, 10.0), y=(-1.0, 1.0), margin=0.0)


def test_canvas_resize_relocks_locked_views() -> None:
    from scan_kit.views.vispy_plot import relock_canvas_2d_views, set_data_range

    class _Node:
        def __init__(self) -> None:
            self.children: list = []

    canvas = _Node()
    camera = MagicMock()
    camera.interactive = False
    view = _Node()
    view.size = (320, 160)
    view.camera = camera
    view.canvas = canvas
    view._update_scene_clipper = MagicMock()
    set_data_range(view, (0.0, 70.0), (-0.2, 1.1))
    camera.set_range.reset_mock()

    canvas.central_widget = _Node()
    canvas.update = MagicMock()
    relock_canvas_2d_views(canvas)
    camera.set_range.assert_called_with(x=(0.0, 70.0), y=(-0.2, 1.1), margin=0.0)
    canvas.update.assert_called()


def test_make_scene_canvas_hooks_canvas_resize() -> None:
    import vispy.scene

    canvas = MagicMock()
    with (
        patch.object(vispy.scene, "SceneCanvas", MagicMock(return_value=canvas)),
        patch("vispy.app.use_app"),
    ):
        from scan_kit.views.vispy_plot import make_scene_canvas

        result = make_scene_canvas(size=(100, 80))
    assert result is canvas
    canvas.events.resize.connect.assert_called()
    handler = canvas.events.resize.connect.call_args[0][0]
    assert callable(handler)


def test_make_scene_canvas_requests_gl_plus() -> None:
    import vispy.scene

    canvas = MagicMock()
    with (
        patch.object(vispy.scene, "SceneCanvas", MagicMock(return_value=canvas)),
        patch("vispy.app.use_app"),
        patch("vispy.use") as use_gl,
    ):
        from scan_kit.views.vispy_plot import make_scene_canvas

        make_scene_canvas(size=(80, 60), gl="gl+")
    use_gl.assert_called_once_with(gl="gl+")


def test_blender_numpad_snaps_turntable_like_blender() -> None:
    from scan_kit.views.vispy_plot import (
        apply_blender_view_action,
        blender_numpad_action,
    )

    assert blender_numpad_action("1") == "snap:180:0"
    assert blender_numpad_action("1", ctrl=True) == "snap:0:0"
    assert blender_numpad_action("3") == "snap:90:0"
    assert blender_numpad_action("3", ctrl=True) == "snap:-90:0"
    assert blender_numpad_action("7") == "snap:0:90"
    assert blender_numpad_action("Home") == "snap:0:90"
    assert blender_numpad_action("7", ctrl=True) == "snap:0:-90"
    assert blender_numpad_action("9") == "opposite"
    assert blender_numpad_action("5") == "ortho"
    assert blender_numpad_action("2") == ""

    cam = MagicMock(azimuth=30.0, elevation=25.0, fov=45.0, roll=10.0)
    assert apply_blender_view_action(cam, "snap:180:0")
    assert cam.azimuth == 180.0
    assert cam.elevation == 0.0
    assert cam.roll == 0.0
    cam.elevation = 90.0
    assert apply_blender_view_action(cam, "opposite")
    assert cam.elevation == -90.0
    cam.elevation = 20.0
    cam.azimuth = 10.0
    assert apply_blender_view_action(cam, "opposite")
    assert cam.azimuth == 190.0
    assert apply_blender_view_action(cam, "ortho")
    assert cam.fov == 0.0
    assert apply_blender_view_action(cam, "ortho")
    assert cam.fov == 45.0


def test_ensure_gl_plus_returns_false_when_backend_missing() -> None:
    with patch("vispy.use", side_effect=RuntimeError("no OpenGL")):
        from scan_kit.views.vispy_plot import ensure_gl_plus

        assert ensure_gl_plus() is False


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


def _render_rgba_or_skip(canvas) -> np.ndarray:
    try:
        img = np.asarray(canvas.render())
    except Exception as exc:
        pytest.skip(f"visPy cannot render offscreen: {exc}")
    if img.ndim != 3 or img.size == 0:
        pytest.skip("visPy render returned an empty image")
    return img


def _trace_pixel_count(img: np.ndarray) -> int:
    # Fill is opaque red (G=0). The agg trace is white; AA edges still have green.
    green = np.asarray(img)[..., 1]
    return int(np.count_nonzero(green > 80))


def test_2d_trace_stays_on_top_after_resize(qapp) -> None:
    """Pin the Audio Explorer failure: traces vanish (or sit behind fill) on resize."""
    from scan_kit.views.vispy_plot import (
        add_fill_mesh,
        add_line,
        add_locked_xy_plot,
        make_scene_canvas,
        relock_canvas_2d_views,
    )

    canvas = make_scene_canvas(size=(240, 160), show=False)
    try:
        grid = canvas.central_widget.add_grid(spacing=0)
        plot = add_locked_xy_plot(
            grid,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            y_axis=False,
            x_axis=False,
        )
        verts = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        add_fill_mesh(plot.view.scene, verts, faces, (1.0, 0.0, 0.0, 1.0))
        add_line(
            plot.view.scene,
            np.array([[0.05, 0.5], [0.95, 0.5]], dtype=np.float32),
            color=(1.0, 1.0, 1.0, 1.0),
            width=10.0,
        )
        native = canvas.native
        native.resize(240, 160)
        qapp.processEvents()
        relock_canvas_2d_views(canvas)
        before = _render_rgba_or_skip(canvas)
        assert _trace_pixel_count(before) > 0

        native.resize(2, 2)
        qapp.processEvents()
        native.resize(480, 280)
        qapp.processEvents()
        relock_canvas_2d_views(canvas)
        after = _render_rgba_or_skip(canvas)
        assert _trace_pixel_count(after) > 0
    finally:
        close = getattr(canvas, "close", None)
        if callable(close):
            close()

