"""Shared visPy 2D plot primitives for analysis views.

View-specific scenes (waveforms, FFT lines, 3D trajectories) live next to
their data modules. This file is the reusable layer those scenes should call:

* :func:`make_scene_canvas` — Qt-backed canvas (used by :class:`VispyViewWindow`)
* :func:`add_locked_xy_plot` — pan/zoom-locked view with optional axes
* :func:`hex_to_rgba`, :func:`add_line`, :func:`add_fill_mesh`,
  :func:`vertical_segments`, :func:`map_canvas_x_to_data`

Lock cameras with :func:`set_data_range` (vispy's default 5% margin is wrong
for aligned axes). Overlay markers (cursors, peak ticks) should use
``order=1`` and not write depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BG = "#1a1a1a"
FG = "#c9d1d9"
AXIS_RGBA = (0.45, 0.48, 0.52, 1.0)
ACCENT_RGBA = (0.0, 212 / 255, 170 / 255, 0.9)

_GUTTER_PX = (8, 12)


def hex_to_rgba(
    hex_color: str,
    alpha: float = 1.0,
) -> tuple[float, float, float, float]:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return (0.7, 0.7, 0.7, alpha)
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    return (r, g, b, alpha)


def make_scene_canvas(
    *,
    keys=None,
    bgcolor: str = BG,
    size: tuple[int, int] = (1200, 800),
    show: bool = False,
):
    """SceneCanvas on the PySide6 vispy app (safe to call more than once)."""
    from vispy import scene
    from vispy.app import use_app

    use_app("pyside6")
    return scene.SceneCanvas(
        keys=keys,
        bgcolor=bgcolor,
        size=size,
        show=show,
    )


def block_canvas_navigation(canvas) -> None:
    """Keep mouse-wheel from zooming a locked 2D plot."""

    def _block(event) -> None:
        event.handled = True

    canvas.events.mouse_wheel.connect(_block)


def lock_panzoom(view, camera=None, *, interactive: bool = False, bgcolor: str = BG):
    from vispy import scene

    if camera is None:
        camera = scene.PanZoomCamera(aspect=None)
    view.bgcolor = bgcolor
    camera.interactive = interactive
    view.camera = camera
    return camera


def set_data_range(
    view,
    x: tuple[float, float],
    y: tuple[float, float],
    *,
    margin: float = 0.0,
) -> None:
    """Lock the camera to *x*/*y* (vispy defaults to a 5% pad)."""
    view.camera.set_range(x=x, y=y, margin=margin)


def axis_widget(orientation: str, *, font_size: float = 8):
    from vispy import scene

    widget = scene.AxisWidget(
        orientation=orientation,
        axis_color=AXIS_RGBA,
        tick_color=AXIS_RGBA,
        text_color=FG,
        font_size=font_size,
    )
    if orientation == "left":
        widget.width_min = 48
        widget.width_max = 56
    else:
        widget.height_min = 36
        widget.height_max = 44
    return widget


def _empty_grid_cell(
    *,
    width_min: int | None = None,
    width_max: int | None = None,
):
    """Blank grid cell. vispy Widget does not take width_min in its constructor."""
    from vispy.scene.widgets import Widget

    widget = Widget()
    if width_min is not None:
        widget.width_min = width_min
    if width_max is not None:
        widget.width_max = width_max
    return widget


@dataclass
class LockedXYPlot:
    """One locked 2D view, optionally with Y and/or X axis widgets."""

    view: object
    yaxis: object | None
    xaxis: object | None
    view_col: int
    row: int


def add_locked_xy_plot(
    grid,
    *,
    row: int = 0,
    col: int = 0,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    y_axis: bool = True,
    x_axis: bool = False,
    right_gutter: bool = False,
    interactive: bool = False,
) -> LockedXYPlot:
    """Add a pan/zoom-locked view to *grid*, with optional linked axes.

    Y axis occupies *col*; the view is *col+1* when a Y axis is present.
    An X axis, if requested, is placed on ``row + 1`` under the view.
    For stacked plots that share one X axis, pass ``x_axis=False`` and call
    :func:`add_shared_x_axis` under the last row.
    """
    from vispy import scene

    view_col = col + 1 if y_axis else col
    yaxis = None
    if y_axis:
        yaxis = axis_widget("left")
        grid.add_widget(yaxis, row=row, col=col)

    view = grid.add_view(row=row, col=view_col)
    lock_panzoom(view, scene.PanZoomCamera(aspect=None), interactive=interactive)
    set_data_range(view, x_range, y_range)
    if yaxis is not None:
        yaxis.link_view(view)

    xaxis = None
    if x_axis:
        axis_row = row + 1
        if y_axis:
            grid.add_widget(_empty_grid_cell(), row=axis_row, col=col)
        xaxis = axis_widget("bottom")
        grid.add_widget(xaxis, row=axis_row, col=view_col)
        xaxis.link_view(view)

    if right_gutter:
        gutter_col = view_col + 1
        lo, hi = _GUTTER_PX
        grid.add_widget(
            _empty_grid_cell(width_min=lo, width_max=hi),
            row=row,
            col=gutter_col,
        )
        if x_axis:
            grid.add_widget(
                _empty_grid_cell(width_min=lo, width_max=hi),
                row=row + 1,
                col=gutter_col,
            )

    return LockedXYPlot(
        view=view, yaxis=yaxis, xaxis=xaxis, view_col=view_col, row=row,
    )


def add_shared_x_axis(grid, *, row: int, view, col: int = 0, view_col: int = 1):
    """X axis under a stack of :func:`add_locked_xy_plot` rows."""
    grid.add_widget(_empty_grid_cell(), row=row, col=col)
    xaxis = axis_widget("bottom")
    grid.add_widget(xaxis, row=row, col=view_col)
    xaxis.link_view(view)
    return xaxis


def add_line(
    parent,
    pos=None,
    *,
    color=FG,
    width: float = 1.2,
    connect: str = "strip",
    antialias: bool = True,
    order: int = 0,
    visible: bool = True,
):
    from vispy import scene

    line = scene.visuals.Line(
        pos=np.zeros((2, 2), dtype=np.float32) if pos is None else pos,
        color=color,
        width=width,
        connect=connect,
        antialias=antialias,
        parent=parent,
    )
    line.order = order
    line.visible = visible
    return line


def add_fill_mesh(parent, vertices, faces, color) -> object:
    """Translucent mesh that does not occlude later overlays."""
    from vispy import scene

    mesh = scene.visuals.Mesh(
        vertices=vertices,
        faces=faces,
        color=color,
        parent=parent,
    )
    mesh.set_gl_state("translucent", depth_test=False, depth_mask=False)
    return mesh


def add_status_text(view, message: str, *, color: str = FG, font_size: float = 14):
    from vispy import scene

    return scene.Text(
        message,
        color=color,
        font_size=font_size,
        pos=(0, 0),
        anchor_x="center",
        anchor_y="center",
        parent=view.scene,
    )


def view_pixel_x_span(view) -> tuple[float, float | None]:
    try:
        x0 = float(view.pos[0])
        width = float(view.size[0])
    except (TypeError, ValueError, IndexError, AttributeError):
        return 0.0, None
    if width <= 0.0:
        return 0.0, None
    return x0, width


def map_canvas_x_to_data(
    view,
    canvas_x: float,
    x_min: float,
    x_max: float,
    *,
    canvas_width: float = 0.0,
) -> float | None:
    """Map a canvas pixel x to data x using *view*'s pixel rect."""
    if x_max <= x_min:
        return None
    x0, width = view_pixel_x_span(view)
    if width is None:
        width = float(canvas_width)
        x0 = 0.0
    if width <= 0.0:
        return None
    rel = (float(canvas_x) - x0) / width
    return max(x_min, min(x_min + (x_max - x_min) * rel, x_max))


def vertical_segments(
    xs: np.ndarray | float,
    y_lo: float,
    y_hi: float,
) -> np.ndarray:
    """Disconnected vertical lines (playhead, FFT peak ticks)."""
    hz = np.asarray(xs, dtype=np.float32).reshape(-1)
    if hz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pos = np.empty((hz.size * 2, 2), dtype=np.float32)
    pos[0::2, 0] = hz
    pos[0::2, 1] = y_lo
    pos[1::2, 0] = hz
    pos[1::2, 1] = y_hi
    return pos


def envelope_mesh_geometry(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangle strip covering a closed min/max envelope polygon."""
    if len(poly) < 4:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    n = len(poly) // 2
    t = poly[:n, 0]
    y_max = poly[:n, 1]
    y_min = poly[n:, 1][::-1]
    pos = np.zeros((n * 2, 3), dtype=np.float32)
    pos[0::2, 0] = t
    pos[0::2, 1] = y_min
    pos[1::2, 0] = t
    pos[1::2, 1] = y_max
    faces = np.empty((2 * max(n - 1, 0), 3), dtype=np.uint32)
    if n > 1:
        idx = np.arange(n - 1, dtype=np.uint32)
        faces[0::2, 0] = idx * 2
        faces[0::2, 1] = idx * 2 + 1
        faces[0::2, 2] = idx * 2 + 2
        faces[1::2, 0] = idx * 2 + 1
        faces[1::2, 1] = idx * 2 + 3
        faces[1::2, 2] = idx * 2 + 2
    return pos, faces
