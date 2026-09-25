"""Shared visPy 2D plot primitives for analysis views.

View-specific scenes (waveforms, FFT lines, 3D trajectories) live next to
their data modules. This file is the reusable layer those scenes should call:

* :func:`make_scene_canvas` — Qt-backed canvas (used by :class:`VispyViewWindow`)
* :func:`add_locked_xy_plot` — pan/zoom-locked view with optional axes
* :func:`set_data_range` — store and apply the camera window (no 5% pad)
* :func:`hex_to_rgba`, :func:`add_line`, :func:`add_fill_mesh`,
  :func:`vertical_segments`, :func:`map_canvas_x_to_data`

2D plots never use the depth buffer. vispy's ``translucent`` preset turns
depth testing on; agg lines then lose to the fill mesh or vanish after a
framebuffer resize. Layer with :data:`ORDER_FILL` / :data:`ORDER_DATA` /
:data:`ORDER_OVERLAY` only. :func:`add_line` uses agg tessellation; axis ticks
are a triangle mesh — GL lines vanish after some framebuffer resizes.

:class:`~vispy.scene.cameras.PanZoomCamera` does not recover from a 0-size
layout pass. :func:`set_data_range` stores the window for the view;
:func:`relock_2d_view` re-applies it once the view has a real size (hooked from
both view and canvas resize).
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import numpy as np

BG = "#1a1a1a"
FG = "#c9d1d9"
AXIS_RGBA = (0.45, 0.48, 0.52, 1.0)
ACCENT_RGBA = (0.0, 212 / 255, 170 / 255, 0.9)

_GUTTER_PX = (8, 12)
# vispy SceneCanvas sorts siblings by Node.order and draws higher later.
ORDER_FILL = 0
ORDER_DATA = 1
ORDER_OVERLAY = 2
# vispy ViewBox is frozen; stored ranges live here so resize can re-apply them.
_LOCKED_RANGES: WeakKeyDictionary = WeakKeyDictionary()


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


def ensure_gl_plus() -> bool:
    """Select vispy ``gl+`` (PyOpenGL) before any canvas exists.

    Instanced draws need this backend. Returns True when
    ``glDrawArraysInstanced`` is available afterwards.
    """
    try:
        from vispy import use

        use(gl="gl+")
        from vispy.gloo import gl

        return hasattr(gl, "glDrawArraysInstanced")
    except Exception:
        return False


def make_scene_canvas(
    *,
    keys=None,
    bgcolor: str = BG,
    size: tuple[int, int] = (1200, 800),
    show: bool = False,
    gl: str | None = None,
):
    """SceneCanvas on the PySide6 vispy app (safe to call more than once).

    *gl* is forwarded to ``vispy.use`` before the canvas is created. Instanced
    draws need ``gl="gl+"``; leave unset for the default GL2 backend.
    """
    if gl is not None:
        from vispy import use

        use(gl=gl)
    from vispy import scene
    from vispy.app import use_app

    use_app("pyside6")
    canvas = scene.SceneCanvas(
        keys=keys,
        bgcolor=bgcolor,
        size=size,
        show=show,
    )
    resize = getattr(getattr(canvas, "events", None), "resize", None)
    if resize is not None and hasattr(resize, "connect"):
        resize.connect(lambda _event=None: relock_canvas_2d_views(canvas))
    return canvas


def block_canvas_navigation(canvas) -> None:
    """Keep mouse-wheel from zooming a locked 2D plot."""

    def _block(event) -> None:
        event.handled = True

    canvas.events.mouse_wheel.connect(_block)


# vispy TurntableCamera: azimuth=0, elevation=0 looks along +Y.
# Blender (Z-up): 1 front (−Y), 3 right (−X), 7 top (−Z), Ctrl for the opposite.
_BLENDER_NUMPAD_ALIASES = {
    "End": "1",
    "PageDown": "3",
    "Home": "7",
    "PageUp": "9",
}
_BLENDER_TURNTABLE_SNAPS = {
    ("1", False): (180.0, 0.0),
    ("1", True): (0.0, 0.0),
    ("3", False): (90.0, 0.0),
    ("3", True): (-90.0, 0.0),
    ("7", False): (0.0, 90.0),
    ("7", True): (0.0, -90.0),
}


def blender_numpad_action(key: str, *, ctrl: bool = False) -> str:
    """Map a vispy key name to ``snap:az:el``, ``opposite``, ``ortho``, or ``""``."""
    digit = _BLENDER_NUMPAD_ALIASES.get(key, key)
    if digit == "5":
        return "ortho"
    if digit == "9":
        return "opposite"
    pose = _BLENDER_TURNTABLE_SNAPS.get((digit, bool(ctrl)))
    if pose is None:
        return ""
    return f"snap:{pose[0]:g}:{pose[1]:g}"


def apply_blender_view_action(camera, action: str) -> bool:
    """Apply :func:`blender_numpad_action` to a vispy TurntableCamera."""
    if not action or camera is None:
        return False
    if action == "ortho":
        fov = float(getattr(camera, "fov", 45.0) or 0.0)
        if fov <= 0.0:
            camera.fov = float(getattr(camera, "_persp_fov", 45.0) or 45.0)
        else:
            camera._persp_fov = fov
            camera.fov = 0.0
        return True
    if action == "opposite":
        elev = float(getattr(camera, "elevation", 0.0))
        if abs(elev) >= 80.0:
            camera.elevation = -elev
        else:
            camera.azimuth = (float(getattr(camera, "azimuth", 0.0)) + 180.0) % 360.0
        camera.roll = 0.0
        return True
    if action.startswith("snap:"):
        _, az, el = action.split(":")
        camera.azimuth = float(az)
        camera.elevation = float(el)
        camera.roll = 0.0
        return True
    return False


def bind_blender_view_keys(canvas, get_camera) -> None:
    """Numpad 1/3/7/9/5 (+ Ctrl) snap a 3D turntable like Blender.

    Click the canvas first so it has focus. Number-row keys work too
    (Blender's emulate-numpad), and End/PgDn/Home/PgUp cover NumLock off.
    """
    native = getattr(canvas, "native", None)
    if native is not None:
        from PySide6.QtCore import Qt

        native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _on_key(event) -> None:
        key = getattr(event, "key", None)
        if key is None:
            return
        name = getattr(key, "name", str(key))
        mods = getattr(event, "modifiers", ()) or ()
        ctrl = any(getattr(m, "name", m) == "Control" for m in mods)
        if apply_blender_view_action(get_camera(), blender_numpad_action(name, ctrl=ctrl)):
            event.handled = True
            canvas.update()

    canvas.events.key_press.connect(_on_key)


def _view_pixel_size(view) -> tuple[float, float] | None:
    size = getattr(view, "size", None)
    if size is None:
        return None
    try:
        width, height = float(size[0]), float(size[1])
    except (TypeError, ValueError, IndexError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def relock_2d_view(view) -> bool:
    """Re-apply a stored 2D camera window once *view* has a real pixel size.

    A 0-size layout pass can bake NaNs into ``PanZoomCamera``; ``view_changed``
    does not recover. Skip until width/height are positive, then call
    ``set_range`` from the limits :func:`set_data_range` stored for the view.
    Returns True if the camera was remapped.
    """
    if _view_pixel_size(view) is None:
        return False
    camera = getattr(view, "camera", None)
    if camera is None:
        return False
    stored = _LOCKED_RANGES.get(view)
    if stored is not None and getattr(camera, "interactive", False) is not True:
        xlim, ylim, margin = stored
        camera.set_range(x=xlim, y=ylim, margin=margin)
    else:
        view_changed = getattr(camera, "view_changed", None)
        if not callable(view_changed):
            return False
        view_changed()
    updater = getattr(view, "_update_scene_clipper", None)
    if callable(updater):
        updater()
    return True


def relock_canvas_2d_views(canvas) -> None:
    """Re-apply stored 2D ranges after a canvas/framebuffer resize."""
    n = 0
    for view in list(_LOCKED_RANGES):
        if getattr(view, "canvas", None) is not canvas:
            continue
        if relock_2d_view(view):
            n += 1
    if n:
        update = getattr(canvas, "update", None)
        if callable(update):
            update()


def lock_panzoom(view, camera=None, *, interactive: bool = False, bgcolor: str = BG):
    from vispy import scene

    if camera is None:
        camera = scene.PanZoomCamera(aspect=None)
    view.bgcolor = bgcolor
    camera.interactive = interactive
    view.camera = camera
    # ViewBox layout can change without a canvas resize (stacked rows).
    # Canvas resize is hooked in :func:`make_scene_canvas`.
    resize = getattr(getattr(view, "events", None), "resize", None)
    if resize is not None and hasattr(resize, "connect"):
        resize.connect(lambda _event=None: relock_2d_view(view))
    return camera


def set_data_range(
    view,
    x: tuple[float, float],
    y: tuple[float, float],
    *,
    margin: float = 0.0,
) -> None:
    """Lock the camera to *x*/*y* (vispy defaults to a 5% pad).

    Limits are stored so :func:`relock_2d_view` can re-apply them after a
    0-size layout pass or framebuffer resize. vispy ViewBox is frozen, so
    the window is kept in a weak map keyed by the view, not as node attrs.
    """
    xlim = (float(x[0]), float(x[1]))
    ylim = (float(y[0]), float(y[1]))
    margin_v = float(margin)
    try:
        _LOCKED_RANGES[view] = (xlim, ylim, margin_v)
    except TypeError:
        pass
    camera = getattr(view, "camera", None)
    if camera is not None:
        camera.set_range(x=xlim, y=ylim, margin=margin_v)


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
        widget.stretch = (0.1, 1)
    else:
        widget.height_min = 36
        widget.height_max = 44
        widget.stretch = (1, 0.1)
    _use_agg_axis_lines(widget)
    return widget


def _use_agg_axis_lines(widget) -> None:
    """Replace GL axis strokes with geometry that survives a resize.

    The spine is a 2-point agg strip. Ticks are a triangle mesh — vispy's
    tick LineVisual uses ``connect='segments'``, which agg cannot draw.
    """
    from vispy.visuals.mesh import MeshVisual

    visual = getattr(widget, "axis", None)
    if visual is None:
        return
    spine = getattr(visual, "_line", None)
    if spine is not None and hasattr(spine, "method"):
        spine.method = "agg"
        if hasattr(spine, "set_gl_state"):
            spine.set_gl_state("translucent", depth_test=False, depth_mask=False)
    ticks = getattr(visual, "_ticks", None)
    if ticks is not None:
        ticks.visible = False
    orig = getattr(visual, "_update_subvisuals", None)
    if not callable(orig):
        return
    mesh = MeshVisual(color=AXIS_RGBA)
    mesh.set_gl_state("translucent", depth_test=False, depth_mask=False)
    adder = getattr(visual, "add_subvisual", None)
    if callable(adder):
        adder(mesh)

    def _update() -> None:
        orig()
        pos = getattr(ticks, "pos", None) if ticks is not None else None
        if pos is None or len(np.asarray(pos).reshape(-1, 2)) < 2:
            mesh.visible = False
            return
        width = float(getattr(visual, "tick_width", 1.5) or 1.5)
        verts, faces = line_segments_mesh(pos, width=width)
        if len(faces) == 0:
            mesh.visible = False
            return
        mesh.set_data(vertices=verts, faces=faces, color=AXIS_RGBA)
        mesh.visible = True

    visual._update_subvisuals = _update


def _empty_grid_cell(
    *,
    width_min: int | None = None,
    width_max: int | None = None,
    stretch: tuple[float, float] = (0.1, 0.1),
):
    """Blank grid cell. vispy Widget does not take width_min in its constructor.

    Default stretch is small so a corner/gutter cannot inflate an axis row
    (vispy Grid assigns stretch (1, 1) when it is left as None).
    """
    from vispy.scene.widgets import Widget

    widget = Widget()
    widget.stretch = stretch
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
    view.stretch = (1, 1)
    lock_panzoom(view, scene.PanZoomCamera(aspect=None), interactive=interactive)
    set_data_range(view, x_range, y_range)
    if yaxis is not None:
        yaxis.link_view(view)
        _pin_axis_domain(yaxis, y_range)

    xaxis = None
    if x_axis:
        axis_row = row + 1
        if y_axis:
            grid.add_widget(_empty_grid_cell(), row=axis_row, col=col)
        xaxis = axis_widget("bottom")
        grid.add_widget(xaxis, row=axis_row, col=view_col)
        xaxis.link_view(view)
        _pin_axis_domain(xaxis, x_range)

    if right_gutter:
        gutter_col = view_col + 1
        lo, hi = _GUTTER_PX
        grid.add_widget(
            _empty_grid_cell(width_min=lo, width_max=hi, stretch=(0.1, 1)),
            row=row,
            col=gutter_col,
        )
        if x_axis:
            grid.add_widget(
                _empty_grid_cell(width_min=lo, width_max=hi, stretch=(0.1, 0.1)),
                row=row + 1,
                col=gutter_col,
            )

    return LockedXYPlot(
        view=view, yaxis=yaxis, xaxis=xaxis, view_col=view_col, row=row,
    )


def add_shared_x_axis(
    grid,
    *,
    row: int,
    view,
    col: int = 0,
    view_col: int = 1,
    x_range: tuple[float, float] | None = None,
):
    """X axis under a stack of :func:`add_locked_xy_plot` rows."""
    grid.add_widget(_empty_grid_cell(), row=row, col=col)
    xaxis = axis_widget("bottom")
    grid.add_widget(xaxis, row=row, col=view_col)
    xaxis.link_view(view)
    if x_range is not None:
        _pin_axis_domain(xaxis, x_range)
    return xaxis


def _pin_axis_domain(axis_w, domain: tuple[float, float]) -> None:
    """Keep locked-plot ticks on the data range.

    vispy AxisWidget infers domain by mapping its pixel ends into the view.
    Before layout (and if the widget does not stretch) that mapping is ~0 to
    widget-width in pixels, so an FFT axis can read as ~230 Hz instead of 500.
    """
    lo, hi = float(domain[0]), float(domain[1])

    def _apply(_event=None) -> None:
        axis_w.axis.domain = (lo, hi)

    view = getattr(axis_w, "_linked_view", None)
    if view is not None:
        view.scene.transform.changed.connect(_apply)
    axis_w.events.resize.connect(_apply)
    _apply()


def _set_plot_gl_state(visual) -> None:
    """2D plot GL state. Depth testing is off; stacking is Node.order."""
    visual.set_gl_state(
        "translucent",
        depth_test=False,
        depth_mask=False,
        cull_face=False,
        blend=True,
    )


def add_line(
    parent,
    pos=None,
    *,
    color=FG,
    width: float = 1.2,
    connect: str = "strip",
    antialias: bool = True,
    order: int = ORDER_DATA,
    visible: bool = True,
):
    from vispy import scene

    # method='agg' draws a triangle strip. GL lines (the vispy default) are
    # not reliably rasterized after a framebuffer resize — they just vanish.
    line = scene.visuals.Line(
        pos=np.zeros((2, 2), dtype=np.float32) if pos is None else pos,
        color=color,
        width=width,
        connect="strip",
        antialias=antialias,
        method="agg",
    )
    _set_plot_gl_state(line)
    line.order = order
    line.visible = visible
    line.parent = parent
    return line


def add_fill_mesh(
    parent, vertices, faces, color, *, order: int = ORDER_FILL,
) -> object:
    """Translucent mesh drawn behind data lines and overlays."""
    from vispy import scene

    mesh = scene.visuals.Mesh(
        vertices=vertices,
        faces=faces,
        color=color,
    )
    _set_plot_gl_state(mesh)
    mesh.order = order
    mesh.parent = parent
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


def line_segments_mesh(
    pos, width: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangle quads covering disconnected line segments (axis ticks)."""
    pts = np.asarray(pos, dtype=np.float32).reshape(-1, 2)
    n = pts.shape[0] // 2
    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
        )
    half = max(float(width), 1.0) * 0.5
    verts = np.zeros((n * 4, 3), dtype=np.float32)
    faces = np.empty((n * 2, 3), dtype=np.uint32)
    for i in range(n):
        a = pts[2 * i]
        b = pts[2 * i + 1]
        delta = b - a
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1e-6:
            nrm = np.array([1.0, 0.0], dtype=np.float32)
        else:
            nrm = np.array(
                [-delta[1] / length, delta[0] / length], dtype=np.float32,
            )
        offset = nrm * half
        base = i * 4
        verts[base, :2] = a + offset
        verts[base + 1, :2] = a - offset
        verts[base + 2, :2] = b - offset
        verts[base + 3, :2] = b + offset
        faces[i * 2] = (base, base + 1, base + 2)
        faces[i * 2 + 1] = (base, base + 2, base + 3)
    return verts, faces


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
