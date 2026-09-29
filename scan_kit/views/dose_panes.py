"""Dose panes to mix and match: slice planes, line plots, and the 3D volume with its color axis.

Each pane is its own canvas, so windows and tabs lay them out with Qt, and a cursor move
repaints only the panes it changes. :class:`DoseWorkspace` is the standard arrangement:
three planes and the 3D volume in a 2×2 grid, over two plots side by side.

A source describes its grid once (:class:`DoseFrame`) and then what to draw on it
(:class:`DoseLayers`). Visuals are made once, and later calls only feed them data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from .color_axis import ColorAxis
from .dose_volume_catalog import DEFAULT_SCALE
from .dose_volume_fill import colormap_samples, ink_rgb, zero_rgb
from .dose_volume_vispy import BOX_ALPHA, NUMBER_ALPHA
from .view_header import ViewHeader, tool_button
from .vispy_plot import (
    FG,
    ORDER_DATA,
    ORDER_FILL,
    ORDER_OVERLAY,
    add_line,
    axis_widget,
    lock_panzoom,
    make_scene_canvas,
    set_data_range,
)

PLANES = ("Axial", "Coronal", "Sagittal")
PLANE_AXES = ((0, 1), (0, 2), (1, 2))  # grid axes (x, y, z) across and up each plane
PLANE_NORMAL = (2, 1, 0)
VIEW_3D = 3
VIEWS = (*PLANES, "3D")
DEFAULT_CELLS = (0, VIEW_3D, 1, 2)  # axial, 3D / coronal, sagittal
OVERLAY_ALPHA = 0.55
WASH_FLOOR = 0.1  # dose below this share of the maximum is not washed
CT_WINDOW = (-500.0, 500.0)  # HU shown black to white
DVH, GAMMA_HIST, DEPTH = "dvh", "gamma", "depth"
LATERAL, LONGITUDINAL, LAT_LONG = "lateral", "longitudinal", "lat_long"
PLOT_KINDS = ((DVH, "DVH"), (GAMMA_HIST, "Gamma histogram"), (DEPTH, "Depth dose"), (LATERAL, "Lateral profile"),
              (LONGITUDINAL, "Longitudinal profile"), (LAT_LONG, "Lateral + longitudinal"))
PROFILES = (DEPTH, LATERAL, LONGITUDINAL, LAT_LONG)
_PROFILE_X = {
    DEPTH: "Depth from the grid edge (mm)",
    LATERAL: "Across the beam from the crosshair (mm)",
    LONGITUDINAL: "Along the beam from the crosshair (mm)",
    LAT_LONG: "From the crosshair (mm)",
}
CURVE_COLORS = ("#4ea1ff", "#f5a524", "#46a758", "#e5484d")
_PLOT_SPAN = 100.0  # plots draw in a 0-100 box; agg lines misplace points when one axis spans ~1e-2 and the other 1e2


@dataclass
class DoseFrame:
    """A grid: voxel corner (0, 0, 0) at *origin*, *spacing* and *shape* along x, y, z, and what lies under the dose."""

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]
    names: tuple[str, str, str] = ("X (mm)", "Y (mm)", "Z (mm)")
    beam_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    ct: np.ndarray | None = None  # HU, (z, y, x)
    rois: Sequence[tuple[str, tuple[int, int, int]]] = ()  # (name, RGB 0-255)
    bits: Sequence[np.ndarray] = ()  # ROI masks packed by :func:`scan_kit.dicom.structures.mask_bits`
    flip_axial: bool = False  # axial row 0 at the top, as a CT is read

    @property
    def extent(self) -> np.ndarray:
        return np.asarray(self.spacing, dtype=float) * self.shape


@dataclass
class DoseLayers:
    """What the panes draw on a frame."""

    wash: np.ndarray | None = None  # volume washed over the slices, (z, y, x)
    scale: str = DEFAULT_SCALE
    lo: float = 0.0
    hi: float = 1.0
    gamma: bool = False  # the wash is γ: only γ > 0 shows
    unit: str = "Gy"
    profiles: dict[str, np.ndarray] = field(default_factory=dict)  # label -> dose volume, for depth and lateral
    dvh: dict[str, dict] = field(default_factory=dict)  # label -> ROI name -> DVH
    main: str = ""  # label drawn bright; the others fade
    gamma_values: np.ndarray | None = None  # γ of the evaluated voxels
    gamma_note: str = ""


def plane_cut(vol: np.ndarray, plane: int, index: int) -> np.ndarray:
    """One plane of a ``(z, y, x)`` volume: rows run up the plane, columns across it."""
    return vol[index] if plane == 0 else vol[:, index, :] if plane == 1 else vol[:, :, index]


def wash_rgba(img: np.ndarray, lut: np.ndarray, lo: float, hi: float, gamma: bool, alpha: float) -> np.ndarray:
    """RGBA uint8: the scale's color, transparent where the dose is faint."""
    t = np.clip((img - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(img)
    rgba = np.empty((*img.shape, 4), np.uint8)
    rgba[..., :3] = lut[(t * (len(lut) - 1)).astype(np.intp)]
    faint = img <= 0 if gamma else np.abs(img) < WASH_FLOOR * max(abs(lo), abs(hi))
    rgba[..., 3] = np.where(faint, 0, round(255 * alpha))
    return rgba


def outline_segments(mask: np.ndarray, pixel) -> np.ndarray:
    """A 2D mask's outline as line segments ``(2n, 2)`` in mm, along voxel edges."""
    from contourpy import contour_generator

    segs = [np.zeros((0, 2), np.float32)]
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size:
        # Contour a one-voxel-padded crop: outlines close, and cost follows the ROI, not the slice.
        r0, c0 = rows[0] - 1, cols[0] - 1
        crop = np.zeros((rows[-1] - r0 + 2, cols[-1] - c0 + 2), np.float32)
        crop[1:-1, 1:-1] = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        for line in contour_generator(z=crop).lines(0.5):
            pts = (line + (c0 + 0.5, r0 + 0.5)) * pixel
            segs.append(np.repeat(pts, 2, axis=0)[1:-1])
    return np.concatenate(segs).astype(np.float32)


def line_profile(vol: np.ndarray, frame: DoseFrame, cursor, direction) -> tuple[np.ndarray, np.ndarray]:
    """Dose along *direction* through the cursor voxel's center: (signed distance mm, dose), inside the grid."""
    from scipy.ndimage import map_coordinates

    sp = np.asarray(frame.spacing, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    reach = float(np.linalg.norm(frame.extent))
    t = np.arange(-reach, reach, 0.5 * float(sp.min()))
    idx = (np.asarray(cursor, dtype=float) + 0.5) + t[:, None] * d / sp - 0.5  # continuous x, y, z index
    inside = np.all((idx >= 0) & (idx <= np.asarray(frame.shape) - 1), axis=1)
    t, idx = t[inside], idx[inside]
    return t, map_coordinates(vol, idx[:, ::-1].T, order=1)


def integral_profile(vol: np.ndarray, frame: DoseFrame, direction) -> tuple[np.ndarray, np.ndarray]:
    """Dose summed over each plane across *direction*: (mm along it from the grid corner, dose·mm²).

    Voxel centers fall in bins one voxel deep along *direction*, which is exact along a grid axis.
    """
    sp = np.asarray(frame.spacing, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    step = float(np.abs(d) @ sp)
    # ponytail: an oblique direction bins voxel centers, so the curve ripples at the voxel pitch;
    # resample the volume along it if that shows.
    nz, ny, nx = vol.shape
    t = ((np.arange(nx) + 0.5) * sp[0] * d[0])[None, None, :] + ((np.arange(ny) + 0.5) * sp[1] * d[1])[None, :, None] \
        + ((np.arange(nz) + 0.5) * sp[2] * d[2])[:, None, None]
    k = np.floor(t / step).astype(np.int64).ravel()
    k0 = int(k.min())
    sums = np.bincount(k - k0, weights=np.asarray(vol, dtype=float).ravel())
    return (k0 + np.arange(sums.size) + 0.5) * step, sums * float(np.prod(sp)) / step


def lateral_dir(beam_dir) -> np.ndarray:
    """Across the beam, along the grid axis least aligned with it."""
    b = np.asarray(beam_dir, dtype=float)
    b = b / (np.linalg.norm(b) or 1.0)
    e = np.eye(3)[int(np.argmin(np.abs(b)))]
    across = e - np.dot(e, b) * b
    return across / np.linalg.norm(across)


def _splitter(orientation, *widgets) -> QSplitter:
    split = QSplitter(orientation)
    split.setChildrenCollapsible(False)
    split.setHandleWidth(6)
    for w in widgets:
        split.addWidget(w)
    # Even, not by size hint (the 3D pane asks for more than a slice). Weights below a pane's
    # minimum width get clamped to it, which skews the split, so they are large.
    split.setSizes([10_000] * len(widgets))
    return split


def _gl_2d(visual, order: int) -> None:
    visual.set_gl_state("translucent", depth_test=False, depth_mask=False, cull_face=False)
    visual.order = order


def _rgba(color, alpha: float = 1.0) -> tuple[float, float, float, float]:
    from vispy.color import Color

    return (*Color(color).rgb, alpha)


def _labelled_axis(grid, view, orientation: str, *, row: int, col: int):
    """A 1 px axis linked to *view*, with room for its title; set the title with ``axis.axis.axis_label``."""
    left = orientation == "left"
    axis = axis_widget(orientation, axis_label=" ", axis_font_size=8, axis_label_margin=40 if left else 28)
    if left:
        axis.width_min = axis.width_max = 62
    else:
        axis.height_min = axis.height_max = 46
    grid.add_widget(axis, row=row, col=col)
    axis.link_view(view)
    return axis


def _ink_axis(axis, fg, label: str | None = None) -> None:
    """The 3D box's muted ink on the spine and ticks, a little stronger for the text."""
    rgb = _rgba(fg)[:3]
    axis.axis.axis_color = axis.axis.tick_color = (*rgb, BOX_ALPHA)
    axis.axis.text_color = (*rgb, NUMBER_ALPHA)
    if label is not None and axis.axis.axis_label != label:
        axis.axis.axis_label = label
    axis.axis._update_subvisuals()


class SlicePane(QWidget):
    """One plane: CT, dose wash, outlines, crosshair, a box with mm axes, and its own color bar.

    :attr:`picked` (mm across and up the plane) on click or drag, :attr:`paged` (steps) on scroll,
    :attr:`changed` when a tool needs the pane redrawn. :attr:`tools` go in the view's header.
    """

    picked = Signal(float, float)
    paged = Signal(int)
    changed = Signal()

    def __init__(self, plane: int, parent: QWidget | None = None) -> None:
        from vispy import scene

        super().__init__(parent)
        self.plane = plane
        self._frame: DoseFrame | None = None
        self._theme = None
        self.canvas = make_scene_canvas(size=(360, 360))
        self.color_axis = ColorAxis()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.canvas.native, 1)
        row.addWidget(self.color_axis)
        grid = self.canvas.central_widget.add_grid(spacing=0, margin=2)
        self._title = scene.Label(PLANES[plane], color=FG, font_size=8)
        self._title.height_min, self._title.height_max = 16, 18
        grid.add_widget(self._title, row=0, col=1)
        self.view = grid.add_view(row=1, col=1)
        lock_panzoom(self.view, scene.PanZoomCamera(aspect=1.0))
        self._axes = [_labelled_axis(grid, self.view, o, row=r, col=c) for o, r, c in (("bottom", 2, 1), ("left", 1, 0))]
        self.view.scene.transform.changed.connect(self._pin_axes, position="last")
        for axis in self._axes:
            axis.events.resize.connect(self._pin_axes, position="last")
        self.turns = 0  # quarter turns counterclockwise
        self._size = (1.0, 1.0)
        self._plane = scene.Node(parent=self.view.scene)  # plane mm; its transform turns them
        self._plane.transform = scene.transforms.MatrixTransform()
        self._box = add_line(self._plane, width=1.0, order=ORDER_OVERLAY)
        self.rotate_button = tool_button("⟲ 90°", "Rotate this plane a quarter turn counterclockwise")
        self.rotate_button.clicked.connect(lambda _c=False: self.rotate())
        self.integral_button = tool_button("∫", "Sum through the volume instead of showing one slice", checkable=True)
        self.integral_button.toggled.connect(lambda _on: self.changed.emit())
        self.tools = [self.integral_button, self.rotate_button]
        self._ct = scene.visuals.Image(np.zeros((2, 2), np.float32), cmap="grays", interpolation="linear",
                                       texture_format="auto", parent=self._plane)
        self._wash = scene.visuals.Image(np.zeros((2, 2, 4), np.uint8), interpolation="linear", parent=self._plane)
        self._lines = scene.visuals.Line(np.zeros((2, 2), np.float32), connect="segments", method="gl", width=1.5,
                                         antialias=True, parent=self._plane)
        for visual, order in ((self._ct, ORDER_FILL), (self._wash, ORDER_DATA), (self._lines, ORDER_OVERLAY)):
            _gl_2d(visual, order)
        for visual in (self._ct, self._wash):
            visual.transform = scene.transforms.STTransform()
        self.canvas.events.mouse_press.connect(self._press)
        self.canvas.events.mouse_move.connect(self._drag)
        self.canvas.events.mouse_wheel.connect(self._wheel)

    @property
    def integral(self) -> bool:
        return self.integral_button.isChecked()

    def set_theme(self, bg, fg) -> None:
        if self._theme == (bg, fg):
            return
        self._theme = (bg, fg)
        self.canvas.bgcolor = self.view.bgcolor = bg
        self._title._text_visual.color = fg  # vispy Label has no public color setter
        self._box.set_data(color=_rgba(fg, BOX_ALPHA))
        for axis in self._axes:
            _ink_axis(axis, fg)
        self.canvas.update()

    def set_frame(self, frame: DoseFrame) -> None:
        self._frame = frame
        self.set_plane(self.plane)

    def set_plane(self, plane: int) -> None:
        self.plane = plane
        f = self._frame
        if f is None:
            return
        (a, b), sp, ext = PLANE_AXES[plane], np.asarray(f.spacing, dtype=float), f.extent
        self._ct.transform.scale = self._wash.transform.scale = (sp[a], sp[b])
        self.view.camera.flip = (False, f.flip_axial and plane == 0)  # row 0 at the top, as a CT is read
        self._size = (float(ext[a]), float(ext[b]))
        w, h = self._size
        self._box.set_data(pos=np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], np.float32))
        self._turn()

    def _pin_axes(self, _event=None) -> None:
        """Ticks in the frame's mm, whichever way the plane is turned or flipped."""
        f = self._frame
        if f is None:
            return
        for axis in self._axes:
            ends = axis.node_transform(self.view.scene).map(axis._axis_ends())[:, :2]
            ends = self._plane.transform.imap(ends)[:, :2]  # into plane mm
            k = int(np.argmax(np.abs(ends[1] - ends[0])))
            g = PLANE_AXES[self.plane][k]
            axis.axis.domain = (f.origin[g] + ends[0, k], f.origin[g] + ends[1, k])
            if axis.axis.axis_label != f.names[g]:
                axis.axis.axis_label = f.names[g]

    def rotate(self, turns: int = 1) -> None:
        """Turn the plane by quarter turns counterclockwise."""
        self.turns = (self.turns + turns) % 4
        self._turn()
        self.canvas.update()

    def _turn(self) -> None:
        w, h = self._size
        c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[self.turns]
        m = np.eye(4)
        m[:2, :2] = ((c, s), (-s, c))  # vispy maps row vectors: p @ m
        m[3, :2] = ((0, 0), (h, 0), (w, h), (0, w))[self.turns]  # back into the positive quadrant
        self._plane.transform.matrix = m
        across, up = (h, w) if self.turns % 2 else (w, h)
        set_data_range(self.view, (0.0, across), (0.0, up))
        self._pin_axes()

    def show(self, ct: np.ndarray | None, wash: np.ndarray | None, lines, title: str, scale=None) -> None:
        """CT slice (HU) or None, wash RGBA or None, (segments mm (2n, 2), colors (2n, 4)),
        and the wash's color bar ``(scale, lo, hi, title)`` or None."""
        self.color_axis.set_scale(*(scale or ("", 0.0, 1.0, "")))
        self._ct.visible = ct is not None
        if ct is not None:
            self._ct.set_data(np.ascontiguousarray(ct, dtype=np.float32))
            self._ct.clim = CT_WINDOW
        self._wash.visible = wash is not None
        if wash is not None:
            self._wash.set_data(wash)
        self._lines.visible = True
        self._lines.set_data(pos=lines[0], color=lines[1])
        self._title.text = title
        self.canvas.update()

    def clear(self, message: str = "") -> None:
        self._ct.visible = self._wash.visible = self._lines.visible = False
        self._title.text = message or PLANES[self.plane]
        self.color_axis.set_scale("", 0.0, 1.0, "")
        self.canvas.update()

    def _hit(self, pos):
        local = self.canvas.scene.node_transform(self.view).map(pos)[:2]
        if not (0 <= local[0] < self.view.size[0] and 0 <= local[1] < self.view.size[1]):
            return None
        x, y = self.canvas.scene.node_transform(self._plane).map(pos)[:2]
        return float(x), float(y)

    def _press(self, event) -> None:
        hit = self._hit(event.pos)
        if hit is not None and event.button == 1:
            self.picked.emit(*hit)

    def _drag(self, event) -> None:
        if event.is_dragging and event.buttons and 1 in event.buttons:
            hit = self._hit(event.pos)
            if hit is not None:
                self.picked.emit(*hit)

    def _wheel(self, event) -> None:
        if self._hit(event.pos) is not None:
            event.handled = True
            self.paged.emit(1 if event.delta[1] > 0 else -1)


class PlotPane(QWidget):
    """A line plot or histogram under a picker of what it shows."""

    kindChanged = Signal(str)

    def __init__(self, kind: str = DVH, parent: QWidget | None = None) -> None:
        from vispy import scene

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = ViewHeader(PLOT_KINDS)
        self.integral_button = tool_button(
            "∫", "Sum the dose over each plane across the line instead of one line through the crosshair",
            checkable=True)
        self.integral_button.toggled.connect(lambda _on: self.kindChanged.emit(self.kind))
        self.header.picked.connect(self._picked)
        self.set_kind(kind, quiet=True)
        layout.addWidget(self.header)
        self.canvas = make_scene_canvas(size=(500, 260))
        layout.addWidget(self.canvas.native, 1)
        grid = self.canvas.central_widget.add_grid(spacing=0, margin=4)
        self.view = grid.add_view(row=0, col=1)
        lock_panzoom(self.view)
        self._fg = FG
        self._domains: dict = {}
        self._axis_y = self._axis(grid, "left", row=0, col=0)
        self._axis_x = self._axis(grid, "bottom", row=1, col=1)
        self._labels = ("", "")
        set_data_range(self.view, (0.0, _PLOT_SPAN), (0.0, _PLOT_SPAN))
        self._lines: list = []  # reused across redraws: making vispy nodes costs ~10 ms each
        self._bars = scene.visuals.Mesh(parent=self.view.scene)
        _gl_2d(self._bars, ORDER_FILL)
        self._marker = add_line(self.view.scene, color="#e5484d", width=1.4, order=ORDER_OVERLAY)
        # The plot's y runs up, which turns vispy's text anchors over: "bottom" hangs each line below its point.
        self._legend = scene.Text("", font_size=7, anchor_x="right", anchor_y="bottom", parent=self.view.scene)
        self._message = scene.Text("", font_size=9, parent=self.view.scene)
        self._legend.order = self._message.order = ORDER_OVERLAY

    @property
    def kind(self) -> str:
        return self.header.current

    @property
    def integral(self) -> bool:
        return self.integral_button.isChecked() and self.kind in PROFILES

    def set_kind(self, kind: str, *, quiet: bool = False) -> None:
        self.header.set_current(kind, quiet=True)
        self._picked(None if quiet else kind)

    def _picked(self, kind) -> None:
        self.header.set_tools([self.integral_button] if self.kind in PROFILES else [])
        if kind is not None:
            self.kindChanged.emit(self.kind)

    def set_labels(self, x: str, y: str) -> None:
        """Axis titles, with units."""
        if (x, y) != self._labels:
            self._labels = (x, y)
            self._axis_x.axis.axis_label, self._axis_y.axis.axis_label = x or " ", y or " "

    def _axis(self, grid, orientation: str, *, row: int, col: int):
        axis = _labelled_axis(grid, self.view, orientation, row=row, col=col)

        def pin(_event=None) -> None:
            if axis in self._domains:
                axis.axis.domain = self._domains[axis]

        self.view.scene.transform.changed.connect(pin, position="last")
        axis.events.resize.connect(pin, position="last")
        return axis

    def set_theme(self, bg: str, fg: str) -> None:
        self._fg = fg
        self.canvas.bgcolor = self.view.bgcolor = bg
        for axis in (self._axis_x, self._axis_y):
            _ink_axis(axis, fg)
        self._message.color = _rgba(fg, 0.7)
        self.canvas.update()

    def _set_domains(self, x, y) -> None:
        for axis, dom in ((self._axis_x, x), (self._axis_y, y)):
            self._domains[axis] = axis.axis.domain = (float(dom[0]), float(dom[1]))

    def _scaled(self, x, y, x_dom, y_dom) -> np.ndarray:
        sx = _PLOT_SPAN / ((x_dom[1] - x_dom[0]) or 1.0)
        sy = _PLOT_SPAN / ((y_dom[1] - y_dom[0]) or 1.0)
        return np.column_stack([(np.asarray(x) - x_dom[0]) * sx, (np.asarray(y) - y_dom[0]) * sy]).astype(np.float32)

    def _say(self, message: str, note: str) -> None:
        self._message.text = message or " "
        self._message.pos = (0.5 * _PLOT_SPAN, 0.5 * _PLOT_SPAN)
        self.header.note.setText(note)

    def show_curves(self, curves, legend, x_dom, y_dom, note: str = "", message: str = "") -> None:
        """*curves* ``[(x, y, rgba, width)]`` in data units over *x_dom* × *y_dom*; *legend* ``[(name, rgba)]``."""
        self._bars.visible = self._marker.visible = False
        while len(self._lines) < len(curves):
            self._lines.append(add_line(self.view.scene, color="w", width=1.0, order=ORDER_DATA))
        for line, (x, y, color, width) in zip(self._lines, curves):
            line.set_data(pos=self._scaled(x, y, x_dom, y_dom), color=color, width=width)
        for n, line in enumerate(self._lines):
            line.visible = n < len(curves)
        self._legend.visible = bool(legend)
        if legend:
            fg = np.array(_rgba(self._fg), np.float32)
            self._legend.text = [name for name, _ in legend]
            self._legend.color = 0.6 * np.array([c for _, c in legend], np.float32) + 0.4 * fg  # readable on any bg
            self._legend.pos = np.array([(98.0, 98.0 - 7.0 * n) for n in range(len(legend))], np.float32)
        self._set_domains(x_dom, y_dom)
        self._say(message, note)
        self.canvas.update()

    def show_bars(self, counts, edges, marker: float | None, note: str = "", message: str = "") -> None:
        """Histogram *counts* over *edges*, with a vertical line at *marker*."""
        for line in self._lines:
            line.visible = False
        self._legend.visible = False
        show = counts is not None and np.any(counts)
        self._bars.visible = bool(show)
        self._marker.visible = bool(show) and marker is not None
        if show:
            x_dom, y_dom = (float(edges[0]), float(edges[-1])), (0.0, float(np.max(counts)) * 1.08)
            y = np.asarray(counts, dtype=float)
            quad = [self._scaled(xs, ys, x_dom, y_dom) for xs, ys in
                    ((edges[:-1], 0 * y), (edges[1:], 0 * y), (edges[1:], y), (edges[:-1], y))]
            verts = np.stack(quad, axis=1).reshape(-1, 2)
            base = 4 * np.arange(len(y), dtype=np.uint32)[:, None]
            faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]]).astype(np.uint32)
            self._bars.set_data(vertices=np.column_stack([verts, np.zeros(len(verts))]).astype(np.float32),
                                faces=faces, color=_rgba("#44aa88", 0.9))
            if marker is not None:
                self._marker.set_data(pos=self._scaled([marker, marker], [0.0, y_dom[1]], x_dom, y_dom))
            self._set_domains(x_dom, y_dom)
        self._say("" if show else message, note)
        self.canvas.update()


class VolumePane(QWidget):
    """The 3D canvas with its color axis glued to the right; a :class:`VolumeScene` draws into :attr:`canvas`."""

    def __init__(self, *, gl: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.canvas = make_scene_canvas(keys="interactive", size=(600, 600), gl=gl)
        self.color_axis = ColorAxis()
        row.addWidget(self.canvas.native, 1)
        row.addWidget(self.color_axis)
        self.tools: list[QWidget] = []  # the 3D view's own settings, shown in whichever header holds it


class _Cell(QWidget):
    """A grid cell: its own slice pane, or the one 3D pane, under a header that picks which."""

    picked = Signal(int)

    def __init__(self, pane: SlicePane, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = ViewHeader(list(enumerate(VIEWS)))
        self.header.picked.connect(self.picked.emit)
        layout.addWidget(self.header)
        self.body = QVBoxLayout()
        layout.addLayout(self.body, 1)
        self.slice = pane
        self.body.addWidget(pane)
        self.view = pane.plane


class DoseWorkspace(QWidget):
    """Four cells (axial and 3D over coronal and sagittal by default, each picks its view) above two plots."""

    cursorMoved = Signal()

    def __init__(self, *, gl: str | None = None, plots: Sequence[str] = (DVH, GAMMA_HIST),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: DoseFrame | None = None
        self._layers = DoseLayers()
        self._checked: list[int] = []
        self._cursor = np.zeros(3, dtype=int)
        self._contours: dict[tuple[int, int, int], np.ndarray] = {}  # (plane, slice or -1 summed, roi) -> segments mm
        self._sums: dict[tuple, object] = {}  # projections and integral profiles of the current layers
        self.volume = VolumePane(gl=gl)
        self._park = QWidget(self)  # the 3D pane waits here, hidden, while no cell shows it
        self._park.hide()
        self.plots = [PlotPane(kind) for kind in plots]
        for pane in self.plots:
            pane.kindChanged.connect(lambda _k, pane=pane: self._draw_plot(pane))

        self.cells = [_Cell(SlicePane(0 if view == VIEW_3D else view)) for view in DEFAULT_CELLS]
        self.slices = [cell.slice for cell in self.cells]
        for cell, view in zip(self.cells, DEFAULT_CELLS):
            pane = cell.slice
            pane.picked.connect(lambda x, y, pane=pane: self._pick(pane.plane, x, y))
            pane.paged.connect(lambda steps, pane=pane: self._page(pane.plane, steps))
            pane.changed.connect(lambda pane=pane: self._draw_slice(pane))
            self._hold(cell, view)
            cell.picked.connect(lambda view, cell=cell: self.show_view(cell, view))

        # Every boundary drags: the two grid rows share their column split so the 2×2 stays square.
        top = _splitter(Qt.Orientation.Horizontal, *self.cells[:2])
        bottom = _splitter(Qt.Orientation.Horizontal, *self.cells[2:])
        top.splitterMoved.connect(lambda *_: bottom.setSizes(top.sizes()))
        bottom.splitterMoved.connect(lambda *_: top.setSizes(bottom.sizes()))
        plot_row = _splitter(Qt.Orientation.Horizontal, *self.plots)
        self.splitter = _splitter(Qt.Orientation.Vertical, top, bottom, plot_row)
        self.splitter.setSizes([350, 350, 260])
        self.rows = (top, bottom, plot_row)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self._apply_theme()
        self.clear("")

    # ---- cells ----------------------------------------------------------------------------------

    def show_view(self, cell: _Cell, view: int) -> None:
        """Show *view* in *cell*. Any cell may show any plane; the one 3D pane leaves its old cell,
        which takes this cell's plane."""
        if view == cell.view:
            return
        holder = next((c for c in self.cells if c.view == VIEW_3D), None)
        if view == VIEW_3D and holder is not None:
            self._hold(holder, cell.view)
        elif cell.view == VIEW_3D:
            self.volume.setParent(self._park)
        self._hold(cell, view)

    def set_volume_tools(self, widgets: Sequence[QWidget]) -> None:
        """The 3D view's settings, for the header of whichever cell shows it."""
        self.volume.tools = list(widgets)
        for cell in self.cells:
            if cell.view == VIEW_3D:
                cell.header.set_tools(self.volume.tools)

    def _hold(self, cell: _Cell, view: int) -> None:
        cell.view = view
        cell.header.set_current(view, quiet=True)
        cell.slice.setVisible(view != VIEW_3D)
        if view == VIEW_3D:
            cell.body.addWidget(self.volume)
            self.volume.show()
            cell.header.set_tools(self.volume.tools)
        else:
            cell.slice.set_plane(view)
            cell.header.set_tools(cell.slice.tools)
            self._draw_slice(cell.slice)

    # ---- theme ----------------------------------------------------------------------------------

    def _apply_theme(self) -> None:
        self._palette = self.palette().window().color().name(), self.palette().windowText().color().name()
        for pane in (*self.slices, *self.plots):
            pane.set_theme(*self._palette)
        if hasattr(self, "_layers"):
            self._draw_slices()

    def _slice_colors(self) -> tuple:
        """Background and ink: the scale's zero color under a dose, as in the 3D view, else the palette."""
        lay = self._layers
        if lay.wash is None:
            return self._palette
        bg = zero_rgb(lay.scale, lay.lo, lay.hi)
        return bg, ink_rgb(bg)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "plots"):
            self._apply_theme()

    # ---- state ----------------------------------------------------------------------------------

    @property
    def frame(self) -> DoseFrame | None:
        return self._frame

    @property
    def cursor(self) -> np.ndarray:
        return self._cursor.copy()

    def clear(self, message: str) -> None:
        self._frame = None
        self._layers = DoseLayers()
        for n, pane in enumerate(self.slices):
            pane.set_theme(*self._palette)
            pane.clear(message if n == 0 else "")
        self._draw_plots()

    def set_frame(self, frame: DoseFrame, cursor=None) -> None:
        """A new grid, with no layers until they are set on it; the cursor goes to *cursor* (x, y, z index) or the middle."""
        self._frame = frame
        self._layers = DoseLayers()
        self._contours.clear()
        self._sums.clear()
        shape = np.asarray(frame.shape)
        self._cursor = np.clip(np.rint(shape // 2 if cursor is None else cursor).astype(int), 0, shape - 1)
        for pane in self.slices:
            pane.set_frame(frame)

    def set_layers(self, layers: DoseLayers) -> None:
        self._layers = layers
        self._sums.clear()
        self._draw_slices()
        self._draw_plots()

    def set_checked(self, rois: Sequence[int]) -> None:
        """ROI indices to outline and to draw DVHs of."""
        self._checked = list(rois)
        self._draw_slices()
        self._draw_plots()

    def set_plot_kinds(self, kinds: Sequence[str]) -> None:
        for pane, kind in zip(self.plots, kinds):
            pane.blockSignals(True)
            pane.set_kind(kind)
            pane.blockSignals(False)
        self._draw_plots()

    def set_cursor(self, cursor) -> None:
        if self._frame is None:
            return
        cursor = np.clip(np.asarray(cursor, dtype=int), 0, np.asarray(self._frame.shape) - 1)
        if not np.array_equal(cursor, self._cursor):
            self._cursor = cursor
            self._draw_slices()
            for pane in self.plots:
                if pane.kind in PROFILES:
                    self._draw_plot(pane)
            self.cursorMoved.emit()

    def _pick(self, plane: int, x: float, y: float) -> None:
        if self._frame is None:
            return
        (a, b), sp = PLANE_AXES[plane], self._frame.spacing
        cursor = self._cursor.copy()
        cursor[a], cursor[b] = int(x // sp[a]), int(y // sp[b])
        self.set_cursor(cursor)

    def _page(self, plane: int, steps: int) -> None:
        cursor = self._cursor.copy()
        cursor[PLANE_NORMAL[plane]] += steps
        self.set_cursor(cursor)

    # ---- drawing --------------------------------------------------------------------------------

    def _draw_slices(self) -> None:
        for pane in self.slices:
            self._draw_slice(pane)

    def _draw_slice(self, pane: SlicePane) -> None:
        f, lay = self._frame, self._layers
        if f is None:
            pane.clear("")
            return
        if pane.isHidden():  # its cell shows 3D; drawn when it comes back
            return
        p, summed = pane.plane, pane.integral
        sp, ext = np.asarray(f.spacing, dtype=float), f.extent
        center = (self._cursor + 0.5) * sp
        bg, ink = self._slice_colors()
        pane.set_theme(bg, ink)
        (a, b), n = PLANE_AXES[p], PLANE_NORMAL[p]
        index = -1 if summed else int(self._cursor[n])
        w, h, cx, cy = ext[a], ext[b], center[a], center[b]
        segs = [np.array([[cx, 0.0], [cx, h], [0.0, cy], [w, cy]], np.float32)]
        colors = [np.tile(np.float32(_rgba(ink, 0.45)), (4, 1))]
        for r in (r for r in self._checked if r < len(f.rois)):
            s = self._outline(p, index, r, (sp[a], sp[b]))
            segs.append(s)
            colors.append(np.tile(np.float32([*(np.asarray(f.rois[r][1]) / 255.0), 1.0]), (len(s), 1)))
        wash = scale = None
        if lay.wash is not None:
            img, lo, hi, unit = plane_cut(lay.wash, p, index), lay.lo, lay.hi, "γ" if lay.gamma else lay.unit
            if summed:
                # γ has no meaningful sum; its worst value along the ray stands in.
                img = self._summed(("wash", p), lambda: lay.wash.max(axis=p) if lay.gamma
                                   else lay.wash.sum(axis=p, dtype=float) * sp[n])
                if not lay.gamma:
                    peak = float(np.abs(img).max()) or 1.0
                    lo, hi, unit = (-peak if lo < 0 else 0.0), peak, f"{unit}·mm"
            lut = (255 * colormap_samples(lay.scale)).astype(np.uint8)
            # Over a CT the dose is a translucent wash; alone it is the picture.
            wash = wash_rgba(img, lut, lo, hi, lay.gamma, OVERLAY_ALPHA if f.ct is not None else 1.0)
            scale = (lay.scale, lo, hi, unit)
        ct = None
        if f.ct is not None:
            ct = self._summed(("ct", p), lambda: f.ct.mean(axis=p)) if summed else plane_cut(f.ct, p, index)
        worst = lay.gamma and lay.wash is not None
        title = f"{'Max' if worst else 'Sum'} of {f.shape[n]} slices" if summed else f"Slice {index + 1}/{f.shape[n]}"
        pane.show(ct, wash, (np.concatenate(segs), np.concatenate(colors)), title, scale)

    def _summed(self, key, make):
        if key not in self._sums:
            self._sums[key] = make()
        return self._sums[key]

    def _outline(self, plane: int, index: int, r: int, pixel) -> np.ndarray:
        """ROI *r*'s outline on slice *index*, or of its shadow through the volume when *index* is −1."""
        key = (plane, index, r)
        if key not in self._contours:
            from ..dicom.structures import MAX_MASK_BITS

            bit = (self._frame.bits[r // MAX_MASK_BITS] >> np.uint32(r % MAX_MASK_BITS)) & 1
            mask = bit.any(axis=plane) if index < 0 else plane_cut(bit, plane, index)
            self._contours[key] = outline_segments(mask, pixel)
        return self._contours[key]

    def _draw_plots(self) -> None:
        for pane in self.plots:
            self._draw_plot(pane)

    def _draw_plot(self, pane: PlotPane) -> None:
        f, lay = self._frame, self._layers
        kind = pane.kind
        if kind == GAMMA_HIST:
            g = lay.gamma_values
            if g is None or f is None:
                pane.show_bars(None, None, None, "", "No gamma")
                return
            cap = 2.0
            pane.set_labels("γ", "Voxels")
            counts, edges = np.histogram(np.minimum(g[g > 0], cap), bins=50, range=(0.0, cap))
            pane.show_bars(counts, edges, 1.0, lay.gamma_note)
            return
        if kind == DVH:
            self._draw_dvh(pane)
            return
        summed = pane.integral
        pane.set_labels(_PROFILE_X[kind], f"Integral dose ({lay.unit}·mm²)" if summed else f"Dose ({lay.unit})")
        if f is None or not lay.profiles:
            pane.show_curves([], [], (0.0, 1.0), (0.0, 1.0), "", "Profiles when the dose is ready")
            return
        lateral = [(lateral_dir(f.beam_dir), "lateral")]
        along = [(np.asarray(f.beam_dir, dtype=float), "longitudinal")]
        lines = {DEPTH: along, LATERAL: lateral, LONGITUDINAL: along, LAT_LONG: lateral + along}[kind]
        curves, legend, peak, x_lo, x_hi = [], [], 0.0, np.inf, -np.inf
        for n, (label, vol) in enumerate(lay.profiles.items()):
            for m, (direction, what) in enumerate(lines):
                t, dose = self._profile(label, vol, direction, summed)
                if not t.size:
                    continue
                x = t - t[0] if kind == DEPTH else t
                hue = CURVE_COLORS[(n * len(lines) + m) % len(CURVE_COLORS)]
                color = _rgba(hue, 1.0 if label == lay.main or not lay.main else 0.7)
                curves.append((x, dose, color, 1.6))
                legend.append((f"{label} {what}" if len(lines) > 1 else label, color))
                peak, x_lo, x_hi = max(peak, float(dose.max())), min(x_lo, float(x[0])), max(x_hi, float(x[-1]))
        if not curves:
            pane.show_curves([], [], (0.0, 1.0), (0.0, 1.0), "", "The crosshair is off the grid")
            return
        pane.show_curves(curves, legend, (x_lo, x_hi), (0.0, (peak or 1.0) * 1.08))

    def _profile(self, label: str, vol: np.ndarray, direction, summed: bool) -> tuple[np.ndarray, np.ndarray]:
        """Signed mm from the crosshair along *direction*, and the dose on that line or summed across it."""
        f = self._frame
        if not summed:
            return line_profile(vol, f, self._cursor, direction)
        d = np.asarray(direction, dtype=float)
        d = d / (np.linalg.norm(d) or 1.0)
        t, dose = self._summed(("profile", label, *d), lambda: integral_profile(vol, f, d))
        return t - float((self._cursor + 0.5) * np.asarray(f.spacing, dtype=float) @ d), dose

    def _draw_dvh(self, pane: PlotPane) -> None:
        f, lay = self._frame, self._layers
        pane.set_labels(f"Dose ({lay.unit})", "Volume (%)")
        names = {f.rois[r][0]: f.rois[r][1] for r in self._checked if f is not None and r < len(f.rois)}
        curves, legend, top = [], [], 0.0
        for label, by_roi in lay.dvh.items():
            faded = bool(lay.main) and label != lay.main
            for name, rgb in names.items():
                h = by_roi.get(name)
                if h is None:
                    continue
                color = (*(np.asarray(rgb) / 255.0), 0.5 if faded else 1.0)
                curves.append((h.edges, 100.0 * h.cumulative, color, 1.1 if faded else 1.6))
                if h.cumulative.any():
                    top = max(top, float(h.edges[np.flatnonzero(h.cumulative > 0)[-1] + 1]))
                if not faded:
                    legend.append((name, color))
        if not curves:
            why = "No structures in this frame" if f is not None and not f.rois else "DVH when the dose is ready"
            pane.show_curves([], [], (0.0, 1.0), (0.0, 102.0), "", why)
            return
        others = [k for k in lay.dvh if k != lay.main]
        note = f"{lay.main} bright, {', '.join(others)} faded" if others else ""
        pane.show_curves(curves, legend, (0.0, (top or 1.0) * 1.05), (0.0, 102.0), note)
