"""Dose panes to mix and match: slice planes, line plots, and the 3D volume with its color axis.

Each pane is its own canvas, so windows and tabs lay them out with Qt, and a cursor move
repaints only the panes it changes. :class:`DoseWorkspace` is the standard arrangement:
three planes and the 3D volume in a 2×2 grid, over two plots side by side.

A source describes its grid once (:class:`DoseFrame`) and then what to draw on it
(:class:`DoseLayers`). Visuals are made once, and later calls only feed them data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from .color_axis import ColorAxis
from .dose_volume_catalog import DEFAULT_SCALE
from .dose_volume_fill import colormap_samples, ink_rgb, zero_rgb
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
OVERLAY_ALPHA = 0.55
WASH_FLOOR = 0.1  # dose below this share of the maximum is not washed
CT_WINDOW = (-500.0, 500.0)  # HU shown black to white
DVH, GAMMA_HIST, DEPTH, LATERAL = "dvh", "gamma", "depth", "lateral"
PLOT_KINDS = ((DVH, "DVH"), (GAMMA_HIST, "Gamma histogram"), (DEPTH, "Depth dose"), (LATERAL, "Lateral profile"))
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
    return split


def _gl_2d(visual, order: int) -> None:
    visual.set_gl_state("translucent", depth_test=False, depth_mask=False, cull_face=False)
    visual.order = order


def _rgba(color, alpha: float = 1.0) -> tuple[float, float, float, float]:
    from vispy.color import Color

    return (*Color(color).rgb, alpha)


class SlicePane:
    """One plane: CT, dose wash, outlines and crosshair. *on_pick(x_mm, y_mm)* on click or drag, *on_page(steps)* on scroll."""

    def __init__(self, plane: int, on_pick: Callable[[float, float], None], on_page: Callable[[int], None]) -> None:
        from vispy import scene

        self.plane = plane
        self._on_pick, self._on_page = on_pick, on_page
        self.canvas = make_scene_canvas(size=(360, 360))
        grid = self.canvas.central_widget.add_grid(margin=2)
        self._title = scene.Label(PLANES[plane], color=FG, font_size=8)
        self._title.height_min, self._title.height_max = 16, 18
        grid.add_widget(self._title, row=0, col=0)
        self.view = grid.add_view(row=1, col=0)
        lock_panzoom(self.view, scene.PanZoomCamera(aspect=1.0))
        self._ct = scene.visuals.Image(np.zeros((2, 2), np.float32), cmap="grays", interpolation="linear",
                                       texture_format="auto", parent=self.view.scene)
        self._wash = scene.visuals.Image(np.zeros((2, 2, 4), np.uint8), interpolation="linear", parent=self.view.scene)
        self._lines = scene.visuals.Line(np.zeros((2, 2), np.float32), connect="segments", method="gl", width=1.5,
                                         antialias=True, parent=self.view.scene)
        for visual, order in ((self._ct, ORDER_FILL), (self._wash, ORDER_DATA), (self._lines, ORDER_OVERLAY)):
            _gl_2d(visual, order)
        for visual in (self._ct, self._wash):
            visual.transform = scene.transforms.STTransform()
        self.canvas.events.mouse_press.connect(self._press)
        self.canvas.events.mouse_move.connect(self._drag)
        self.canvas.events.mouse_wheel.connect(self._wheel)

    @property
    def native(self):
        return self.canvas.native

    def set_theme(self, bg, fg) -> None:
        self.canvas.bgcolor = self.view.bgcolor = bg
        self._title._text_visual.color = fg  # vispy Label has no public color setter
        self.canvas.update()

    def set_extent(self, size: tuple[float, float], pixel: tuple[float, float], flip: bool) -> None:
        """Plane (width, height) and pixel (across, up) in mm; *flip* puts row 0 at the top."""
        self._ct.transform.scale = self._wash.transform.scale = pixel
        self.view.camera.flip = (False, flip)
        set_data_range(self.view, (0.0, size[0]), (0.0, size[1]))

    def show(self, ct: np.ndarray | None, wash: np.ndarray | None, lines, title: str) -> None:
        """CT slice (HU) or None, wash RGBA or None, (segments mm (2n, 2), colors (2n, 4))."""
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
        self.canvas.update()

    def _hit(self, pos):
        local = self.canvas.scene.node_transform(self.view).map(pos)[:2]
        if not (0 <= local[0] < self.view.size[0] and 0 <= local[1] < self.view.size[1]):
            return None
        x, y = self.canvas.scene.node_transform(self.view.scene).map(pos)[:2]
        return float(x), float(y)

    def _press(self, event) -> None:
        hit = self._hit(event.pos)
        if hit is not None and event.button == 1:
            self._on_pick(*hit)

    def _drag(self, event) -> None:
        if event.is_dragging and event.buttons and 1 in event.buttons:
            hit = self._hit(event.pos)
            if hit is not None:
                self._on_pick(*hit)

    def _wheel(self, event) -> None:
        if self._hit(event.pos) is not None:
            event.handled = True
            self._on_page(1 if event.delta[1] > 0 else -1)


class PlotPane(QWidget):
    """A line plot or histogram under a picker of what it shows."""

    kindChanged = Signal(str)

    def __init__(self, kind: str = DVH, parent: QWidget | None = None) -> None:
        from vispy import scene

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 0)
        self._picker = QComboBox()
        for key, text in PLOT_KINDS:
            self._picker.addItem(text, key)
        self._picker.setCurrentIndex(max(self._picker.findData(kind), 0))
        self._picker.currentIndexChanged.connect(lambda _i: self.kindChanged.emit(self.kind))
        self._note = QLabel("")
        self._note.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._picker)
        header.addWidget(self._note, 1)
        layout.addLayout(header)
        self.canvas = make_scene_canvas(size=(500, 260))
        layout.addWidget(self.canvas.native, 1)
        grid = self.canvas.central_widget.add_grid(spacing=0, margin=4)
        self.view = grid.add_view(row=0, col=1)
        lock_panzoom(self.view)
        self._fg = FG
        self._domains: dict = {}
        self._axis_y = self._axis(grid, "left", row=0, col=0)
        self._axis_x = self._axis(grid, "bottom", row=1, col=1)
        set_data_range(self.view, (0.0, _PLOT_SPAN), (0.0, _PLOT_SPAN))
        self._lines: list = []  # reused across redraws: making vispy nodes costs ~10 ms each
        self._bars = scene.visuals.Mesh(parent=self.view.scene)
        _gl_2d(self._bars, ORDER_FILL)
        self._marker = add_line(self.view.scene, color="#e5484d", width=1.4, order=ORDER_OVERLAY)
        self._legend = scene.Text("", font_size=7, anchor_x="right", anchor_y="top", parent=self.view.scene)
        self._message = scene.Text("", font_size=9, parent=self.view.scene)
        self._legend.order = self._message.order = ORDER_OVERLAY

    @property
    def kind(self) -> str:
        return self._picker.currentData()

    def set_kind(self, kind: str) -> None:
        self._picker.setCurrentIndex(max(self._picker.findData(kind), 0))

    def _axis(self, grid, orientation: str, *, row: int, col: int):
        axis = axis_widget(orientation)
        grid.add_widget(axis, row=row, col=col)
        axis.link_view(self.view)

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
            axis.axis.text_color = fg
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
        self._note.setText(note)

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


class DoseWorkspace(QWidget):
    """Axial and 3D over coronal and sagittal, above two plots whose content each pane picks."""

    cursorMoved = Signal()

    def __init__(self, *, gl: str | None = None, plots: Sequence[str] = (DVH, GAMMA_HIST),
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: DoseFrame | None = None
        self._layers = DoseLayers()
        self._checked: list[int] = []
        self._cursor = np.zeros(3, dtype=int)
        self._contours: dict[tuple[int, int, int], np.ndarray] = {}  # (plane, slice, roi) -> segments mm
        self.slices = [SlicePane(p, lambda x, y, p=p: self._pick(p, x, y), lambda s, p=p: self._page(p, s))
                       for p in range(3)]
        self.volume = VolumePane(gl=gl)
        self.plots = [PlotPane(kind) for kind in plots]
        for pane in self.plots:
            pane.kindChanged.connect(lambda _k, pane=pane: self._draw_plot(pane))

        # Every boundary drags: the two grid rows share their column split so the 2×2 stays square.
        top = _splitter(Qt.Orientation.Horizontal, self.slices[0].native, self.volume)
        bottom = _splitter(Qt.Orientation.Horizontal, self.slices[1].native, self.slices[2].native)
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
        """A new grid; the cursor goes to *cursor* (x, y, z index) or the middle."""
        self._frame = frame
        self._contours.clear()
        shape = np.asarray(frame.shape)
        self._cursor = np.clip(np.rint(shape // 2 if cursor is None else cursor).astype(int), 0, shape - 1)
        sp, ext = np.asarray(frame.spacing, dtype=float), frame.extent
        for p, (a, b) in enumerate(PLANE_AXES):
            self.slices[p].set_extent((ext[a], ext[b]), (sp[a], sp[b]), frame.flip_axial and p == 0)

    def set_layers(self, layers: DoseLayers) -> None:
        self._layers = layers
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
                if pane.kind in (DEPTH, LATERAL):
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
        f, lay = self._frame, self._layers
        if f is None:
            return
        sp, ext = np.asarray(f.spacing, dtype=float), f.extent
        center = (self._cursor + 0.5) * sp
        lut = (255 * colormap_samples(lay.scale)).astype(np.uint8) if lay.wash is not None else None
        # Over a CT the dose is a translucent wash; alone it is the picture.
        alpha = OVERLAY_ALPHA if f.ct is not None else 1.0
        checked = [r for r in self._checked if r < len(f.rois)]
        bg, ink = self._slice_colors()
        cross = np.tile(np.float32(_rgba(ink, 0.45)), (4, 1))
        for p, pane in enumerate(self.slices):
            pane.set_theme(bg, ink)
            (a, b), index = PLANE_AXES[p], int(self._cursor[PLANE_NORMAL[p]])
            w, h, cx, cy = ext[a], ext[b], center[a], center[b]
            segs = [np.array([[cx, 0.0], [cx, h], [0.0, cy], [w, cy]], np.float32)]
            colors = [cross]
            for r in checked:
                s = self._outline(p, index, r, (sp[a], sp[b]))
                segs.append(s)
                colors.append(np.tile(np.float32([*(np.asarray(f.rois[r][1]) / 255.0), 1.0]), (len(s), 1)))
            wash = None if lut is None else wash_rgba(plane_cut(lay.wash, p, index), lut, lay.lo, lay.hi,
                                                      lay.gamma, alpha)
            ct = None if f.ct is None else plane_cut(f.ct, p, index)
            title = f"{PLANES[p]} · {index + 1}/{f.shape[PLANE_NORMAL[p]]}"
            pane.show(ct, wash, (np.concatenate(segs), np.concatenate(colors)), title)

    def _outline(self, plane: int, index: int, r: int, pixel) -> np.ndarray:
        key = (plane, index, r)
        if key not in self._contours:
            from ..dicom.structures import MAX_MASK_BITS

            bits = self._frame.bits[r // MAX_MASK_BITS]
            mask = plane_cut((bits >> np.uint32(r % MAX_MASK_BITS)) & 1, plane, index)
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
            counts, edges = np.histogram(np.minimum(g[g > 0], cap), bins=50, range=(0.0, cap))
            pane.show_bars(counts, edges, 1.0, lay.gamma_note)
            return
        if kind == DVH:
            self._draw_dvh(pane)
            return
        if f is None or not lay.profiles:
            pane.show_curves([], [], (0.0, 1.0), (0.0, 1.0), "", "Profiles when the dose is ready")
            return
        direction = f.beam_dir if kind == DEPTH else lateral_dir(f.beam_dir)
        curves, legend, peak, x_lo, x_hi = [], [], 0.0, np.inf, -np.inf
        for n, (label, vol) in enumerate(lay.profiles.items()):
            t, dose = line_profile(vol, f, self._cursor, direction)
            if not t.size:
                continue
            x = t - t[0] if kind == DEPTH else t
            color = _rgba(CURVE_COLORS[n % len(CURVE_COLORS)], 1.0 if label == lay.main or not lay.main else 0.7)
            curves.append((x, dose, color, 1.6))
            legend.append((label, color))
            peak, x_lo, x_hi = max(peak, float(dose.max())), min(x_lo, float(x[0])), max(x_hi, float(x[-1]))
        if not curves:
            pane.show_curves([], [], (0.0, 1.0), (0.0, 1.0), "", "The crosshair is off the grid")
            return
        what = "along the beam from the grid edge" if kind == DEPTH else "across the beam from the crosshair"
        pane.show_curves(curves, legend, (x_lo, x_hi), (0.0, (peak or 1.0) * 1.08), f"{lay.unit} vs mm {what}")

    def _draw_dvh(self, pane: PlotPane) -> None:
        f, lay = self._frame, self._layers
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
        note = f"volume % vs {lay.unit}" + (f" · {lay.main} bright, {', '.join(others)} faded" if others else "")
        pane.show_curves(curves, legend, (0.0, (top or 1.0) * 1.05), (0.0, 102.0), note)
