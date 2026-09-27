"""vispy panes of the patient QA window: tri-planar slices, color bar, DVH and gamma histogram.

One :class:`~vispy.scene.SceneCanvas` holds them all. Visuals are made once and only
their data changes, so paging slices or dragging the crosshair re-uploads three small
textures and three line buffers instead of re-rendering a figure.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .vispy_plot import ORDER_DATA, ORDER_FILL, ORDER_OVERLAY, add_line, axis_widget, lock_panzoom, set_data_range

VIEW_NAMES = ("Axial", "Coronal", "Sagittal")
OVERLAY_ALPHA = 0.55


def _gl_2d(visual, order: int) -> None:
    visual.set_gl_state("translucent", depth_test=False, depth_mask=False, cull_face=False)
    visual.order = order


def _rgba(color, alpha: float = 1.0) -> tuple[float, float, float, float]:
    from vispy.color import Color

    return (*Color(color).rgb, alpha)


class QaPlots:
    """The left pane. *on_pick(view, x_mm, y_mm)* on click or drag in a slice, *on_page(view, steps)* on scroll."""

    def __init__(self, on_pick: Callable[[int, float, float], None], on_page: Callable[[int, int], None]) -> None:
        from vispy import scene

        from .vispy_plot import make_scene_canvas

        self._on_pick, self._on_page = on_pick, on_page
        self.canvas = make_scene_canvas(size=(900, 800))
        self._fg = "#c9d1d9"
        self._labels: list = []
        self._axes: list = []
        self._domains: dict = {}
        top = self.canvas.central_widget.add_grid(spacing=4, margin=6)

        slices = top.add_grid(row=0, col=0)
        slices.stretch = (1, 2.6)
        self._titles, self._views, self._ct, self._wash, self._lines = [], [], [], [], []
        for c, name in enumerate(VIEW_NAMES):
            self._titles.append(self._label(slices, name, row=0, col=c))
            view = slices.add_view(row=1, col=c)
            view.stretch = (1, 12)
            lock_panzoom(view, scene.PanZoomCamera(aspect=1.0, flip=(False, c == 0)))
            ct = scene.visuals.Image(np.zeros((2, 2), np.float32), cmap="grays", interpolation="linear",
                                     texture_format="auto", parent=view.scene)
            wash = scene.visuals.Image(np.zeros((2, 2, 4), np.uint8), interpolation="linear", parent=view.scene)
            lines = scene.visuals.Line(np.zeros((2, 2), np.float32), connect="segments", method="gl", width=1.5,
                                       antialias=True, parent=view.scene)
            for visual, order in ((ct, ORDER_FILL), (wash, ORDER_DATA), (lines, ORDER_OVERLAY)):
                _gl_2d(visual, order)
                visual.transform = scene.transforms.STTransform()
            self._views.append(view)
            self._ct.append(ct)
            self._wash.append(wash)
            self._lines.append(lines)

        bar = top.add_grid(row=1, col=0)
        bar.stretch = (1, 0.75)
        self._bar_title = self._label(bar, "", row=0, col=1)
        for col in (0, 2):  # the bar spans the middle third
            bar.add_widget(scene.widgets.Widget(), row=1, col=col).stretch = (1, 1)
        self._bar_view = bar.add_view(row=1, col=1)
        self._bar_view.stretch = (1, 1)
        self._bar_view.height_min = 12
        lock_panzoom(self._bar_view)
        self._bar = scene.visuals.Image(np.zeros((1, 2, 4), np.uint8), parent=self._bar_view.scene)
        _gl_2d(self._bar, ORDER_FILL)
        self._bar_axis = self._axis(bar, self._bar_view, "bottom", row=2, col=1)

        plots = top.add_grid(row=2, col=0)
        plots.stretch = (1, 2.4)
        self._dvh_title = self._label(plots, "", row=0, col=1)
        self._gamma_title = self._label(plots, "", row=0, col=3)
        self._dvh_view = plots.add_view(row=1, col=1)
        self._dvh_view.stretch = (2.2, 1)
        lock_panzoom(self._dvh_view)
        self._dvh_axis_y = self._axis(plots, self._dvh_view, "left", row=1, col=0)
        self._dvh_axis_x = self._axis(plots, self._dvh_view, "bottom", row=2, col=1)
        self._gamma_view = plots.add_view(row=1, col=3)
        self._gamma_view.stretch = (1, 1)
        lock_panzoom(self._gamma_view)
        self._gamma_axis = self._axis(plots, self._gamma_view, "bottom", row=2, col=3)
        plots.add_widget(scene.widgets.Widget(), row=1, col=2).width_max = 16
        self._gamma_bars = scene.visuals.Mesh(parent=self._gamma_view.scene)
        _gl_2d(self._gamma_bars, ORDER_FILL)
        self._gamma_pass = add_line(self._gamma_view.scene, color="#e5484d", width=1.4, order=ORDER_OVERLAY)
        self._dvh_lines: list = []  # reused across redraws: making vispy nodes costs ~10 ms each
        self._legend = scene.Text("", font_size=7, anchor_x="right", anchor_y="top", parent=self._dvh_view.scene)
        self._message = scene.Text("", font_size=9, parent=self._dvh_view.scene)
        self._legend.order = self._message.order = ORDER_OVERLAY

        self.canvas.events.mouse_press.connect(self._press)
        self.canvas.events.mouse_move.connect(self._drag)
        self.canvas.events.mouse_wheel.connect(self._wheel)

    @property
    def native(self):
        return self.canvas.native

    # ---- layout helpers -------------------------------------------------------------------------

    def _label(self, grid, text: str, *, row: int, col: int):
        from vispy import scene

        label = scene.Label(text, color=self._fg, font_size=8)
        label.height_min, label.height_max = 16, 18
        grid.add_widget(label, row=row, col=col)
        self._labels.append(label)
        return label

    def _axis(self, grid, view, orientation: str, *, row: int, col: int):
        axis = axis_widget(orientation)
        axis.axis.text_color = self._fg
        grid.add_widget(axis, row=row, col=col)
        axis.link_view(view)

        def pin(_event=None) -> None:
            if axis in self._domains:
                axis.axis.domain = self._domains[axis]

        view.scene.transform.changed.connect(pin, position="last")
        axis.events.resize.connect(pin, position="last")
        self._axes.append(axis)
        return axis

    def _range(self, view, x, y, axes=()) -> None:
        set_data_range(view, x, y)
        for axis in axes:
            self._domains[axis] = y if axis.orientation == "left" else x
            axis.axis.domain = self._domains[axis]

    def set_theme(self, bg: str, fg: str) -> None:
        """Blend into the window: its background, its text color."""
        self._fg = fg
        self.canvas.bgcolor = bg
        for view in (*self._views, self._bar_view, self._dvh_view, self._gamma_view):
            view.bgcolor = bg
        for label in self._labels:
            label._text_visual.color = fg  # vispy Label has no public color setter
        for axis in self._axes:
            axis.axis.text_color = fg
        self._message.color = _rgba(fg, 0.7)
        self.canvas.update()

    # ---- slices ---------------------------------------------------------------------------------

    def set_extents(self, extents: Sequence[tuple[float, float]], pixels: Sequence[tuple[float, float]]) -> None:
        """Each view's (width, height) in mm and pixel size (x, y) in mm."""
        for view, (w, h), (sx, sy), ct, wash in zip(self._views, extents, pixels, self._ct, self._wash):
            ct.transform.scale = wash.transform.scale = (sx, sy)
            self._range(view, (0.0, w), (0.0, h))

    def set_slices(self, ct, wash, lines, titles, clim) -> None:
        """Per view: CT slice (HU), dose wash RGBA or None, (segments mm (2n, 2), colors (2n, 4)), title."""
        for n in range(3):
            self._ct[n].set_data(np.ascontiguousarray(ct[n], dtype=np.float32))
            self._ct[n].clim = clim
            self._wash[n].visible = wash is not None
            if wash is not None:
                self._wash[n].set_data(wash[n])
            pos, color = lines[n]
            self._lines[n].set_data(pos=pos, color=color)
            self._titles[n].text = titles[n]
        self.canvas.update()

    def clear(self, message: str) -> None:
        for n in range(3):
            self._ct[n].visible = self._wash[n].visible = self._lines[n].visible = False
            self._titles[n].text = message if n == 0 else ""
        self.set_colorbar(None, 0.0, 1.0, "")
        self.set_dvh([], [], 1.0, "")
        self.set_gamma(None, None, "")

    def show_slices(self) -> None:
        for n in range(3):
            self._ct[n].visible = self._lines[n].visible = True

    # ---- color bar, DVH, gamma ------------------------------------------------------------------

    def set_colorbar(self, lut: np.ndarray | None, lo: float, hi: float, title: str) -> None:
        """*lut* ``(n, 3)`` RGB 0-1 from *lo* to *hi*; None hides the bar."""
        self._bar.visible = self._bar_axis.visible = lut is not None
        self._bar_title.text = title
        if lut is None:
            return
        hi = hi if hi > lo else lo + 1.0
        rgba = np.concatenate([lut, np.ones((len(lut), 1))], axis=1)
        self._bar.set_data((255 * rgba[None]).astype(np.uint8))
        self._range(self._bar_view, (0.0, float(len(lut))), (0.0, 1.0), (self._bar_axis,))
        self._domains[self._bar_axis] = self._bar_axis.axis.domain = (lo, hi)

    def set_dvh(self, curves, legend, top: float, message: str) -> None:
        """*curves* ``[(dose Gy, volume %, rgba, width)]``; *legend* ``[(name, rgba)]`` top right."""
        top = top if top > 0 else 1.0
        while len(self._dvh_lines) < len(curves):
            self._dvh_lines.append(add_line(self._dvh_view.scene, color="w", width=1.0, order=ORDER_DATA))
        for line, curve in zip(self._dvh_lines, curves):
            x, y, color, width = curve
            # Scene x is % of *top*: agg lines misplace points when one axis spans ~1e-2 and the other 1e2.
            line.set_data(pos=np.column_stack([x * (100.0 / top), y]).astype(np.float32), color=color, width=width)
        for n, line in enumerate(self._dvh_lines):
            line.visible = n < len(curves)
        self._legend.visible = bool(legend)
        if legend:
            self._legend.text = [name for name, _ in legend]
            fg = np.array(_rgba(self._fg), np.float32)
            self._legend.color = 0.6 * np.array([color for _, color in legend], np.float32) + 0.4 * fg
            self._legend.pos = np.array([(98.0, 99.0 - 7.0 * n) for n in range(len(legend))], np.float32)
        self._message.text = message or " "
        self._message.pos = (50.0, 50.0)
        self._dvh_title.text = "DVH" if not message else ""
        self._range(self._dvh_view, (0.0, 100.0), (0.0, 102.0), (self._dvh_axis_x, self._dvh_axis_y))
        self._domains[self._dvh_axis_x] = self._dvh_axis_x.axis.domain = (0.0, top)
        self.canvas.update()

    def set_dvh_title(self, text: str) -> None:
        self._dvh_title.text = text

    def set_gamma(self, counts: np.ndarray | None, edges: np.ndarray | None, title: str) -> None:
        self._gamma_title.text = title
        show = counts is not None and counts.any()
        self._gamma_bars.visible = self._gamma_pass.visible = self._gamma_axis.visible = bool(show)
        if not show:
            return
        peak = float(counts.max()) * 1.08
        x0, x1 = edges[:-1], edges[1:]
        y = counts.astype(np.float32)
        verts = np.stack([np.column_stack([x0, 0 * y]), np.column_stack([x1, 0 * y]),
                          np.column_stack([x1, y]), np.column_stack([x0, y])], axis=1).reshape(-1, 2)
        base = 4 * np.arange(len(y), dtype=np.uint32)[:, None]
        faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]]).astype(np.uint32)
        self._gamma_bars.set_data(vertices=np.column_stack([verts, np.zeros(len(verts))]).astype(np.float32),
                                  faces=faces, color=_rgba("#44aa88", 0.9))
        self._gamma_pass.set_data(pos=np.array([[1.0, 0.0], [1.0, peak]], np.float32))
        self._range(self._gamma_view, (float(edges[0]), float(edges[-1])), (0.0, peak), (self._gamma_axis,))

    # ---- mouse ----------------------------------------------------------------------------------

    def _hit(self, pos):
        for n, view in enumerate(self._views):
            local = self.canvas.scene.node_transform(view).map(pos)[:2]
            if 0 <= local[0] < view.size[0] and 0 <= local[1] < view.size[1]:
                x, y = self.canvas.scene.node_transform(view.scene).map(pos)[:2]
                return n, float(x), float(y)
        return None

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
        hit = self._hit(event.pos)
        if hit is not None:
            event.handled = True
            self._on_page(hit[0], 1 if event.delta[1] > 0 else -1)
