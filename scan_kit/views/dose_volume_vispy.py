"""visPy scene: gantry orbit, axis guides, and the dose-volume ray march."""

from __future__ import annotations

import logging

import numpy as np

from .dose_volume_catalog import (
    AGREE_TRANSPARENT,
    DEFAULT_ERROR_SCALE,
    DEFAULT_GAIN,
    DEFAULT_RAY,
    DEFAULT_SCALE,
    ERROR_ABSOLUTE,
    RAY_INTEGRAL,
    RAY_MAXIMUM,
    RAY_TRANSPARENT,
    WEIGHT_PROTONS,
    active_scale,
)
from .dose_volume_data import DepthAxis, SplatBatch, protons_from_mu
from .dose_volume_fill import (
    GaussianSmearKernel,
    dose_field_weights,
    dose_grid,
    typical_cell,
    typical_ray,
)
from .dose_volume_raycast import (
    _alloc_texture,
    auto_color_range,
    fill_texture,
    make_dose_box_node,
    manual_color_limits,
)
from .vispy_plot import FG, bind_blender_view_keys

_log = logging.getLogger(__name__)


def _limits_moved(old_lo, old_hi, new: tuple[float, float]) -> bool:
    if old_lo is None or old_hi is None:
        return True
    for old, got in ((old_lo, new[0]), (old_hi, new[1])):
        scale = max(abs(old), abs(got), 1e-6)
        if abs(old - got) > 0.005 * scale:
            return True
    return False


def default_session_colors(n: int) -> list[str]:
    from ..common.plotting import DEFAULT_SESSION_COLORS

    if n <= 0:
        return []
    return [DEFAULT_SESSION_COLORS[i % len(DEFAULT_SESSION_COLORS)] for i in range(n)]


def gantry_rx_matrix(degrees: float) -> np.ndarray:
    """Rotation about +X (degrees). 90° maps (x, y, z) → (x, −z, y)."""
    th = np.deg2rad(float(degrees))
    c = float(np.cos(th))
    s = float(np.sin(th))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def apply_gantry(points, degrees: float) -> np.ndarray:
    """Apply :func:`gantry_rx_matrix` to ``(N, 3)`` scene points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return pts @ gantry_rx_matrix(degrees).T


def axis_guide_points(extent, *, depth_sign: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis start / tip / label positions past the data AABB.

    X and Y point +axis. Z points toward increasing depth (high energy):
    ``depth_sign=-1`` sends that guide down so it matches ``z = −R``.
    """
    lo = np.asarray(extent[0], dtype=np.float64).reshape(3)
    hi = np.asarray(extent[1], dtype=np.float64).reshape(3)
    span = np.maximum(hi - lo, 1.0)
    origin = (lo + hi) / 2.0
    starts = np.tile(origin, (3, 1))
    ends = np.tile(origin, (3, 1))
    labels = np.tile(origin, (3, 1))
    for i in (0, 1):
        starts[i, i] = origin[i] - 0.10 * span[i]
        ends[i, i] = hi[i] + 0.35 * span[i]
        labels[i, i] = ends[i, i] + 0.22 * span[i]
    z_dir = 1.0 if depth_sign >= 0 else -1.0
    z_tip = hi[2] if z_dir > 0 else lo[2]
    starts[2, 2] = origin[2] - z_dir * 0.10 * span[2]
    ends[2, 2] = z_tip + z_dir * 0.35 * span[2]
    labels[2, 2] = ends[2, 2] + z_dir * 0.22 * span[2]
    return starts, ends, labels


def volume_corners(origin, shape) -> np.ndarray:
    """Eight corners of the dose grid, in beam millimetres."""
    o = np.asarray(origin, dtype=float).reshape(3)
    h = o + np.asarray(shape, dtype=float).reshape(3)
    return np.array(
        [[x, y, z] for x in (o[0], h[0]) for y in (o[1], h[1]) for z in (o[2], h[2])],
        dtype=float,
    )


def keep_camera(framed: bool, framed_gantry: float | None, gantry_deg: float) -> bool:
    """Leave orbit, zoom, and pan alone when the gantry has not moved.

    Switching IC1 / plan / ISO ray rebuilds the volume in the same beam
    frame, so the camera the user already set stays put.
    """
    if not framed or framed_gantry is None:
        return False
    return abs(float(framed_gantry) - float(gantry_deg)) < 1e-3


def volume_frame(world_corners) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AABB of the gantry-mapped grid. The center is the grid center."""
    world = np.asarray(world_corners, dtype=float)
    world = world.reshape(-1, world.shape[-1])[:, :3]
    lo = world.min(axis=0)
    hi = world.max(axis=0)
    return lo, hi, (lo + hi) / 2.0


_RAY_CODE = {RAY_INTEGRAL: 0.0, RAY_MAXIMUM: 1.0, RAY_TRANSPARENT: 2.0}


def _gesture_key(event) -> tuple[frozenset, frozenset]:
    mouse = getattr(event, "mouse_event", None)
    mods = () if mouse is None else (mouse.modifiers or ())
    return frozenset(event.buttons), frozenset(mods)


def make_dose_camera():
    """Turntable whose field-of-view drag starts from the scalar fov.

    vispy keeps the last gesture in ``_event_value``. A zoom stores a
    2-tuple and a shift-pan stores the center. The next shift-right move
    subtracts a numpy pixel delta from that sequence, and ``max(0, fov)``
    then raises because the result is an array.
    """
    from vispy import scene

    class DoseTurntable(scene.cameras.TurntableCamera):
        _gesture = None

        def viewbox_mouse_event(self, event):
            if event.type == "mouse_press":
                self._event_value = None
                self._gesture = None
            elif event.type == "mouse_move" and event.press_event is not None:
                gesture = _gesture_key(event)
                if gesture != self._gesture:
                    self._event_value = None
                    self._gesture = gesture
            elif event.type == "mouse_release":
                self._gesture = None
            return super().viewbox_mouse_event(event)

    return DoseTurntable(
        fov=45,
        distance=400,
        center=(0.0, 0.0, 0.0),
        elevation=25,
    )


def deposit_amounts(batch: SplatBatch, weight_mode: str, ic_gap_mm: float) -> np.ndarray:
    """MU (delivered where we have it) or protons, one value per spot."""
    mu = dose_field_weights(batch.dose_mu, batch.weight)
    if weight_mode == WEIGHT_PROTONS:
        return np.asarray(
            protons_from_mu(mu, batch.energy_mev, ic_gap_mm, batch.k_mu),
            dtype=np.float32,
        )
    return np.asarray(mu, dtype=np.float32)


class DoseScene:
    """Turntable view of a measured dose volume, optionally minus the plan."""

    def __init__(self, canvas) -> None:
        from vispy import scene

        self._canvas = canvas
        self._view = canvas.central_widget.add_view()
        self._view.camera = make_dose_camera()
        bind_blender_view_keys(canvas, lambda: self._view.camera)
        self._frame_bounds = None
        self._framed = False
        self._framed_gantry: float | None = None
        self._pending_gantry = 0.0
        canvas.events.resize.connect(self._on_canvas_resize)
        self._gantry = scene.Node(parent=self._view.scene)
        self._gantry.transform = scene.transforms.MatrixTransform()
        self._nodes: list = []
        self._box = make_dose_box_node()(parent=self._gantry)
        self._box.visible = False
        self._meas_tex = None
        self._plan_tex = None
        self._tex_shape: tuple[int, int, int] | None = None
        self._has_volume = False
        self._broken = False
        self._gain = DEFAULT_GAIN
        self._typical = 1.0
        self.dose_peak = 1.0
        self.ray_peak = 1.0
        self._ray_mode = DEFAULT_RAY
        self._scale = DEFAULT_SCALE
        self._error_scale = DEFAULT_ERROR_SCALE
        self._absolute = False
        self._difference = False
        self._auto = True
        self.auto_lo: float | None = None
        self.auto_hi: float | None = None
        self.range_listener = None
        self.volume_note = ""
        self._canvas.on_draw = self._on_draw

    def _set_gantry(self, degrees: float) -> None:
        xf = self._gantry.transform
        xf.reset()
        xf.rotate(float(degrees), (1.0, 0.0, 0.0))

    def clear_guides(self) -> None:
        for node in self._nodes:
            node.parent = None
        self._nodes.clear()

    def _add_axes(self, axis: DepthAxis, extent: np.ndarray) -> None:
        from vispy import scene

        starts, ends, label_pos = axis_guide_points(
            extent, depth_sign=int(getattr(axis, "depth_sign", 1)),
        )
        colors = (
            (0.85, 0.35, 0.35, 1.0),
            (0.35, 0.75, 0.40, 1.0),
            (0.40, 0.55, 0.95, 1.0),
        )
        names = ("X (mm)", "Y (mm)", axis.axis_label)
        for i, (color, name) in enumerate(zip(colors, names)):
            line = scene.visuals.Line(
                pos=np.vstack([starts[i], ends[i]]),
                color=color,
                width=2,
                parent=self._gantry,
            )
            text = scene.Text(
                name,
                color=color,
                font_size=10,
                pos=tuple(label_pos[i]),
                parent=self._gantry,
            )
            self._nodes.extend((line, text))

    def _ensure_textures(self, shape_zyx) -> None:
        key = tuple(int(v) for v in shape_zyx)
        if self._tex_shape == key and self._meas_tex is not None and self._plan_tex is not None:
            return
        self._canvas.set_current()
        self._tex_shape = key
        self._meas_tex = _alloc_texture(key)
        self._plan_tex = _alloc_texture(key)
        self._box.set_volumes(self._meas_tex, self._plan_tex)

    def _color_limits(self) -> tuple[float, float]:
        if (
            self._auto
            and self.auto_lo is not None
            and self.auto_hi is not None
            and np.isfinite(self.auto_lo)
            and np.isfinite(self.auto_hi)
        ):
            return float(self.auto_lo), float(self.auto_hi)
        integral = self._ray_mode == RAY_INTEGRAL
        return manual_color_limits(
            difference=self._difference,
            transparent=self._ray_mode == RAY_TRANSPARENT,
            integral=integral,
            gain=self._gain,
            typical=self._typical,
            ray_scale=self.ray_peak if integral else self.dose_peak,
            error_scale=self._error_scale,
            absolute=self._absolute,
        )

    def _push_display(self) -> None:
        scale = self.ray_peak if self._ray_mode == RAY_INTEGRAL else self.dose_peak
        lo, hi = self._color_limits()
        self._box.set_display(
            gain=self._gain,
            typical=self._typical,
            error_scale=self._error_scale,
            difference=self._difference,
            absolute=self._absolute,
            ray=_RAY_CODE.get(self._ray_mode, 2.0),
            ray_scale=scale,
            scale_name=active_scale(self._difference, self._scale),
            auto=self._auto,
            lo=lo,
            hi=hi,
        )

    def render(
        self,
        measured: SplatBatch | None,
        plan: SplatBatch | None,
        axis: DepthAxis,
        *,
        gain: float,
        gantry_deg: float = 90.0,
        difference: bool = False,
        agreement: str = AGREE_TRANSPARENT,
        error_mode: str = ERROR_ABSOLUTE,
        error_scale: float = DEFAULT_ERROR_SCALE,
        weight_mode: str = "mu",
        ic_gap_mm: float = 10.0,
        smear: float = 0.5,
        ray_mode: str = DEFAULT_RAY,
        scale: str = DEFAULT_SCALE,
        auto_scale: bool = True,
        status: str | None = None,
    ) -> None:
        from vispy import scene

        self.clear_guides()
        self._set_gantry(gantry_deg)
        self._has_volume = False
        self._broken = False
        self.volume_note = ""
        self._gain = float(gain)
        self._error_scale = float(error_scale)
        self._absolute = error_mode == ERROR_ABSOLUTE
        self._difference = bool(difference)
        self._ray_mode = ray_mode if ray_mode in _RAY_CODE else DEFAULT_RAY
        self._scale = active_scale(self._difference, scale)
        self._auto = bool(auto_scale)
        self.auto_lo = None
        self.auto_hi = None

        if status:
            text = scene.Text(
                status, color=FG, font_size=14, pos=(0.0, 0.0, 0.0), parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        parts = []
        if measured is not None and measured.center.size:
            parts.append(measured.center)
        if plan is not None and plan.center.size:
            parts.append(plan.center)
        if not parts:
            text = scene.Text(
                "No dose samples for this mode",
                color=FG, font_size=14, pos=(0.0, 0.0, 0.0), parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        sig_parts = []
        if measured is not None and measured.center.size:
            sig_parts.append(measured.sigma)
        if plan is not None and plan.center.size:
            sig_parts.append(plan.sigma)
        grid = dose_grid(np.vstack(parts), np.vstack(sig_parts))
        nx, ny, nz = grid.shape
        self.volume_note = f"Grid {nx} × {ny} × {nz} mm"
        if grid.cropped:
            self.volume_note += " (cropped to 512)"
        kernel = GaussianSmearKernel(axis, smear)
        self._ensure_textures((nz, ny, nx))
        gpu = True
        if measured is not None and measured.center.size:
            amounts = deposit_amounts(measured, weight_mode, ic_gap_mm)
            gpu = fill_texture(
                self._canvas, self._meas_tex,
                measured.center, measured.sigma, amounts, measured.energy_mev,
                kernel, grid,
            ) and gpu
            peak_w, peak_s = amounts, measured.sigma
        else:
            fill_texture(
                self._canvas, self._meas_tex,
                np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                kernel, grid,
            )
            peak_w, peak_s = np.zeros(0), np.zeros((0, 3))
        if difference and plan is not None and plan.center.size:
            amounts = deposit_amounts(plan, weight_mode, ic_gap_mm)
            gpu = fill_texture(
                self._canvas, self._plan_tex,
                plan.center, plan.sigma, amounts, plan.energy_mev,
                kernel, grid,
            ) and gpu
            peak_w, peak_s = amounts, plan.sigma
        else:
            self._difference = False
            fill_texture(
                self._canvas, self._plan_tex,
                np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                kernel, grid,
            )
        if not gpu:
            self.volume_note += "\nCPU fill"
        self._typical = typical_cell(peak_w, peak_s)
        self.dose_peak = self._typical
        self.ray_peak = typical_ray(peak_w, peak_s)
        self._box.set_box(grid.origin, grid.shape)
        self._scale = active_scale(self._difference, self._scale)
        self._push_display()
        self._has_volume = True

        o = np.asarray(grid.origin, dtype=float).reshape(3)
        hi_g = o + np.asarray(grid.shape, dtype=float)
        pts = np.vstack([np.vstack(parts), o, hi_g])
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = np.vstack([lo, hi])
        self._add_axes(axis, extent)
        # Frame the grid through the same transform that draws it. Axis
        # labels are not part of this box, or they pull the volume off center.
        world = self._gantry.transform.map(volume_corners(grid.origin, grid.shape))
        flo, fhi, center = volume_frame(world)
        pad = np.maximum((fhi - flo) * 0.12, 1.0)
        self._frame_bounds = (flo, fhi, pad, center)
        self._pending_gantry = float(gantry_deg)
        if not keep_camera(self._framed, self._framed_gantry, gantry_deg):
            self._apply_frame()
        self._canvas.update()

    def _apply_frame(self) -> None:
        if self._frame_bounds is None:
            return
        w, h = self._view.size
        if w < 2 or h < 2:
            return
        flo, fhi, pad, center = self._frame_bounds
        self._view.camera.set_range(
            x=(flo[0] - pad[0], fhi[0] + pad[0]),
            y=(flo[1] - pad[1], fhi[1] + pad[1]),
            z=(flo[2] - pad[2], fhi[2] + pad[2]),
            margin=0.0,
        )
        # set_range keeps a center that was set before the first call.
        self._view.camera.center = tuple(float(v) for v in center)
        self._framed = True
        self._framed_gantry = self._pending_gantry

    def _on_canvas_resize(self, _event=None) -> None:
        if not self._framed:
            self._apply_frame()

    def set_ray(self, mode: str) -> None:
        self._ray_mode = mode if mode in _RAY_CODE else DEFAULT_RAY
        self.auto_lo = None
        self.auto_hi = None
        self._push_display()
        self._canvas.update()

    def set_auto(self, auto: bool) -> None:
        self._auto = bool(auto)
        if not self._auto:
            self.auto_lo = None
            self.auto_hi = None
        self._push_display()
        self._canvas.update()

    def set_scale(self, name: str) -> None:
        self._scale = active_scale(self._difference, name)
        self._push_display()
        self._canvas.update()

    def set_gain(self, gain: float) -> None:
        self._gain = float(gain)
        self._push_display()
        self._canvas.update()

    def set_error_metric(self, mode: str, scale: float) -> None:
        self._absolute = mode == ERROR_ABSOLUTE
        self._error_scale = float(scale)
        self._push_display()
        self._canvas.update()

    def _on_draw(self, event) -> None:
        from vispy.scene import SceneCanvas

        self._box.visible = False
        SceneCanvas.on_draw(self._canvas, event)
        if not self._has_volume or self._broken:
            return
        try:
            span = self._box.draw_volume(self._canvas)
        except Exception:
            if not self._broken:
                _log.exception("Dose volume ray march failed")
            self._broken = True
            self._canvas.update()
            return
        self._consume_auto_span(span)

    def _consume_auto_span(self, span) -> None:
        if not self._auto or span is None:
            return
        got = auto_color_range(self._difference, span[0], span[1])
        if got is None or not _limits_moved(self.auto_lo, self.auto_hi, got):
            return
        self.auto_lo, self.auto_hi = got
        self._push_display()
        listener = self.range_listener
        if listener is not None:
            listener()
        self._canvas.update()
