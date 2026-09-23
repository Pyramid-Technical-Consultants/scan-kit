"""visPy scene: gantry orbit, axis guides, and the dose-volume ray march."""

from __future__ import annotations

import functools
import logging
import math

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
    WEIGHT_DOSE,
    WEIGHT_PROTONS,
    active_scale,
)
from .dose_volume_data import DepthAxis, SplatBatch, protons_from_mu
from .dose_volume_fill import (
    GAMMA_CMAP,
    MAX_CELLS,
    VOXEL_MM,
    GaussianSmearKernel,
    dose_field_weights,
    dose_grid,
    ink_rgb,
    typical_cell,
    typical_ray,
    zero_rgb,
)
from .dose_volume_physics import (
    GammaCriteria,
    csda_range_mm,
    depth_sigma_mm,
    dose_weights,
    effective_sigma_z,
    end_scatter_mm,
    layer_kernel,
    medium_for,
    through_wet,
)
from .dose_volume_raycast import (
    _alloc_texture,
    auto_color_range,
    fill_texture,
    gamma_texture,
    make_dose_box_node,
    manual_color_limits,
)
from .vispy_plot import bind_blender_view_keys

_log = logging.getLogger(__name__)


@functools.cache
def _late_line_class():
    from vispy import scene

    class LateLine(scene.visuals.Line):
        """Line the scene pass skips while ``held``, so it can draw after the march."""

        held = False

        def draw(self):
            if not self.held:
                super().draw()

    return LateLine


def _limits_moved(old_lo, old_hi, new: tuple[float, float]) -> bool:
    if old_lo is None or old_hi is None:
        return True
    for old, got in ((old_lo, new[0]), (old_hi, new[1])):
        scale = max(abs(old), abs(got), 1e-6)
        if abs(old - got) > 0.005 * scale:
            return True
    return False


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


def volume_corners(origin, extent_mm) -> np.ndarray:
    """Eight corners of the dose grid, in beam millimetres."""
    o = np.asarray(origin, dtype=float).reshape(3)
    h = o + np.asarray(extent_mm, dtype=float).reshape(3)
    return np.array(
        [[x, y, z] for x in (o[0], h[0]) for y in (o[1], h[1]) for z in (o[2], h[2])],
        dtype=float,
    )


def box_edge_segments(origin, extent_mm) -> np.ndarray:
    """Twelve box edges as 24 points, paired for ``connect="segments"``."""
    c = volume_corners(origin, extent_mm)
    # Corner index bits are (x, y, z) = (4, 2, 1); edges flip one bit.
    pairs = [(a, a | bit) for bit in (4, 2, 1) for a in range(8) if not a & bit]
    return np.array([c[i] for pair in pairs for i in pair], dtype=float)


PHANTOM_PAD_MM = 10.0
PHANTOM_RGBA = (0.30, 0.80, 0.90, 0.75)


def phantom_box(grid_origin, grid_extent_mm, depth_mm: float, pad_mm: float = PHANTOM_PAD_MM):
    """(origin, extent) of the phantom: surface to back face, a pad wider than the grid."""
    o = np.asarray(grid_origin, dtype=float).reshape(3).copy()
    e = np.asarray(grid_extent_mm, dtype=float).reshape(3).copy()
    o[:2] -= pad_mm
    e[:2] += 2.0 * pad_mm
    o[2], e[2] = -float(depth_mm), float(depth_mm)
    return o, e


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
    """MU (delivered where we have it) or protons, one value per spot. Dose starts from protons."""
    mu = dose_field_weights(batch.dose_mu, batch.weight)
    if weight_mode in (WEIGHT_PROTONS, WEIGHT_DOSE):
        return np.asarray(
            protons_from_mu(mu, batch.energy_mev, ic_gap_mm, batch.k_mu),
            dtype=np.float32,
        )
    return np.asarray(mu, dtype=np.float32)


def auto_depth_mm(medium, entry_energy, spread_pct: float, margin_sigma: float) -> float:
    """Phantom depth that holds every spot's range plus *margin_sigma* of range spread."""
    e = np.maximum(np.asarray(entry_energy, dtype=float), 1.0)
    if e.size == 0:
        return 0.0
    reach = csda_range_mm(medium, e) + float(margin_sigma) * depth_sigma_mm(medium, e, spread_pct)
    return float(np.max(reach))


def phantom_note(
    medium, depth_mm: float, wet_mm: float, *, auto: bool, n_nozzle: int, entry_energy,
) -> str:
    """One line on the phantom: its depth (auto or fixed) and which spots do not stay inside."""
    e = np.asarray(entry_energy, dtype=float)
    exits = 0.0
    if auto:
        head = f"Phantom auto {math.ceil(depth_mm):g} mm {medium.key}"
    else:
        head = f"Phantom {depth_mm:g} mm {medium.key}"
        if e.size:
            exits = float(np.mean(csda_range_mm(medium, np.maximum(e, 1.0)) > depth_mm))
    parts = [head]
    if wet_mm > 0.0:
        stopped = 1.0 - e.size / max(int(n_nozzle), 1)
        wet = f"{wet_mm:g} mm WET in front"
        parts.append(wet + (f", {100.0 * stopped:.0f} % of spots stop in it" if stopped > 0.0 else ""))
    if exits > 0.0:
        parts.append(f"{100.0 * exits:.0f} % of spots range out past the back")
    return "; ".join(parts)


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
        self._meas_tex = None
        self._plan_tex = None
        self._gamma_tex = None
        self._gamma = False
        self._gamma_cap = GammaCriteria().cap
        # (passed, evaluated) voxels of the last γ map.
        self.gamma_pass: tuple[int, int] | None = None
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
        self._bg: tuple[float, float, float] | None = None
        self._ink = ink_rgb((0.1, 0.1, 0.1))
        self._bounds = None
        self._late: list = []
        self._canvas.on_draw = self._on_draw

    @property
    def difference(self) -> bool:
        """True when the volume shows measured minus plan."""
        return self._difference

    @property
    def gamma(self) -> bool:
        """True when the volume shows the γ map."""
        return self._gamma

    def _scale_name(self) -> str:
        return GAMMA_CMAP if self._gamma else active_scale(self._difference, self._scale)

    def _set_gantry(self, degrees: float) -> None:
        xf = self._gantry.transform
        xf.reset()
        xf.rotate(float(degrees), (1.0, 0.0, 0.0))

    def clear_guides(self) -> None:
        for node in self._nodes:
            node.parent = None
        self._nodes.clear()
        self._late.clear()

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
            line = self._late_line(
                pos=np.vstack([starts[i], ends[i]]),
                color=color,
                width=2,
            )
            text = scene.Text(
                name,
                color=color,
                font_size=10,
                pos=tuple(label_pos[i]),
                parent=self._gantry,
            )
            self._nodes.extend((line, text))

    def _late_line(self, **kwargs):
        line = _late_line_class()(parent=self._gantry, **kwargs)
        # Line sets no GL state of its own and would inherit the march's "always".
        line.set_gl_state("translucent", depth_test=True, depth_func="lequal")
        self._late.append(line)
        return line

    def _add_bounds(self, origin, extent_mm) -> None:
        self._bounds = self._late_line(
            pos=box_edge_segments(origin, extent_mm),
            connect="segments",
            color=(*self._ink, 0.3),
            width=1,
        )
        self._nodes.append(self._bounds)

    def _set_background(self, rgb) -> None:
        """The view background is the scale's zero color, so empty space reads as 0."""
        if rgb == self._bg:
            return
        self._bg = rgb
        self._ink = ink_rgb(rgb)
        self._canvas.bgcolor = rgb
        if self._bounds is not None and self._bounds.parent is not None:
            self._bounds.set_data(color=(*self._ink, 0.3))

    def _ensure_textures(self, shape_zyx) -> None:
        key = tuple(int(v) for v in shape_zyx)
        if self._tex_shape == key and self._meas_tex is not None and self._plan_tex is not None:
            return
        self._canvas.set_current()
        self._tex_shape = key
        self._meas_tex = _alloc_texture(key)
        self._plan_tex = _alloc_texture(key)
        self._gamma_tex = _alloc_texture(key)
        self._box.set_volumes(self._meas_tex, self._plan_tex)

    def _color_limits(self) -> tuple[float, float]:
        if self._gamma:
            return 0.0, float(self._gamma_cap)
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
        self._set_background(zero_rgb(self._scale_name(), lo, hi))
        # γ is the worst voxel along each ray, on fixed limits.
        ray = RAY_MAXIMUM if self._gamma else self._ray_mode
        self._box.set_display(
            gain=self._gain,
            typical=self._typical,
            error_scale=self._error_scale,
            difference=self._difference,
            absolute=self._absolute,
            ray=_RAY_CODE.get(ray, 2.0),
            ray_scale=scale,
            scale_name=self._scale_name(),
            auto=self._auto and not self._gamma,
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
        voxel_mm: float = VOXEL_MM,
        smooth: bool = True,
        gamma: bool = False,
        gamma_criteria: GammaCriteria | None = None,
        phantom_mm: float = 0.0,
        entrance_wet_mm: float = 0.0,
        auto_margin_sigma: float = 5.0,
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
        self._box.set_smooth(smooth)
        self._bounds = None

        if status:
            text = scene.Text(
                status, color=self._ink, font_size=14, pos=(0.0, 0.0, 0.0), parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        batches = [b for b in (measured, plan) if b is not None and b.center.size]
        if not batches:
            text = scene.Text(
                "No dose samples for this mode",
                color=self._ink, font_size=14, pos=(0.0, 0.0, 0.0), parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        medium_key = getattr(axis, "medium", None)
        medium = medium_for(medium_key or "water")
        dose = weight_mode == WEIGHT_DOSE
        wet = max(float(entrance_wet_mm), 0.0) if medium_key else 0.0

        def entering(batch: SplatBatch):
            """(center, sigma, amount, energy) of the spots that reach the phantom surface."""
            # IC readings convert at the nozzle energy; entrance material acts after that.
            amounts = np.asarray(deposit_amounts(batch, weight_mode, ic_gap_mm), dtype=float)
            center = np.array(batch.center, dtype=float)
            sigma = np.array(batch.sigma, dtype=float)
            e = np.asarray(batch.energy_mev, dtype=float)
            if wet > 0.0:
                e, kept = through_wet(e, wet)
                keep = kept > 0.0
                center, sigma, e = center[keep], sigma[keep], e[keep]
                amounts = (amounts * kept)[keep]
                center[:, 2] = axis.z_scene_mm(e)
                sigma[:, 2] = axis.sigma_z_scene_mm(e, smear)
            return center, sigma, amounts, e

        meas_in = entering(measured) if measured is not None and measured.center.size else None
        plan_in = entering(plan) if plan is not None and plan.center.size else None
        live = [s for s in (meas_in, plan_in) if s is not None]
        if dose:
            kernel = layer_kernel(medium.key, np.concatenate([s[3] for s in live]), smear)
        else:
            kernel = GaussianSmearKernel(axis, smear)

        def prepared(spots):
            """Spots with deposit sigmas and weights, plus the (center, sigma) box the grid holds."""
            center, sigma, amounts, e = spots
            if dose:
                c, s = kernel.span(center, sigma, e)
                return center, sigma, dose_weights(kernel, amounts, e, medium), e, c, s
            if medium_key:
                # Stops land wider than the IC saw the beam: scatter over the full range.
                sigma = sigma.copy()
                sigma[:, 0:2] = np.hypot(sigma[:, 0:2], end_scatter_mm(medium, e)[:, None])
            return center, sigma, amounts, e, center, sigma

        meas = prepared(meas_in) if meas_in is not None else None
        plan_prep = prepared(plan_in) if plan_in is not None else None
        # The grid holds the plan either way, so switching Show keeps the frame.
        spans = [p for p in (meas, plan_prep) if p is not None]
        planned = plan_prep if difference or gamma else None
        entry_energy = np.concatenate([s[3] for s in live])
        depth = None
        if medium_key:
            depth = float(phantom_mm) if phantom_mm > 0.0 else auto_depth_mm(
                medium, entry_energy, smear, auto_margin_sigma,
            )
        grid = dose_grid(
            np.vstack([p[4] for p in spans]), np.vstack([p[5] for p in spans]), voxel_mm,
            z_floor=-depth if depth else None,
        )
        nx, ny, nz = grid.shape
        self.volume_note = f"Grid {nx} × {ny} × {nz} at {grid.voxel:g} mm"
        if grid.cropped:
            self.volume_note += f" (cropped to {MAX_CELLS} cells)"
        if medium_key:
            self.volume_note += "\n" + phantom_note(
                medium, depth or 0.0, wet,
                auto=phantom_mm <= 0.0,
                n_nozzle=sum(len(b.energy_mev) for b in batches),
                entry_energy=entry_energy,
            )
        self._ensure_textures((nz, ny, nx))
        empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        gpu = True
        peak_w, peak_s, peak_e = np.zeros(0), np.zeros((0, 3)), np.zeros(0)
        for tex, prep in ((self._meas_tex, meas), (self._plan_tex, planned)):
            if prep is None:
                fill_texture(self._canvas, tex, *empty, kernel, grid)
                continue
            center, sigma, amounts, e = prep[:4]
            gpu = fill_texture(self._canvas, tex, center, sigma, amounts, e, kernel, grid) and gpu
            peak_w, peak_s, peak_e = amounts, np.array(sigma, dtype=float), e
        if planned is None:
            self._difference = False
        if not gpu:
            self.volume_note += "\nCPU fill"
        if dose and peak_w.size:
            peak_s[:, 2] = effective_sigma_z(kernel, peak_e)
        self._typical = typical_cell(peak_w, peak_s)
        self.dose_peak = self._typical
        self.ray_peak = typical_ray(peak_w, peak_s)

        self._gamma = bool(gamma) and meas is not None and planned is not None
        self.gamma_pass = None
        if self._gamma:
            crit = gamma_criteria or GammaCriteria()
            self._gamma_cap = crit.cap
            self.gamma_pass = gamma_texture(
                self._canvas, self._meas_tex, self._plan_tex, self._gamma_tex, grid, crit,
            )
            self._difference = False
        self._box.set_volumes(self._gamma_tex if self._gamma else self._meas_tex, self._plan_tex)
        self._box.set_box(grid.origin, grid.shape, grid.voxel)
        self._scale = active_scale(self._difference, self._scale)
        self._push_display()
        self._has_volume = True

        o = np.asarray(grid.origin, dtype=float).reshape(3)
        hi_g = o + grid.extent_mm
        corners = volume_corners(grid.origin, grid.extent_mm)
        self._add_bounds(grid.origin, grid.extent_mm)
        if depth:
            p_o, p_e = phantom_box(grid.origin, grid.extent_mm, depth)
            self._nodes.append(self._late_line(
                pos=box_edge_segments(p_o, p_e), connect="segments", color=PHANTOM_RGBA, width=1.5,
            ))
            corners = np.vstack([corners, volume_corners(p_o, p_e)])
        pts = np.vstack([np.vstack([p[4] for p in spans]), o, hi_g])
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = np.vstack([lo, hi])
        self._add_axes(axis, extent)
        # Frame the grid and phantom through the same transform that draws them.
        # Axis labels are not part of this box, or they pull the volume off center.
        world = self._gantry.transform.map(corners)
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

    def reframe(self) -> None:
        """Fit the camera to the next render even if the gantry has not moved."""
        self._framed = False

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

    def set_smooth(self, smooth: bool) -> None:
        self._box.set_smooth(smooth)
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

        marching = self._has_volume and not self._broken
        for line in self._late:
            line.held = marching
        SceneCanvas.on_draw(self._canvas, event)
        if not marching:
            return
        try:
            span = self._box.draw_volume(self._canvas)
        except Exception:
            if not self._broken:
                _log.exception("Dose volume ray march failed")
            self._broken = True
            self._canvas.update()
            return
        # After the march, so each line depth tests against where the dose sits.
        for line in self._late:
            line.held = False
            line.draw()
        self._consume_auto_span(span)

    def _consume_auto_span(self, span) -> None:
        if not self._auto or self._gamma or span is None:
            return
        got = auto_color_range(self._difference, span[0], span[1])
        if got is None or not _limits_moved(self.auto_lo, self.auto_hi, got):
            return
        self.auto_lo, self.auto_hi = got
        self._push_display()
        listener = self.range_listener
        if listener is not None:
            listener()
        self._request_redraw()

    def _request_redraw(self) -> None:
        # A repaint requested inside paintGL can be coalesced away, which left
        # the last drag frame on a stale range until the next click.
        try:
            from PySide6.QtCore import QTimer
        except ImportError:
            self._canvas.update()
            return
        QTimer.singleShot(0, self._canvas.update)
