"""visPy scene: gantry orbit, axis guides, and the dose-volume ray march."""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from .dose_volume_catalog import (
    DEFAULT_ERROR_SCALE,
    DEFAULT_GAIN,
    DEFAULT_MC_HISTORIES,
    DEFAULT_RAY,
    field_edge_spec,
    DEFAULT_SCALE,
    ERROR_ABSOLUTE,
    MC_MEDIA,
    MC_SEED,
    MODEL_ANALYTIC,
    MODEL_MC,
    RAY_INTEGRAL,
    RAY_MAXIMUM,
    RAY_TRANSPARENT,
    WEIGHT_DOSE,
    WEIGHT_MU,
    WEIGHT_PROTONS,
    active_scale,
)
from .dose_mc import McRun, mc_note
from .dose_volume_data import DepthAxis, SplatBatch, protons_from_mu
from .dose_volume_fill import (
    GAMMA_CMAP,
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
    read_texture,
    manual_color_limits,
)
from .vispy_plot import bind_blender_view_keys

_log = logging.getLogger(__name__)

MC_PREVIEW_S = 0.25  # refining Monte Carlo redraws this often
# Transport slice between frames: short while the view is being moved, long (full GPU) otherwise.
MC_SLICE_S, MC_IDLE_SLICE_S = 0.012, 0.05


@functools.cache
def _late_class(name: str = "Line"):
    from vispy import scene

    class Late(getattr(scene.visuals, name)):
        """Visual the scene pass skips while ``held``, so it can draw after the march."""

        held = False

        def draw(self):
            if not self.held:
                super().draw()

    Late.__name__ = f"Late{name}"
    return Late


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


TICK_MM = 10.0
# Numbered ticks are every 1, 2, 5, … cm, whichever keeps an edge to MAX_AXIS_LABELS numbers.
_LABEL_EVERY = (1, 2, 5, 10, 20, 50, 100)
MAX_AXIS_LABELS = 8


def axis_ticks(lo: float, hi: float, step: float = TICK_MM) -> tuple[np.ndarray, np.ndarray]:
    """Tick positions every *step* mm inside [lo, hi], and which of them carry a number.

    The end ticks always carry one so the extent reads; round ones crowding an end drop out.
    """
    k = np.arange(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)
    every = next((n for n in _LABEL_EVERY if k.size <= n * MAX_AXIS_LABELS), _LABEL_EVERY[-1])
    labeled = k % every == 0
    if k.size:
        labeled &= np.minimum(k - k[0], k[-1] - k) >= 0.6 * every
        labeled[[0, -1]] = True
    vals = k * step
    # The box corner is the extent, even when it falls between centimeter ticks.
    if vals.size == 0 or float(vals[0] - lo) > 1e-6:
        vals = np.r_[lo, vals]
        labeled = np.r_[True, labeled]
    if vals.size == 0 or float(hi - vals[-1]) > 1e-6:
        vals = np.r_[vals, hi]
        labeled = np.r_[labeled, True]
    return vals, labeled


@dataclass(frozen=True)
class AxisText:
    """One edge's labels: numbers hang off the tick tips, the title off the midpoint."""

    axis: int  # 0, 1, 2 for the X, Y, Z edge
    tips: np.ndarray  # (n, 3) tips of the numbered ticks, along the default outward
    numbers: list[str]
    mid: np.ndarray  # tick-length out from the edge midpoint
    out: np.ndarray  # default unit vector away from the box
    title: str
    length: float  # edge length, mm
    roots: np.ndarray  # (n, 3) every tick, on the edge
    major: np.ndarray  # which roots carry a number
    outs: np.ndarray  # (2, 3) the two directions that leave the box
    edge_mid: np.ndarray
    tick_mm: float


@dataclass(frozen=True)
class BoxAxes:
    """Tick marks and text for three box edges that meet at the shallow (−x, −y) corner."""

    ticks: np.ndarray  # segment pairs for ``connect="segments"``
    axes: list[AxisText]


def box_axes(origin, extent_mm, names, *, depth_sign: int = 1, step: float = TICK_MM) -> BoxAxes:
    """X and Y run along the surface face, Z down the corner edge; ticks point away from the box.

    Z numbers read as depth: ``depth_sign=-1`` shows ``z = −R`` as positive R.
    """
    o = np.asarray(origin, dtype=float).reshape(3)
    h = o + np.asarray(extent_mm, dtype=float).reshape(3)
    tick = float(np.clip(0.02 * float(np.max(h - o)), 1.5, 6.0))
    # Each edge offers the two directions that leave the box; the view picks one per frame.
    edges = (
        (np.array([0.0, o[1], h[2]]), (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])), 1),
        (np.array([o[0], 0.0, h[2]]), (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])), 1),
        (np.array([o[0], o[1], 0.0]), (np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])), depth_sign),
    )
    ticks, axes = [], []
    for i, ((base, outs, sign), name) in enumerate(zip(edges, names)):
        out = outs[0]
        vals, labeled = axis_ticks(o[i], h[i], step)
        roots, major, tips, numbers = [], [], [], []
        for v, is_major in zip(vals, labeled):
            p = base.copy()
            p[i] = v
            roots.append(p)
            major.append(bool(is_major))
            ticks += [p, p + out * tick * (1.0 if is_major else 0.55)]
            if is_major:
                tips.append(p + out * tick)
                numbers.append(f"{v * sign + 0.0:g}")
        edge_mid = base.copy()
        edge_mid[i] = 0.5 * (o[i] + h[i])
        axes.append(AxisText(
            i, np.array(tips, dtype=float).reshape(-1, 3), numbers, edge_mid + out * tick, out, name,
            float(h[i] - o[i]), np.array(roots, dtype=float).reshape(-1, 3), np.array(major, dtype=bool),
            np.stack(outs), edge_mid, tick,
        ))
    return BoxAxes(np.array(ticks, dtype=float).reshape(-1, 3), axes)


NUMBER_PT = 7
TITLE_PT = 9
LABEL_GAP_PX = 14.0


def outward_index(edge, candidates, away) -> int:
    """Which candidate points off the edge, on the side *away* from the box.

    *edge*, *candidates* ``(2, 2)`` and *away* are screen pixels. A candidate that
    runs along the edge scores nothing, so a side view takes the other one.
    """
    edge = np.asarray(edge, dtype=float)
    cands = np.asarray(candidates, dtype=float).reshape(2, 2)
    away = np.asarray(away, dtype=float)
    n = float(np.hypot(*edge))
    if n < 1e-6:
        return int(np.argmax(cands @ away))
    perp = np.array([-edge[1], edge[0]]) / n
    if float(perp @ away) < 0.0:
        perp = -perp
    return int(np.argmax(cands @ perp))


def label_direction(edge, out) -> np.ndarray:
    """Screen unit vector perpendicular to *edge*, on the side *out* points to.

    Falls back to *out* when the edge is seen end-on.
    """
    edge, out = np.asarray(edge, dtype=float), np.asarray(out, dtype=float)
    n = float(np.hypot(*edge))
    d = np.array([-edge[1], edge[0]]) / n if n > 1e-6 else out
    if n > 1e-6 and float(d @ out) < 0.0:
        d = -d
    return d / max(float(np.hypot(*d)), 1e-9)


def spaced_labels(tips_px, need_px: float) -> list[int]:
    """Indices of numbers that clear their neighbors. Both ends always stay."""
    tips = np.asarray(tips_px, dtype=float).reshape(-1, 2)
    n = len(tips)
    if n < 2:
        return list(range(n))
    keep = [0]
    for i in range(1, n - 1):
        if float(np.hypot(*(tips[i] - tips[keep[-1]]))) >= need_px:
            keep.append(i)
    while len(keep) > 1 and float(np.hypot(*(tips[-1] - tips[keep[-1]]))) < need_px:
        keep.pop()
    keep.append(n - 1)
    return keep


def readable_angle(edge) -> float:
    """Clockwise degrees of a screen edge (y down), folded into [-90, 90].

    vispy text rotation is clockwise. Folding keeps the baseline along the
    axis with the top of the letters toward the top of the view.
    """
    deg = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    if deg > 90.0:
        deg -= 180.0
    elif deg < -90.0:
        deg += 180.0
    return deg


def label_anchors(d) -> tuple[str, str]:
    """Text anchor for a label hanging off along screen direction *d* (view pixels, y down)."""
    ax = "right" if d[0] < -0.38 else "left" if d[0] > 0.38 else "center"
    ay = "top" if d[1] > 0.38 else "bottom" if d[1] < -0.38 else "center"
    return ax, ay


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


# Wider than the axis titles, which sit about 36 mm off the voxel box.
PHANTOM_PAD_MM = 48.0
# ICRU 78 takes the geometric field edge as the 50 % isodose.
FIELD_FRACTION = 0.5
# Box edges and ticks share one muted ink; text needs a little more to stay legible.
BOX_ALPHA = 0.3
NUMBER_ALPHA = 0.6
TITLE_ALPHA = 0.75
# Auto difference never spans less than ±1 % of the dose scale.
AUTO_DIFF_FLOOR = 0.01
PHANTOM_RGBA = (0.30, 0.80, 0.90, 0.35)
FIELD_RGBA = (0.95, 0.85, 0.45, 0.55)


def field_box(
    volume_zyx, origin, voxel: float, fraction: float = FIELD_FRACTION, *, per_slice: bool = False,
):
    """Axis-aligned box of voxels at or above *fraction* of the reference dose.

    The reference is the volume peak. With *per_slice*, a depth counts only when
    its own maximum reaches that fraction of the peak, and the lateral edge
    inside the slice is the same fraction of the slice maximum. Returns
    ``(origin, extent)`` in beam millimetres, or None when nothing qualifies.
    """
    vol = np.asarray(volume_zyx, dtype=float)
    if vol.size == 0:
        return None
    if per_slice:
        peak = np.max(vol, axis=(1, 2))
        global_peak = float(np.max(peak)) if peak.size else 0.0
        if not math.isfinite(global_peak) or global_peak <= 0.0:
            return None
        kept = peak >= fraction * global_peak
        mask = vol >= (fraction * peak)[:, None, None]
        mask &= kept[:, None, None]
        zz, yy, xx = np.nonzero(mask)
    else:
        peak_v = float(np.max(vol))
        if not math.isfinite(peak_v) or peak_v <= 0.0:
            return None
        zz, yy, xx = np.nonzero(vol >= fraction * peak_v)
    if xx.size == 0:
        return None
    v = float(voxel)
    o = np.asarray(origin, dtype=float).reshape(3)
    lo = o + np.array([xx.min(), yy.min(), zz.min()], dtype=float) * v
    hi = o + np.array([xx.max() + 1, yy.max() + 1, zz.max() + 1], dtype=float) * v
    return lo, hi - lo


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

    # No fixed distance: vispy then backs off with the fitted scale, so a deep
    # volume's near end is not blown past the view by perspective.
    return DoseTurntable(
        fov=45,
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
        # Per texture slot: (inputs key less histories, McRun) it holds, refining or finished.
        # ponytail: a finished run keeps its GPU sums (4 floats a voxel) so more histories can resume it.
        self._mc: dict[str, tuple] = {}
        self._mc_preview_at = 0.0
        self._mc_turn = -1
        self._mc_redraw = False  # the next draw is a preview mc_step asked for
        self._moved_at = 0.0  # last draw something else asked for
        self._note_base = ""
        # What render left for the finished volume: (grid, want γ, criteria, field-box source, field edge spec, show).
        self._post: tuple | None = None
        self._want_gamma = False
        self._has_volume = False
        self._broken = False
        self._gain = DEFAULT_GAIN
        self._typical = 1.0
        self.dose_peak = 1.0
        self.ray_peak = 1.0
        self.field_extent: tuple[float, float, float] | None = None
        self.phantom_note = ""
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
        # (node, alpha) drawn in the background's ink; recolored when the background changes.
        self._inked: list = []
        # (2D Text, gantry-space anchors) reprojected every draw.
        self._labels: list = []
        self._tick_line = None
        self._axes_center = np.zeros(3)
        self._canvas.on_draw = self._on_draw

    @property
    def difference(self) -> bool:
        """True when the volume shows measured minus plan."""
        return self._difference

    @property
    def gamma(self) -> bool:
        """True when the volume shows the γ map."""
        return self._gamma

    @property
    def mc_refining(self) -> bool:
        """True while a Monte Carlo run is still filling the volume."""
        return any(not run.done for _, run in self._mc.values())

    @property
    def mc_progress(self) -> float | None:
        """Share of the refining Monte Carlo transported, or None when nothing is refining."""
        live = [run.progress for _, run in self._mc.values() if not run.done]
        return min(live) if live else None

    @property
    def gamma_pending(self) -> bool:
        """True when γ waits for a refining Monte Carlo run."""
        return self._want_gamma and self.mc_refining

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
        self._inked.clear()
        self._labels.clear()

    def _add_axes(self, axis: DepthAxis, origin, extent_mm) -> None:
        """Ticks every cm on the voxel box's edges, in the box's own muted ink."""
        self._axes_center = np.asarray(origin, dtype=float) + 0.5 * np.asarray(extent_mm, dtype=float)
        guides = box_axes(
            origin, extent_mm, ("X (mm)", "Y (mm)", axis.axis_label),
            depth_sign=int(getattr(axis, "depth_sign", 1)),
        )
        self._tick_line = None
        if guides.ticks.size:
            ticks = self._late_line(
                pos=guides.ticks, connect="segments", color=(*self._ink, BOX_ALPHA), width=1,
            )
            self._tick_line = ticks
            self._nodes.append(ticks)
            self._inked.append((ticks, BOX_ALPHA))
        # 3D Text shrinks by the perspective w, so labels are 2D text on the view, placed each draw.
        late_text = _late_class("Text")
        for ax in guides.axes:
            texts = []
            for strings, size, alpha in ((ax.numbers, NUMBER_PT, NUMBER_ALPHA), ([ax.title], TITLE_PT, TITLE_ALPHA)):
                text = late_text(
                    strings, color=(*self._ink, alpha), font_size=size,
                    pos=np.zeros((len(strings), 2)), parent=self._view,
                ) if strings else None
                if text is not None:
                    self._late.append(text)
                    self._nodes.append(text)
                    self._inked.append((text, alpha))
                texts.append(text)
            self._labels.append((*texts, ax))

    def _place_labels(self) -> None:
        """Hang each edge's numbers and title a fixed pixel gap off its projected outward side."""
        if not self._labels:
            return
        to_view = self._gantry.node_transform(self._view)
        px_per_pt = float(self._canvas.dpi) / 72.0

        def screen(points):
            q = to_view.map(np.atleast_2d(points))
            return q[:, :2] / q[:, 3:4]

        center = screen(self._axes_center)[0]
        em = NUMBER_PT * px_per_pt
        projected = []
        segments = []
        for _numbers, _title, ax in self._labels:
            along = np.zeros(3)
            along[ax.axis] = 1.0
            pts = screen(np.vstack([
                ax.edge_mid, ax.edge_mid + ax.outs[0], ax.edge_mid + ax.outs[1], ax.edge_mid + along,
            ]))
            edge = pts[3] - pts[0]
            per_mm = float(np.hypot(*edge))
            away = pts[0] - center
            idx = outward_index(edge, pts[1:3] - pts[0], away if np.hypot(*away) > 1e-6 else pts[1] - pts[0])
            choice = ax.outs[idx]
            span = ax.tick_mm * np.where(ax.major, 1.0, 0.55)[:, None]
            ends = ax.roots + choice * span
            segments += [p for pair in zip(ax.roots, ends) for p in pair]
            projected.append((pts[0], pts[1 + idx] - pts[0], edge, per_mm, ends[ax.major]))
        if self._tick_line is not None and segments:
            self._tick_line.set_data(pos=np.asarray(segments, dtype=float))
        widest = max(p[3] for p in projected)
        for (numbers, title, ax), (a, side, edge, per_mm, tips3) in zip(self._labels, projected):
            # Edge seen (nearly) end-on: its labels would pile onto the corner.
            shown = per_mm * ax.length >= 3.0 * em and per_mm >= 0.3 * widest
            title.visible = shown
            if numbers is not None:
                numbers.visible = shown
            if not shown:
                continue
            # Outward toward the camera picks no side; step away from the box instead.
            d = label_direction(edge, side if np.hypot(*side) >= 0.25 * per_mm else a - center)
            angle = readable_angle(edge)
            # Rotated glyphs spin about their center, so the center sits a glyph-height off the edge.
            gap = LABEL_GAP_PX + 0.8 * em
            anchors = ("center", "center")
            if numbers is not None:
                wide = 0.6 * em * max(len(s) for s in ax.numbers)
                tips = screen(tips3)
                keep = spaced_labels(tips, wide + 4.0)
                strings = [ax.numbers[i] for i in keep]
                numbers.visible = bool(keep)
                if strings and strings != numbers.text:
                    numbers.text = strings
                if keep:
                    numbers.pos = tips[keep] + d * gap
                    if numbers.anchors != anchors:
                        numbers.anchors = anchors
                    if abs(float(np.reshape(numbers.rotation, -1)[0]) - angle) > 0.5:
                        numbers.rotation = angle
            # Numbers are centered `gap` past the tick tip. The title clears their
            # outer edge by another gap, including however far the tick projects.
            tick_px = abs(float(np.dot(side, d))) * ax.tick_mm
            title_em = TITLE_PT * px_per_pt
            pos = screen(ax.edge_mid)[0] + d * (tick_px + gap + em + LABEL_GAP_PX + 0.6 * title_em)
            pos[0] = min(max(pos[0], 4.0), max(float(self._view.size[0]) - 4.0, 4.0))
            title.pos = pos
            if title.anchors != anchors:
                title.anchors = anchors
            if abs(float(np.reshape(title.rotation, -1)[0]) - angle) > 0.5:
                title.rotation = angle

    def _late_line(self, **kwargs):
        line = _late_class()(parent=self._gantry, **kwargs)
        # Line sets no GL state of its own and would inherit the march's "always".
        line.set_gl_state("translucent", depth_test=True, depth_func="lequal")
        self._late.append(line)
        return line

    def _add_bounds(self, origin, extent_mm) -> None:
        self._bounds = self._late_line(
            pos=box_edge_segments(origin, extent_mm),
            connect="segments",
            color=(*self._ink, BOX_ALPHA),
            width=1,
        )
        self._nodes.append(self._bounds)
        self._inked.append((self._bounds, BOX_ALPHA))

    def _set_background(self, rgb) -> None:
        """The view background is the scale's zero color, so empty space reads as 0."""
        if rgb == self._bg:
            return
        self._bg = rgb
        self._ink = ink_rgb(rgb)
        self._canvas.bgcolor = rgb
        for node, alpha in self._inked:
            if node.parent is None:
                continue
            if hasattr(node, "set_data"):
                node.set_data(color=(*self._ink, alpha))
            else:
                node.color = (*self._ink, alpha)

    def _ensure_textures(self, shape_zyx) -> None:
        key = tuple(int(v) for v in shape_zyx)
        if self._tex_shape == key and self._meas_tex is not None and self._plan_tex is not None:
            return
        self._canvas.set_current()
        self._tex_shape = key
        self._mc_drop()
        self._meas_tex = _alloc_texture(key)
        self._plan_tex = _alloc_texture(key)
        self._gamma_tex = _alloc_texture(key)
        self._box.set_volumes(self._meas_tex, self._plan_tex)

    def _mc_start(self, slot, tex, batch, grid, medium_key, *, depth, wet, spread_pct, histories, ic_gap_mm):
        """Start filling *tex* with *batch* by Monte Carlo, or retarget the run it holds for the same inputs.

        Returns the McResult when the texture already holds the finished run, else None.
        """
        if medium_key not in MC_MEDIA:
            raise ValueError(f"no MCsquare material data for {medium_key!r}")
        protons = deposit_amounts(batch, WEIGHT_DOSE, ic_gap_mm)
        c = np.asarray(batch.center, dtype=float)
        s = np.asarray(batch.sigma, dtype=float)
        e = np.asarray(batch.energy_mev, dtype=float)
        digest = hashlib.sha1()
        for a in (c[:, 0:2], s[:, 0:2], e, protons):
            digest.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        key = (
            digest.hexdigest(), tuple(float(v) for v in grid.origin), tuple(grid.shape), float(grid.voxel),
            medium_key, float(depth), float(wet), float(spread_pct), MC_SEED,
        )
        held = self._mc.get(slot)
        if held is not None and held[0] == key and held[1].extend(histories):
            run = held[1]
        else:
            self._mc_drop(slot)
            run = McRun.slab(
                c[:, 0], c[:, 1], s[:, 0], s[:, 1], e, protons, medium_key, grid,
                depth=depth, wet=wet, spread_pct=spread_pct, histories=histories, seed=MC_SEED,
            )
            self._mc[slot] = (key, run)
            if run.done:
                tex.set_data(run.dose)
        if run.done:
            return run.result
        self._mc_preview_at = 0.0
        return None

    def _mc_drop(self, slot: str | None = None) -> None:
        """Free the run in *slot*, or all of them."""
        for s in [slot] if slot is not None else list(self._mc):
            held = self._mc.pop(s, None)
            if held is not None:
                held[1].close()

    def _mc_stop(self) -> None:
        """Drop the runs still refining; finished ones stay for reuse."""
        for slot, (_key, run) in list(self._mc.items()):
            if not run.done:
                self._mc_drop(slot)

    def _mc_texture(self, slot: str):
        return self._meas_tex if slot == "meas" else self._plan_tex

    def _mc_progress_note(self) -> str:
        return f" · MC {100.0 * self.mc_progress:.0f} %"

    def mc_step(self, budget_s: float | None = None) -> bool:
        """Advance one refining Monte Carlo run for about *budget_s*; True while any is still going.

        Measured and plan take turns, so both previews stay at about the same progress.
        By default the slice is short while the view is moving, so a drag keeps its frames.
        """
        pending = [(slot, run) for slot, (_, run) in self._mc.items() if not run.done]
        if not pending:
            return False
        if budget_s is None:
            moving = time.perf_counter() - self._moved_at < MC_PREVIEW_S
            budget_s = MC_SLICE_S if moving else MC_IDLE_SLICE_S
        self._mc_turn = (self._mc_turn + 1) % len(pending)
        try:
            slot, run = pending[self._mc_turn]
            if run.step(budget_s):
                self._mc_texture(slot).set_data(run.dose)
        except Exception:
            _log.warning("Monte Carlo dose unavailable", exc_info=True)
            for slot in list(self._mc):
                self._mc_drop(slot)
                self._mc_texture(slot).set_data(np.zeros(self._tex_shape, dtype=np.float32))
            self._canvas.context.flush_commands()
            self.volume_note = self._note_base + " · MC unavailable"
            self._post = None
            self._canvas.update()
            return False
        if not self.mc_refining:
            first = self._mc.get("meas") or self._mc.get("plan")
            self.volume_note = self._note_base + mc_note(first[1].result)
            self._finish_volume()
        elif time.perf_counter() >= self._mc_preview_at:
            for slot, run in pending:
                image = run.preview()
                if image is not None:
                    self._mc_texture(slot).set_data(image)
            self._mc_preview_at = time.perf_counter() + MC_PREVIEW_S
            self.volume_note = self._note_base + self._mc_progress_note()
        else:
            return True
        self._mc_redraw = True
        self._canvas.update()
        return self.mc_refining

    def _finish_volume(self) -> None:
        """γ and the field box, which need the finished dose."""
        post, self._post = self._post, None
        if post is None:
            return
        grid, want_gamma, crit, src, (fraction, per_slice), show_field = post
        nx, ny, nz = grid.shape
        if want_gamma:
            self._gamma = True
            self.gamma_pass = gamma_texture(
                self._canvas, self._meas_tex, self._plan_tex, self._gamma_tex, grid, crit,
            )
            self._box.set_volumes(self._gamma_tex, self._plan_tex)
            self._push_display()
        try:
            boxed = field_box(read_texture(self._canvas, src, (nz, ny, nx)), grid.origin, grid.voxel, fraction,
                              per_slice=per_slice)
        except Exception:
            _log.debug("field box unavailable", exc_info=True)
            boxed = None
        if boxed is not None:
            f_o, f_e = boxed
            self.field_extent = (float(f_e[0]), float(f_e[1]), float(f_e[2]))
            if show_field:
                self._nodes.append(self._late_line(
                    pos=box_edge_segments(f_o, f_e), connect="segments", color=FIELD_RGBA, width=1,
                ))

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
        error_mode: str = ERROR_ABSOLUTE,
        error_scale: float = DEFAULT_ERROR_SCALE,
        weight_mode: str = WEIGHT_MU,
        ic_gap_mm: float = 10.0,
        smear: float = 0.5,
        ray_mode: str = DEFAULT_RAY,
        scale: str = DEFAULT_SCALE,
        auto_scale: bool = True,
        voxel_mm: float = VOXEL_MM,
        interp: str = "linear",
        gamma: bool = False,
        gamma_criteria: GammaCriteria | None = None,
        phantom_mm: float = 0.0,
        entrance_wet_mm: float = 0.0,
        auto_margin_sigma: float = 5.0,
        scatter: bool = True,
        field_edge: str = "slice50",
        show_phantom: bool = False,
        show_field: bool = True,
        status: str | None = None,
        model: str = MODEL_ANALYTIC,
        mc_histories: int = DEFAULT_MC_HISTORIES,
    ) -> None:
        from vispy import scene

        self.clear_guides()
        self._set_gantry(gantry_deg)
        self._has_volume = False
        self._broken = False
        self.volume_note = ""
        self.phantom_note = ""
        self.field_extent = None
        self._gain = float(gain)
        self._error_scale = float(error_scale)
        self._absolute = error_mode == ERROR_ABSOLUTE
        self._difference = bool(difference)
        self._ray_mode = ray_mode if ray_mode in _RAY_CODE else DEFAULT_RAY
        self._scale = active_scale(self._difference, scale)
        self._auto = bool(auto_scale)
        self.auto_lo = None
        self.auto_hi = None
        self._box.set_interp(interp)
        self._bounds = None
        self._post = None
        self._want_gamma = False

        if status:
            self._mc_stop()
            text = scene.Text(
                status, color=self._ink, font_size=14, pos=(0.0, 0.0, 0.0), parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        batches = [b for b in (measured, plan) if b is not None and b.center.size]
        if not batches:
            self._mc_stop()
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
            kernel = layer_kernel(medium.key, np.concatenate([s[3] for s in live]), smear, scatter)
        else:
            kernel = GaussianSmearKernel(axis, smear)

        def prepared(spots):
            """Spots with deposit sigmas and weights, plus the (center, sigma) box the grid holds."""
            center, sigma, amounts, e = spots
            if dose:
                c, s = kernel.span(center, sigma, e)
                return center, sigma, dose_weights(kernel, amounts, e, medium), e, c, s
            if medium_key and scatter:
                # Stops land wider than the IC saw the beam: scatter over the full range.
                sigma = sigma.copy()
                sigma[:, 0:2] = np.hypot(sigma[:, 0:2], end_scatter_mm(medium, e)[:, None])
            return center, sigma, amounts, e, center, sigma

        meas = prepared(meas_in) if meas_in is not None else None
        plan_prep = prepared(plan_in) if plan_in is not None else None
        # The grid holds the plan either way, so switching Show keeps the frame.
        spans = [p for p in (meas, plan_prep) if p is not None]
        fraction, per_slice, from_plan = field_edge_spec(field_edge)
        # The march ignores the plan texture unless a difference is shown, so the
        # planned field can be filled without changing the picture.
        planned = plan_prep if difference or gamma or from_plan else None
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
        self.volume_note = f"{nx} × {ny} × {nz}"
        if grid.cropped:
            self.volume_note += " · cropped"
        if medium_key:
            self.phantom_note = phantom_note(
                medium, depth or 0.0, wet,
                auto=phantom_mm <= 0.0,
                n_nozzle=sum(len(b.energy_mev) for b in batches),
                entry_energy=entry_energy,
            )
        self._ensure_textures((nz, ny, nx))
        empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
        gpu = True
        mc = dose and model == MODEL_MC
        mc_result = None
        mc_failed = False
        peak_w, peak_s, peak_e = np.zeros(0), np.zeros((0, 3)), np.zeros(0)
        slots = (("meas", self._meas_tex, meas, measured), ("plan", self._plan_tex, planned, plan))
        for slot, tex, prep, raw in slots:
            if prep is None or not mc:
                self._mc_drop(slot)
            if prep is None:
                fill_texture(self._canvas, tex, *empty, kernel, grid)
                continue
            center, sigma, amounts, e = prep[:4]
            peak_w, peak_s, peak_e = amounts, np.array(sigma, dtype=float), e
            if not mc:
                gpu = fill_texture(self._canvas, tex, center, sigma, amounts, e, kernel, grid) and gpu
                continue
            try:
                result = self._mc_start(
                    slot, tex, raw, grid, medium_key, depth=depth, wet=wet, spread_pct=smear,
                    histories=mc_histories, ic_gap_mm=ic_gap_mm,
                )
                mc_result = mc_result or result
            except Exception:
                _log.warning("Monte Carlo dose unavailable", exc_info=True)
                self._mc_drop(slot)
                tex.set_data(np.zeros((nz, ny, nx), dtype=np.float32))
                self._canvas.context.flush_commands()
                mc_failed = True
        if planned is None:
            self._difference = False
        if mc_failed:
            self._mc_stop()
            self.volume_note += " · MC unavailable"
        self._note_base = self.volume_note
        if self.mc_refining:
            self.volume_note += self._mc_progress_note()
        elif mc_result is not None:
            self.volume_note += mc_note(mc_result)
        if not gpu:
            self.volume_note += " · CPU fill"
        if dose and peak_w.size:
            peak_s[:, 2] = effective_sigma_z(kernel, peak_e)
        self._typical = typical_cell(peak_w, peak_s)
        self.dose_peak = self._typical
        self.ray_peak = typical_ray(peak_w, peak_s)

        self._want_gamma = bool(gamma) and meas is not None and planned is not None
        self._gamma = False
        self.gamma_pass = None
        crit = gamma_criteria or GammaCriteria()
        if self._want_gamma:
            self._gamma_cap = crit.cap
            self._difference = False
        self._box.set_volumes(self._meas_tex, self._plan_tex)
        self._box.set_box(grid.origin, grid.shape, grid.voxel)
        self._scale = active_scale(self._difference, self._scale)
        self._push_display()
        self._has_volume = True

        corners = volume_corners(grid.origin, grid.extent_mm)
        self._add_bounds(grid.origin, grid.extent_mm)
        self._add_axes(axis, grid.origin, grid.extent_mm)
        src = self._plan_tex if from_plan else self._meas_tex if meas is not None else self._plan_tex
        self._post = (grid, self._want_gamma, crit, src, (fraction, per_slice), show_field)
        if not self.mc_refining:
            self._finish_volume()
        if depth:
            p_o, p_e = phantom_box(grid.origin, grid.extent_mm, depth)
            if show_phantom:
                self._nodes.append(self._late_line(
                    pos=box_edge_segments(p_o, p_e), connect="segments", color=PHANTOM_RGBA, width=1,
                ))
                corners = np.vstack([corners, volume_corners(p_o, p_e)])
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

    def set_interp(self, mode: str) -> None:
        self._box.set_interp(mode)
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

        if self._mc_redraw:
            self._mc_redraw = False
        else:
            self._moved_at = time.perf_counter()
        marching = self._has_volume and not self._broken
        for line in self._late:
            line.held = marching
        self._place_labels()
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
            if line.visible:
                line.draw()
        self._consume_auto_span(span)

    def _consume_auto_span(self, span) -> None:
        if not self._auto or self._gamma or span is None:
            return
        scale = self.ray_peak if self._ray_mode == RAY_INTEGRAL else self.dose_peak
        got = auto_color_range(self._difference, span[0], span[1], AUTO_DIFF_FLOOR * scale)
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
