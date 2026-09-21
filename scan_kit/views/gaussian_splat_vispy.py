"""visPy 3D scene for Gaussian splat clouds."""

from __future__ import annotations

import logging

import numpy as np

from .gaussian_splat_catalog import (
    AGREE_TRANSPARENT,
    AGREE_WHITE,
    DEFAULT_ERROR_SCALE,
    DEFAULT_GAIN,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
)
from .gaussian_splat_data import DepthAxis, SplatBatch
from .gaussian_splat_visual import (
    COLD_RGB,
    HOT_RGB,
    auto_amp_scale,
    make_gaussian_splat_node,
    residual_abs_fbo_scale,
)
from .vispy_plot import FG, bind_blender_view_keys

_log = logging.getLogger(__name__)

_COMPOSITE_VERT = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = 0.5 * (a_position + 1.0);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_COMPOSITE_FRAG = """
uniform sampler2D u_tex;
uniform float u_white;
uniform float u_percent;
uniform float u_scale;
varying vec2 v_uv;

void main() {
    vec4 s = texture2D(u_tex, v_uv);
    float meas = s.r;
    float plan = s.b;
    float tot = meas + plan;
    if (tot < 1e-4) {
        discard;
    }
    float d = meas - plan;
    float mag;
    if (u_percent > 0.5) {
        if (meas < 0.08 && plan < 0.08) {
            discard;
        }
        mag = abs(d) / max(plan, 0.08);
    } else {
        mag = abs(d);
    }
    vec3 col = d >= 0.0 ? vec3(%f, %f, %f) : vec3(%f, %f, %f);
    float lo = u_scale * 0.02 / 0.30;
    float a = smoothstep(lo, max(u_scale, 1e-6), mag);
    if (u_white > 0.5) {
        gl_FragColor = vec4(mix(vec3(1.0), col, a), 1.0);
    } else {
        if (a < 1e-3) {
            discard;
        }
        gl_FragColor = vec4(col * a, a);
    }
}
""" % (HOT_RGB + COLD_RGB)


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


class SplatScene:
    """Turntable view: X/Y lateral mm, Z from :class:`DepthAxis`, then gantry Rx."""

    def __init__(self, canvas) -> None:
        from vispy import scene

        self._canvas = canvas
        self._view = canvas.central_widget.add_view()
        self._view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=400,
            center=(0.0, 0.0, 0.0),
            elevation=25,
        )
        bind_blender_view_keys(canvas, lambda: self._view.camera)
        self._gantry = scene.Node(parent=self._view.scene)
        self._gantry.transform = scene.transforms.MatrixTransform()
        self._nodes: list = []
        splat_cls = make_gaussian_splat_node()
        self._measured = splat_cls(parent=self._gantry)
        self._plan = splat_cls(parent=self._gantry)
        self._axis_label = "Energy (MeV)"
        self._residual = False
        self._agreement = AGREE_TRANSPARENT
        self._gain = DEFAULT_GAIN
        self._error_mode = ERROR_PERCENT
        self._error_scale = DEFAULT_ERROR_SCALE
        self._plan_weights = np.empty((0,), dtype=np.float32)
        self._fbo = None
        self._tex = None
        self._fbo_size = (0, 0)
        self._comp = None
        self._residual_broken = False
        # getattr(canvas, 'on_draw') each frame — wrapping runs before flush.
        self._canvas.on_draw = self._on_draw_with_residual

    def _set_gantry(self, degrees: float) -> None:
        xf = self._gantry.transform
        xf.reset()
        xf.rotate(float(degrees), (1.0, 0.0, 0.0))

    def clear_guides(self) -> None:
        for node in self._nodes:
            node.parent = None
        self._nodes.clear()

    def _set_empty(self, measured: bool, plan: bool) -> None:
        empty_c = np.empty((0, 3), dtype=np.float32)
        empty_s = np.empty((0, 3), dtype=np.float32)
        empty_w = np.empty((0,), dtype=np.float32)
        empty_rgb = np.empty((0, 3), dtype=np.float32)
        if measured:
            self._measured.set_data(empty_c, empty_s, empty_w, empty_rgb)
        if plan:
            self._plan.set_data(empty_c, empty_s, empty_w, empty_rgb)

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

    def render(
        self,
        measured: SplatBatch | None,
        plan: SplatBatch | None,
        axis: DepthAxis,
        *,
        gain: float,
        gantry_deg: float = 90.0,
        residual: bool = False,
        agreement: str = AGREE_TRANSPARENT,
        error_mode: str = ERROR_PERCENT,
        error_scale: float = DEFAULT_ERROR_SCALE,
        status: str | None = None,
    ) -> None:
        from vispy import scene

        self.clear_guides()
        self._set_gantry(gantry_deg)
        self._axis_label = axis.axis_label

        if status:
            self._residual = False
            self._set_empty(True, True)
            self._measured.visible = False
            self._plan.visible = False
            text = scene.Text(
                status,
                color=FG,
                font_size=14,
                pos=(0.0, 0.0, 0.0),
                parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        has_meas = measured is not None and measured.center.size
        has_plan = plan is not None and plan.center.size
        residual = bool(residual) and has_meas and has_plan
        self._residual = residual
        self._agreement = AGREE_WHITE if agreement == AGREE_WHITE else AGREE_TRANSPARENT
        self._gain = float(gain)
        self._error_mode = ERROR_PERCENT if error_mode == ERROR_PERCENT else ERROR_ABSOLUTE
        self._error_scale = float(error_scale)
        self._plan_weights = (
            np.asarray(plan.weight, dtype=np.float32).reshape(-1)
            if residual else np.empty((0,), dtype=np.float32)
        )
        dummy_rgb = np.ones((1, 3), dtype=np.float32)
        shared_scale = None
        if residual:
            shared_scale = auto_amp_scale(
                np.concatenate([measured.weight, plan.weight]),
                np.concatenate([measured.sigma, plan.sigma]),
            )

        if has_meas:
            rgb = dummy_rgb if residual else measured.rgb
            n = int(measured.center.shape[0])
            self._measured.set_data(
                measured.center, measured.sigma, measured.weight,
                np.broadcast_to(rgb, (n, 3)).copy() if residual else measured.rgb,
                gain=gain, diverging=residual, amp_scale=shared_scale,
            )
        else:
            self._set_empty(True, False)
        if has_plan:
            n = int(plan.center.shape[0])
            if residual:
                plan_w = np.ascontiguousarray(-plan.weight, dtype=np.float32)
                plan_rgb = np.broadcast_to(dummy_rgb, (n, 3)).copy()
            else:
                plan_w = plan.weight
                plan_rgb = plan.rgb
            self._plan.set_data(
                plan.center, plan.sigma, plan_w, plan_rgb,
                gain=gain, diverging=residual, amp_scale=shared_scale,
            )
        else:
            self._set_empty(False, True)

        hide = residual and not self._residual_broken
        self._measured.visible = has_meas and not hide
        self._plan.visible = has_plan and not hide

        parts = []
        if measured is not None and measured.center.size:
            parts.append(measured.center)
        if plan is not None and plan.center.size:
            parts.append(plan.center)
        if not parts:
            self._residual = False
            self._set_empty(True, True)
            text = scene.Text(
                "No Gaussian samples for this mode",
                color=FG,
                font_size=14,
                pos=(0.0, 0.0, 0.0),
                parent=self._view.scene,
            )
            self._nodes.append(text)
            self._canvas.update()
            return

        pts = np.vstack(parts)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        extent = np.vstack([lo, hi])
        self._add_axes(axis, extent)
        _starts, _ends, label_pts = axis_guide_points(
            extent, depth_sign=int(getattr(axis, "depth_sign", 1)),
        )
        corners = np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ],
            dtype=np.float64,
        )
        world = apply_gantry(np.vstack([corners, label_pts]), gantry_deg)
        wlo = world.min(axis=0)
        whi = world.max(axis=0)
        pad = np.maximum((whi - wlo) * 0.08, 1.0)
        self._view.camera.center = tuple((wlo + whi) / 2.0)
        self._view.camera.set_range(
            x=(wlo[0] - pad[0], whi[0] + pad[0]),
            y=(wlo[1] - pad[1], whi[1] + pad[1]),
            z=(wlo[2] - pad[2], whi[2] + pad[2]),
        )
        self._canvas.update()

    def set_agreement(self, zero: str) -> None:
        self._agreement = AGREE_WHITE if zero == AGREE_WHITE else AGREE_TRANSPARENT
        self._canvas.update()

    def set_gain(self, gain: float) -> None:
        self._gain = float(gain)
        self._measured.set_gain(gain)
        self._plan.set_gain(gain)
        self._canvas.update()

    def set_error_metric(self, mode: str, scale: float) -> None:
        self._error_mode = ERROR_PERCENT if mode == ERROR_PERCENT else ERROR_ABSOLUTE
        self._error_scale = float(scale)
        self._canvas.update()

    def _composite_u_scale(self) -> float:
        if self._error_mode == ERROR_PERCENT:
            return max(self._error_scale, 1e-6)
        return residual_abs_fbo_scale(self._plan_weights, self._error_scale, self._gain)

    def _on_draw_with_residual(self, event) -> None:
        from vispy.scene import SceneCanvas

        SceneCanvas.on_draw(self._canvas, event)
        self._draw_residual_overlay()

    def _draw_residual_overlay(self) -> None:
        if not self._residual or self._residual_broken:
            return
        try:
            self._composite_residual()
        except Exception:
            if not self._residual_broken:
                _log.exception("Gaussian splat residual composite failed")
            self._residual_broken = True
            self._measured.visible = True
            self._plan.visible = True
            self._canvas.update()

    def _ensure_fbo(self, width: int, height: int):
        from vispy import gloo

        if self._fbo is not None and self._fbo_size == (width, height):
            return
        self._fbo_size = (width, height)
        try:
            tex = gloo.Texture2D(
                shape=(height, width, 4),
                format="rgba",
                internalformat="rgba16f",
                interpolation="linear",
            )
        except Exception:
            tex = gloo.Texture2D(
                shape=(height, width, 4),
                format="rgba",
                interpolation="linear",
            )
        self._tex = tex
        self._fbo = gloo.FrameBuffer(color=tex)

    def _ensure_comp(self):
        from vispy import gloo

        if self._comp is not None:
            return
        self._comp = gloo.Program(_COMPOSITE_VERT, _COMPOSITE_FRAG)
        self._comp["a_position"] = np.array(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )

    def _composite_residual(self) -> None:
        from vispy import gloo

        w, h = self._canvas.physical_size
        width, height = int(w), int(h)
        if width < 2 or height < 2:
            return
        self._ensure_fbo(width, height)
        self._ensure_comp()
        self._measured.visible = True
        self._plan.visible = True
        self._canvas.push_fbo(self._fbo, (0, 0), (width, height))
        try:
            gloo.set_state(depth_test=False, blend=True, blend_func=("one", "one"))
            self._canvas.context.clear(color=(0.0, 0.0, 0.0, 0.0), depth=True)
            self._measured.draw()
            self._plan.draw()
        finally:
            self._canvas.pop_fbo()
            self._measured.visible = False
            self._plan.visible = False
        self._comp["u_tex"] = self._tex
        self._comp["u_white"] = 1.0 if self._agreement == AGREE_WHITE else 0.0
        self._comp["u_percent"] = 1.0 if self._error_mode == ERROR_PERCENT else 0.0
        self._comp["u_scale"] = self._composite_u_scale()
        # Premul overlay: ONE, 1-SRC_ALPHA. SRC_ALPHA + unassociated rgb
        # painted full-strength color onto the dark scene (no transparency).
        gloo.set_state(
            depth_test=False,
            blend=True,
            blend_func=("one", "one_minus_src_alpha"),
        )
        self._comp.draw("triangle_strip")
