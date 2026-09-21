"""Instanced 3D Gaussian splat visual (EWA / 3DGS projection, additive blend)."""

from __future__ import annotations

import numpy as np

from .gaussian_splat_catalog import (
    DEFAULT_ERROR_PCT,
    ERROR_ABSOLUTE,
    ERROR_PERCENT,
)

_VERT = """
attribute vec2 a_corner;
attribute vec3 a_center;
attribute vec3 a_sigma;
attribute float a_weight;
attribute vec3 a_rgb;

uniform float u_gain;

varying vec2 v_local;
varying vec3 v_rgb;
varying float v_amp;

const float EPS = 0.25;

void main() {
    vec4 fb0 = $visual_to_framebuffer(vec4(a_center, 1.0));
    vec4 fbx = $visual_to_framebuffer(vec4(a_center + vec3(EPS, 0.0, 0.0), 1.0));
    vec4 fby = $visual_to_framebuffer(vec4(a_center + vec3(0.0, EPS, 0.0), 1.0));
    vec4 fbz = $visual_to_framebuffer(vec4(a_center + vec3(0.0, 0.0, EPS), 1.0));

    float w0 = max(abs(fb0.w), 1e-8);
    vec2 c = fb0.xy / w0;
    vec2 jx = (fbx.xy / max(abs(fbx.w), 1e-8) - c) / EPS;
    vec2 jy = (fby.xy / max(abs(fby.w), 1e-8) - c) / EPS;
    vec2 jz = (fbz.xy / max(abs(fbz.w), 1e-8) - c) / EPS;

    float sx2 = a_sigma.x * a_sigma.x;
    float sy2 = a_sigma.y * a_sigma.y;
    float sz2 = a_sigma.z * a_sigma.z;
    float a = jx.x * jx.x * sx2 + jy.x * jy.x * sy2 + jz.x * jz.x * sz2;
    float b = jx.x * jx.y * sx2 + jy.x * jy.y * sy2 + jz.x * jz.y * sz2;
    float d = jx.y * jx.y * sx2 + jy.y * jy.y * sy2 + jz.y * jz.y * sz2;

    float tr = a + d;
    float det = a * d - b * b;
    float disc = sqrt(max(tr * tr - 4.0 * det, 0.0));
    float l1 = max(0.5 * (tr + disc), 1.0);
    float l2 = max(0.5 * (tr - disc), 0.25);

    vec2 e1;
    if (abs(b) > 1e-8) {
        e1 = normalize(vec2(b, l1 - a));
    } else {
        e1 = a >= d ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    }
    vec2 e2 = vec2(-e1.y, e1.x);

    vec2 offset = 3.0 * (a_corner.x * sqrt(l1) * e1 + a_corner.y * sqrt(l2) * e2);
    vec4 fb = vec4((c + offset) * fb0.w, fb0.z, fb0.w);
    gl_Position = $framebuffer_to_render(fb);

    v_local = 3.0 * a_corner;
    v_rgb = a_rgb;
    float two_pi_s = 6.283185307179586 * max(a_sigma.x * a_sigma.y, 1e-6);
    v_amp = u_gain * a_weight / two_pi_s;
}
"""

_FRAG = """
varying vec2 v_local;
varying vec3 v_rgb;
varying float v_amp;

uniform float u_diverging;

void main() {
    float r2 = dot(v_local, v_local);
    if (r2 > 9.0) {
        discard;
    }
    float gauss = exp(-0.5 * r2);
    float amp = v_amp * gauss;
    if (u_diverging > 0.5) {
        gl_FragColor = vec4(max(amp, 0.0), 0.0, max(-amp, 0.0), 1.0);
    } else {
        gl_FragColor = vec4(v_rgb * max(amp, 0.0), 1.0);
    }
}
"""

_QUAD = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
# Two triangles covering the same unit quad when instancing is unavailable.
_TRI_CORNERS = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ],
    dtype=np.float32,
)


def instanced_gl_available() -> bool:
    """True when vispy's current GL wrapper can issue instanced draws."""
    try:
        from vispy.gloo import gl

        return hasattr(gl, "glDrawArraysInstanced")
    except Exception:
        return False


def expand_splat_vertices(centers, sigmas, weights, rgb):
    """Repeat per-splat attributes onto 6 triangle vertices (CPU fallback)."""
    n = int(np.asarray(centers).shape[0])
    if n == 0:
        empty2 = np.empty((0, 2), dtype=np.float32)
        empty3 = np.empty((0, 3), dtype=np.float32)
        empty1 = np.empty((0,), dtype=np.float32)
        return empty2, empty3, empty3, empty1, empty3
    corners = np.tile(_TRI_CORNERS, (n, 1))
    return (
        corners,
        np.repeat(centers, 6, axis=0),
        np.repeat(sigmas, 6, axis=0),
        np.repeat(weights, 6, axis=0),
        np.repeat(rgb, 6, axis=0),
    )


HOT_RGB = (0.90, 0.16, 0.14)
COLD_RGB = (0.16, 0.40, 0.90)
# Must match the residual composite shader's smoothstep.
_RESIDUAL_ALPHA_LO = 0.02
_RESIDUAL_ALPHA_HI = 0.30
# Peak-normalized floor so percent-of-plan does not divide by empty tails.
_PLAN_FLOOR = 0.08


def residual_error_mag(meas, plan, *, mode: str = ERROR_ABSOLUTE):
    """Unsigned residual in display units (fraction of plan, or FBO |Δ|)."""
    meas = np.asarray(meas, dtype=np.float64)
    plan = np.asarray(plan, dtype=np.float64)
    delta = meas - plan
    if mode == ERROR_PERCENT:
        return np.abs(delta) / np.maximum(plan, _PLAN_FLOOR)
    return np.abs(delta)


def residual_error_alpha(mag, scale: float = _RESIDUAL_ALPHA_HI):
    """Hermite smoothstep of error magnitude — same as the composite GLSL."""
    lo = float(scale) * (_RESIDUAL_ALPHA_LO / _RESIDUAL_ALPHA_HI)
    hi = max(float(scale), 1e-12)
    x = np.clip((np.asarray(mag, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def residual_typical_mu(weights) -> float:
    w = np.abs(np.asarray(weights, dtype=float).reshape(-1))
    w = w[np.isfinite(w) & (w > 0.0)]
    return float(np.median(w)) if w.size else 1.0


def residual_abs_fbo_scale(weights, error_scale_mu: float, gain: float) -> float:
    """FBO |meas−plan| that saturates the color axis at ``error_scale_mu``.

    Auto-amp maps a typical peak to ~1 at gain=1, so the MU scale is
    ``gain * error_scale_mu / typical_mu``.
    """
    typical = residual_typical_mu(weights)
    return max(float(gain), 1e-6) * float(error_scale_mu) / max(typical, 1e-6)


def residual_agreement_rgba(
    meas,
    plan,
    *,
    zero: str = "transparent",
    mode: str = ERROR_ABSOLUTE,
    scale: float | None = None,
):
    """Colormap accumulated measured (R) vs plan (B) fields.

    Default alpha follows absolute ``|m−p|``, not ``|m−p|/(m+p)``. Relative
    error lights up every Gaussian tail (one field only) as a hollow shell.
    Percent-of-plan ignores faint pairs below :data:`_PLAN_FLOOR`.
    Agreement is transparent or white; empty pixels stay clear.
    """
    meas = np.asarray(meas, dtype=np.float64)
    plan = np.asarray(plan, dtype=np.float64)
    if scale is None:
        scale = (
            DEFAULT_ERROR_PCT / 100.0 if mode == ERROR_PERCENT
            else _RESIDUAL_ALPHA_HI
        )
    tot = meas + plan
    empty = tot <= 1e-6
    if mode == ERROR_PERCENT:
        empty = empty | ((meas < _PLAN_FLOOR) & (plan < _PLAN_FLOOR))
    mag = residual_error_mag(meas, plan, mode=mode)
    rgb = np.empty(meas.shape + (3,), dtype=np.float64)
    hot = meas >= plan
    rgb[hot] = HOT_RGB
    rgb[~hot] = COLD_RGB
    w = residual_error_alpha(mag, scale)
    if zero == "white":
        rgb = (1.0 - w[..., None]) + w[..., None] * rgb
        alpha = np.where(empty, 0.0, 1.0)
    else:
        alpha = np.where(empty, 0.0, w)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def auto_amp_scale(weights, sigmas) -> float:
    """Scale unit-mass peaks so the 95th-percentile splat is ~1.0 at gain=1."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    s = np.asarray(sigmas, dtype=float).reshape(-1, 3)
    if w.size == 0 or s.shape[0] == 0:
        return 1.0
    peak = np.abs(w) / np.maximum(2.0 * np.pi * s[:, 0] * s[:, 1], 1e-6)
    peak = peak[np.isfinite(peak) & (peak > 0.0)]
    if peak.size == 0:
        return 1.0
    return 1.0 / max(float(np.percentile(peak, 95)), 1e-8)


def finite_diff_jacobian(
    map_xy,
    center,
    *,
    eps: float = 0.25,
) -> np.ndarray:
    """2×3 Jacobian of *map_xy* at *center* (same stencil as the vertex shader)."""
    c = np.asarray(center, dtype=float).reshape(3)
    p0 = np.asarray(map_xy(c), dtype=float).reshape(2)
    axes = np.eye(3, dtype=float)
    cols = [
        (np.asarray(map_xy(c + eps * axes[i]), dtype=float).reshape(2) - p0) / eps
        for i in range(3)
    ]
    return np.column_stack(cols)


def project_covariance(
    sigma,
    jacobian: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Eigenvalues and 2×2 eigenvectors of ``J diag(σ²) Jᵀ`` for a (2, 3) Jacobian."""
    sig = np.asarray(sigma, dtype=float).reshape(3)
    jac = np.asarray(jacobian, dtype=float).reshape(2, 3)
    screen = jac @ np.diag(np.square(sig)) @ jac.T
    evals, evecs = np.linalg.eigh(screen)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-12)
    return float(evals[0]), float(evals[1]), evecs[:, order]


def _visual_class():
    from vispy import gloo
    from vispy.visuals import Visual

    class GaussianSplatVisual(Visual):
        """Instanced anisotropic Gaussian splats with additive blending."""

        def __init__(self) -> None:
            Visual.__init__(self, vcode=_VERT, fcode=_FRAG)
            self._corner_vbo = gloo.VertexBuffer(_QUAD)
            self._center_vbo = gloo.VertexBuffer()
            self._sigma_vbo = gloo.VertexBuffer()
            self._weight_vbo = gloo.VertexBuffer()
            self._rgb_vbo = gloo.VertexBuffer()
            self.shared_program["a_corner"] = self._corner_vbo
            self.shared_program["a_center"] = self._center_vbo
            self.shared_program["a_sigma"] = self._sigma_vbo
            self.shared_program["a_weight"] = self._weight_vbo
            self.shared_program["a_rgb"] = self._rgb_vbo
            self.shared_program["u_gain"] = 1.0
            self.shared_program["u_diverging"] = 0.0
            self._draw_mode = "triangle_strip"
            self._n = 0
            self._amp_scale = 1.0
            self._user_gain = 1.0
            self._centers = np.empty((0, 3), dtype=np.float32)
            self.set_gl_state(
                depth_test=False,
                blend=True,
                blend_func=("one", "one"),
                cull_face=False,
            )

        def set_data(
            self,
            centers,
            sigmas,
            weights,
            rgb,
            *,
            gain: float = 1.0,
            diverging: bool = False,
            amp_scale: float | None = None,
        ) -> None:
            centers = np.ascontiguousarray(centers, dtype=np.float32).reshape(-1, 3)
            sigmas = np.ascontiguousarray(sigmas, dtype=np.float32).reshape(-1, 3)
            weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
            rgb = np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1, 3)
            n = int(centers.shape[0])
            if sigmas.shape[0] != n or weights.shape[0] != n or rgb.shape[0] != n:
                raise ValueError("splat attribute lengths must match")
            self._n = n
            self._centers = centers
            self._user_gain = float(gain)
            self._amp_scale = (
                float(amp_scale) if amp_scale is not None
                else auto_amp_scale(weights, sigmas)
            )
            self.shared_program["u_diverging"] = 1.0 if diverging else 0.0
            if n == 0:
                self.update()
                return
            # Mutate existing VBOs — scene nodes are frozen after create_visual_node.
            # Re-bind after set_data so vispy's ATTRIBUTE command picks up divisor.
            if instanced_gl_available():
                self._draw_mode = "triangle_strip"
                self._upload_vbo(self._corner_vbo, "a_corner", _QUAD, None)
                self._upload_vbo(self._center_vbo, "a_center", centers, 1)
                self._upload_vbo(self._sigma_vbo, "a_sigma", sigmas, 1)
                self._upload_vbo(self._weight_vbo, "a_weight", weights, 1)
                self._upload_vbo(self._rgb_vbo, "a_rgb", rgb, 1)
            else:
                corners, centers, sigmas, weights, rgb = expand_splat_vertices(
                    centers, sigmas, weights, rgb,
                )
                self._draw_mode = "triangles"
                self._upload_vbo(self._corner_vbo, "a_corner", corners, None)
                self._upload_vbo(self._center_vbo, "a_center", centers, None)
                self._upload_vbo(self._sigma_vbo, "a_sigma", sigmas, None)
                self._upload_vbo(self._weight_vbo, "a_weight", weights, None)
                self._upload_vbo(self._rgb_vbo, "a_rgb", rgb, None)
            self.shared_program["u_gain"] = self._user_gain * self._amp_scale
            self.update()

        def _upload_vbo(self, vbo, name: str, data, divisor) -> None:
            vbo.set_data(data)
            vbo.divisor = divisor
            self.shared_program[name] = vbo

        def set_gain(self, gain: float) -> None:
            self._user_gain = float(gain)
            self.shared_program["u_gain"] = self._user_gain * self._amp_scale
            self.update()

        def _prepare_transforms(self, view) -> None:
            view.view_program.vert["visual_to_framebuffer"] = view.get_transform(
                "visual", "framebuffer",
            )
            view.view_program.vert["framebuffer_to_render"] = view.get_transform(
                "framebuffer", "render",
            )

        def _prepare_draw(self, view=None):
            if self._n <= 0:
                return False
            return True

        def _compute_bounds(self, axis, view):
            if self._centers.size == 0 or axis >= 3:
                return None
            col = self._centers[:, axis]
            return float(col.min()), float(col.max())

    return GaussianSplatVisual


_GaussianSplatVisual = None
_GaussianSplatNode = None


def GaussianSplatVisual():
    """Return the visPy Visual subclass (imported lazily)."""
    global _GaussianSplatVisual
    if _GaussianSplatVisual is None:
        _GaussianSplatVisual = _visual_class()
    return _GaussianSplatVisual


def make_gaussian_splat_node():
    """Scene-graph node class for the splat visual."""
    global _GaussianSplatNode
    if _GaussianSplatNode is None:
        from vispy.scene.visuals import create_visual_node

        _GaussianSplatNode = create_visual_node(GaussianSplatVisual())
    return _GaussianSplatNode
