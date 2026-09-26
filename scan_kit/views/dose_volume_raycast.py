"""GPU tile fill and box ray march for a voxel dose volume.

Compute builds the 2D beam-plane tile lists, then writes 8-cell bricks of
MU / mm³ straight into the float texture the ray marcher samples. Below
OpenGL 4.3 the same integral runs in Python.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .dose_volume_fill import (
    DOSE_FLOOR,
    VIEW_DEPTH_MM,
    DoseGrid,
    GaussianSmearKernel,
    deposit_gaussians,
    tile_shape,
    tile_slot_capacity,
)
from .dose_volume_physics import GammaCriteria, LayerDoseKernel, gamma_index

_log = logging.getLogger(__name__)


class _BoundImageFilter(logging.Filter):
    """``ray_image`` is bound by unit in raw GL, so vispy's unset warning is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "ray_image" not in record.getMessage()


logging.getLogger("vispy").addFilter(_BoundImageFilter())

_ERF = """
float erf_as(float x) {
    float s = sign(x);
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float p = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - p * exp(-ax * ax));
}
float gmass(float mu, float sigma, float lo, float hi) {
    float sig = max(sigma, 1e-6);
    float inv = 1.0 / (sig * 1.41421356237);
    return 0.5 * (erf_as((hi - mu) * inv) - erf_as((lo - mu) * inv));
}
"""

_COUNT = """
#version 430
layout(local_size_x = 64) in;
struct Spot { vec4 c; vec4 s; vec4 k; };
layout(std430, binding = 0) readonly buffer Spots { Spot spots[]; };
layout(std430, binding = 1) buffer Counts { uint counts[]; };
uniform vec3 u_origin;
uniform ivec2 u_ntiles;
uniform int u_nspots;
uniform float u_voxel;

ivec2 span(float coord, float sigma, float origin, int ntiles) {
    float sig = max(sigma, 1e-6);
    float tile = 16.0 * u_voxel;
    int t0 = int(floor((coord - 4.0 * sig - origin) / tile));
    int t1 = int(floor((coord + 4.0 * sig - origin) / tile));
    t0 = clamp(t0, 0, ntiles - 1);
    t1 = clamp(t1, 0, ntiles - 1);
    return ivec2(min(t0, t1), max(t0, t1));
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_nspots)) return;
    Spot sp = spots[i];
    if (sp.c.w == 0.0) return;
    ivec2 tx = span(sp.c.x, sp.s.x, u_origin.x, u_ntiles.x);
    ivec2 ty = span(sp.c.y, sp.s.y, u_origin.y, u_ntiles.y);
    for (int y = ty.x; y <= ty.y; ++y) {
        for (int x = tx.x; x <= tx.y; ++x) {
            atomicAdd(counts[uint(y * u_ntiles.x + x)], 1u);
        }
    }
}
"""

_SCAN = """
#version 430
layout(local_size_x = 1024) in;
layout(std430, binding = 1) readonly buffer Counts { uint counts[]; };
layout(std430, binding = 2) buffer Offsets { uint offsets[]; };
layout(std430, binding = 4) buffer Cursors { uint cursors[]; };
uniform int u_n;
shared uint sdata[1024];

void main() {
    uint i = gl_LocalInvocationID.x;
    uint n = uint(u_n);
    uint v = (i < n) ? counts[i] : 0u;
    sdata[i] = v;
    barrier();
    for (uint off = 1u; off < 1024u; off <<= 1u) {
        uint add = (i >= off) ? sdata[i - off] : 0u;
        barrier();
        if (i >= off) sdata[i] += add;
        barrier();
    }
    if (i < n) {
        uint excl = sdata[i] - v;
        offsets[i] = excl;
        cursors[i] = excl;
    }
}
"""

_FILL = """
#version 430
layout(local_size_x = 64) in;
struct Spot { vec4 c; vec4 s; vec4 k; };
layout(std430, binding = 0) readonly buffer Spots { Spot spots[]; };
layout(std430, binding = 3) writeonly buffer Ids { uint ids[]; };
layout(std430, binding = 4) buffer Cursors { uint cursors[]; };
uniform vec3 u_origin;
uniform ivec2 u_ntiles;
uniform int u_nspots;
uniform int u_capacity;
uniform float u_voxel;

ivec2 span(float coord, float sigma, float origin, int ntiles) {
    float sig = max(sigma, 1e-6);
    float tile = 16.0 * u_voxel;
    int t0 = int(floor((coord - 4.0 * sig - origin) / tile));
    int t1 = int(floor((coord + 4.0 * sig - origin) / tile));
    t0 = clamp(t0, 0, ntiles - 1);
    t1 = clamp(t1, 0, ntiles - 1);
    return ivec2(min(t0, t1), max(t0, t1));
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_nspots)) return;
    Spot sp = spots[i];
    if (sp.c.w == 0.0) return;
    ivec2 tx = span(sp.c.x, sp.s.x, u_origin.x, u_ntiles.x);
    ivec2 ty = span(sp.c.y, sp.s.y, u_origin.y, u_ntiles.y);
    for (int y = ty.x; y <= ty.y; ++y) {
        for (int x = tx.x; x <= tx.y; ++x) {
            uint tile = uint(y * u_ntiles.x + x);
            uint slot = atomicAdd(cursors[tile], 1u);
            if (slot < uint(u_capacity)) ids[slot] = i;
        }
    }
}
"""

_BRICK = """
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
struct Spot { vec4 c; vec4 s; vec4 k; };
layout(std430, binding = 0) readonly buffer Spots { Spot spots[]; };
layout(std430, binding = 1) readonly buffer Counts { uint counts[]; };
layout(std430, binding = 2) readonly buffer Offsets { uint offsets[]; };
layout(std430, binding = 3) readonly buffer Ids { uint ids[]; };
layout(binding = 1, r32f) writeonly uniform image3D u_img;
uniform vec3 u_origin;
uniform ivec3 u_shape;
uniform ivec2 u_ntiles;
uniform float u_voxel;
// Depth-dose tables: row per energy layer, (cdf, scatter width) per node.
uniform sampler2D u_layers;
uniform int u_nodes;
uniform int u_table;
""" + _ERF + """
// Spot: c = (x, y, z, weight), s = (span σx, span σy, σz, layer), k = (σx, σy, zmin, zmax).
vec2 layer_at(int layer, float z, float zmin, float zmax) {
    float t = clamp((z - zmin) / max(zmax - zmin, 1e-6), 0.0, 1.0) * float(u_nodes - 1);
    int i0 = int(floor(t));
    int i1 = min(i0 + 1, u_nodes - 1);
    vec2 a = texelFetch(u_layers, ivec2(i0, layer), 0).rg;
    vec2 b = texelFetch(u_layers, ivec2(i1, layer), 0).rg;
    return mix(a, b, t - float(i0));
}

void main() {
    ivec3 vox = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(vox, u_shape))) return;
    int tx = vox.x / 16;
    int ty = vox.y / 16;
    uint tile = uint(ty * u_ntiles.x + tx);
    uint begin = offsets[tile];
    uint end = begin + counts[tile];
    vec3 lo = u_origin + vec3(vox) * u_voxel;
    vec3 hi = lo + vec3(u_voxel);
    float acc = 0.0;
    for (uint k = begin; k < end; ++k) {
        Spot sp = spots[ids[k]];
        if (u_table == 1) {
            if (hi.z < sp.k.z || lo.z > sp.k.w) continue;
            int layer = int(sp.s.w + 0.5);
            float fz = layer_at(layer, hi.z, sp.k.z, sp.k.w).x
                - layer_at(layer, lo.z, sp.k.z, sp.k.w).x;
            float mcs = layer_at(layer, 0.5 * (lo.z + hi.z), sp.k.z, sp.k.w).y;
            float sx = sqrt(sp.k.x * sp.k.x + mcs * mcs);
            float sy = sqrt(sp.k.y * sp.k.y + mcs * mcs);
            acc += sp.c.w * fz
                * gmass(sp.c.x, sx, lo.x, hi.x)
                * gmass(sp.c.y, sy, lo.y, hi.y);
            continue;
        }
        float sigz = max(sp.s.z, 1e-6);
        if (hi.z < sp.c.z - 4.0 * sigz || lo.z > sp.c.z + 4.0 * sigz) continue;
        acc += sp.c.w
            * gmass(sp.c.x, sp.s.x, lo.x, hi.x)
            * gmass(sp.c.y, sp.s.y, lo.y, hi.y)
            * gmass(sp.c.z, sp.s.z, lo.z, hi.z);
    }
    // One invocation owns each voxel, so the float sum is written directly.
    imageStore(u_img, vox, vec4(acc / (u_voxel * u_voxel * u_voxel), 0.0, 0.0, 0.0));
}
"""

_BOX_VERT = """
#version 430 compatibility
attribute vec3 a_position;
varying vec3 v_pos;
void main() {
    v_pos = a_position;
    gl_Position = $framebuffer_to_render($visual_to_framebuffer(vec4(a_position, 1.0)));
}
"""

_BOX_FRAG = """
#version 430 compatibility
uniform int u_pass;
uniform sampler2D u_exit;
uniform sampler2D u_cmap;
uniform sampler3D u_meas;
uniform sampler3D u_plan;
uniform vec3 u_origin;
uniform vec3 u_shape;
uniform float u_voxel;
uniform float u_smooth;
uniform vec2 u_viewport;
uniform float u_gain;
uniform float u_typical;
uniform float u_error_scale;
uniform float u_diff;
uniform float u_absolute;
uniform float u_ray;
uniform float u_ray_scale;
uniform float u_auto;
uniform float u_lo;
uniform float u_hi;
// vispy treats a bare uniform image as one of its own variables.
#define RAY_IMAGE layout(binding = 4, rg32f) writeonly uniform image2D ray_image
RAY_IMAGE;
varying vec3 v_pos;

float voxel(sampler3D tex, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || float(x) >= u_shape.x
            || float(y) >= u_shape.y || float(z) >= u_shape.z) {
        return 0.0;
    }
    vec3 uv = (vec3(float(x), float(y), float(z)) + 0.5) / u_shape;
    return texture3D(tex, uv).r;
}

vec3 cmap(float t) {
    return texture2D(u_cmap, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb;
}

vec3 color_span(float v) {
    float span = max(u_hi - u_lo, 1e-12);
    return cmap(clamp((v - u_lo) / span, 0.0, 1.0));
}

float depth_of(vec3 pos) {
    vec4 c = $visual_to_render(vec4(pos, 1.0));
    return clamp(0.5 * c.z / c.w + 0.5, 0.0, 1.0);
}

void store_ray(float lo, float hi) {
    if (u_auto < 0.5) return;
    imageStore(ray_image, ivec2(gl_FragCoord.xy), vec4(lo, hi, 0.0, 0.0));
}

float catrom(float x) {
    float a = abs(x);
    if (a < 1.0)
        return (1.5 * a - 2.5) * a * a + 1.0;
    if (a < 2.0)
        return ((-0.5 * a + 2.5) * a - 4.0) * a + 2.0;
    return 0.0;
}

float sample_vol(sampler3D tex, vec3 p) {
    vec3 cell = (p - u_origin) / u_voxel;
    // 0 nearest, 1 trilinear, 2 Catmull-Rom tricubic.
    if (u_smooth < 0.5) {
        vec3 c = floor(cell);
        return voxel(tex, int(c.x), int(c.y), int(c.z));
    }
    vec3 q = cell - vec3(0.5);
    vec3 base = floor(q);
    vec3 f = q - base;
    if (u_smooth < 1.5) {
        float acc = 0.0;
        for (int dz = 0; dz < 2; ++dz) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    float wx = dx == 1 ? f.x : 1.0 - f.x;
                    float wy = dy == 1 ? f.y : 1.0 - f.y;
                    float wz = dz == 1 ? f.z : 1.0 - f.z;
                    acc += wx * wy * wz * voxel(
                        tex, int(base.x) + dx, int(base.y) + dy, int(base.z) + dz
                    );
                }
            }
        }
        return acc;
    }
    float acc = 0.0;
    for (int dz = 0; dz < 4; ++dz) {
        float wz = catrom(f.z - float(dz - 1));
        for (int dy = 0; dy < 4; ++dy) {
            float wy = catrom(f.y - float(dy - 1));
            for (int dx = 0; dx < 4; ++dx) {
                float wx = catrom(f.x - float(dx - 1));
                acc += wx * wy * wz * voxel(
                    tex, int(base.x) + dx - 1, int(base.y) + dy - 1, int(base.z) + dz - 1
                );
            }
        }
    }
    return acc;
}

void main() {
    if (u_pass == 0) {
        gl_FragColor = vec4(v_pos, 1.0);
        gl_FragDepth = gl_FragCoord.z;
        return;
    }
    vec2 uv = gl_FragCoord.xy / max(u_viewport, vec2(1.0));
    vec4 far = texture2D(u_exit, uv);
    if (far.a < 0.5) discard;
    vec3 entry = v_pos;
    vec3 exitp = far.xyz;
    vec3 delta = exitp - entry;
    float dist = length(delta);
    if (dist < 1e-3) discard;
    // One sample per voxel edge; the integral is in mm, so each adds meas * step.
    float step_mm = u_voxel;
    vec3 stepv = delta / dist * step_mm;
    int nstep = int(min(ceil(dist / step_mm), 1024.0));
    vec3 p = entry + 0.5 * stepv;
    float trans = 1.0;
    vec3 col = vec3(0.0);
    float integ = 0.0;
    float integ_signed = 0.0;
    float integ_tot = 0.0;
    float peak = 0.0;
    float peak_signed = 0.0;
    float peak_abs = 0.0;
    float peak_tot = 0.0;
    float ray_lo = 1.0 / 0.0;
    float ray_hi = -1.0 / 0.0;
    bool saw = false;
    // Where the dose sits along the ray, so the guide lines can depth test against it.
    vec3 wpos = vec3(0.0);
    float wsum = 0.0;
    vec3 hit = entry;
    bool half_seen = false;
    float typical = max(u_typical, 1e-8);
    float ray_scale = max(u_ray_scale, 1e-8);
    float floor_rel = %(floor).5f;
    // u_ray: 0 integrate the whole ray, 1 maximum, 2 transparent fog.
    for (int i = 0; i < 1024; ++i) {
        if (i >= nstep) break;
        if (u_ray > 1.5 && trans < 0.02) break;
        float meas = sample_vol(u_meas, p);
        if (u_ray < 1.5) {
            float plan = u_diff > 0.5 ? sample_vol(u_plan, p) : 0.0;
            if (u_ray < 0.5) {
                integ += meas * step_mm;
                integ_signed += (meas - plan) * step_mm;
                integ_tot += (meas + plan) * step_mm;
                float wgt = u_diff > 0.5 ? abs(meas - plan) : meas;
                wsum += wgt;
                wpos += wgt * p;
            } else {
                if (u_diff < 0.5 && meas > peak) hit = p;
                peak = max(peak, meas);
                float s = meas - plan;
                float am = abs(s);
                if (am >= peak_abs) {
                    if (u_diff > 0.5) hit = p;
                    peak_abs = am;
                    peak_signed = s;
                }
                peak_tot = max(peak_tot, meas + plan);
            }
        } else {
            float a = 0.0;
            vec3 rgb = vec3(0.0);
            if (u_diff > 0.5) {
                float plan = sample_vol(u_plan, p);
                float tot = meas + plan;
                if (tot >= typical * floor_rel) {
                    float s = meas - plan;
                    ray_lo = min(ray_lo, s);
                    ray_hi = max(ray_hi, s);
                    saw = true;
                    // Opacity stays on the manual window so it never chases u_lo/u_hi.
                    float window = u_absolute > 0.5 ? u_error_scale : u_error_scale * typical;
                    rgb = color_span(s);
                    float n = s / max(window, 1e-12);
                    a = clamp(abs(n), 0.0, 1.0) * clamp(u_gain, 0.0, 1.0)
                        * step_mm / %(depth).5f;
                }
            } else {
                ray_hi = max(ray_hi, meas);
                if (meas > 0.0) saw = true;
                rgb = color_span(meas);
                float tau = clamp(u_gain, 0.0, 1.0) * meas / typical * step_mm / %(depth).5f;
                a = 1.0 - exp(-tau);
            }
            a = clamp(a, 0.0, 1.0);
            col += trans * rgb * a;
            trans *= 1.0 - a;
            if (!half_seen && trans < 0.5) {
                half_seen = true;
                hit = p;
            }
        }
        p += stepv;
    }
    if (u_ray < 1.5) {
        float shown = u_ray < 0.5 ? integ : peak;
        float shown_signed = u_ray < 0.5 ? integ_signed : peak_signed;
        float shown_tot = u_ray < 0.5 ? integ_tot : peak_tot;
        float scale = u_ray < 0.5 ? ray_scale : typical;
        float value = u_diff > 0.5 ? shown_signed : shown;
        float support = u_diff > 0.5 ? shown_tot : shown;
        if (support < scale * floor_rel) discard;
        store_ray(value, value);
        gl_FragColor = vec4(color_span(value), 1.0);
        // A ray that looks like background must not hide the lines behind it.
        float reach = max(max(abs(u_lo), abs(u_hi)), 1e-12);
        if (u_ray < 0.5 && wsum > 0.0) hit = wpos / wsum;
        gl_FragDepth = abs(value) >= 0.1 * reach ? depth_of(hit) : 1.0;
    } else {
        float alpha = 1.0 - trans;
        if (alpha >= 0.02) {
            if (saw) {
                if (u_diff > 0.5) store_ray(ray_lo, ray_hi);
                else store_ray(ray_hi, ray_hi);
            }
            gl_FragColor = vec4(col, alpha);
            // Lines behind half opacity are hidden; thinner fog lets them draw on top.
            gl_FragDepth = half_seen ? depth_of(hit) : 1.0;
        } else discard;
    }
}
"""


def _box_frag() -> str:
    return _BOX_FRAG % {"floor": DOSE_FLOOR, "depth": VIEW_DEPTH_MM}


def box_triangles(origin, shape) -> np.ndarray:
    """Outward faces of the voxel grid, in beam millimetres."""
    o = np.asarray(origin, dtype=np.float32).reshape(3)
    h = o + np.asarray(shape, dtype=np.float32).reshape(3)
    c = np.array(
        [
            [o[0], o[1], o[2]],
            [h[0], o[1], o[2]],
            [h[0], h[1], o[2]],
            [o[0], h[1], o[2]],
            [o[0], o[1], h[2]],
            [h[0], o[1], h[2]],
            [h[0], h[1], h[2]],
            [o[0], h[1], h[2]],
        ],
        dtype=np.float32,
    )
    faces = (
        (4, 5, 6), (4, 6, 7),
        (1, 0, 3), (1, 3, 2),
        (0, 4, 7), (0, 7, 3),
        (5, 1, 2), (5, 2, 6),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
    )
    return np.vstack([c[list(tri)] for tri in faces])


def _gl_at_least(major: int, minor: int) -> bool:
    from OpenGL.GL import GL_VERSION, glGetString

    raw = glGetString(GL_VERSION)
    if not raw:
        return False
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    head = text.split()[0].split(".")
    try:
        got = (int(head[0]), int(head[1]))
    except (IndexError, ValueError):
        return False
    return got >= (major, minor)


# Positive-float bits stay ordered if the sign bit is flipped. These two
# values are ordered(+inf) and ordered(-inf); the reduce shader hardcodes them.
ORDERED_POS_INF = 0xFF800000
ORDERED_NEG_INF = 0x007FFFFF


def ordered_from_float(value: float) -> int:
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return (bits | 0x80000000) & 0xFFFFFFFF


def float_from_ordered(ordered: int) -> float:
    bits = int(ordered) & 0xFFFFFFFF
    raw = (bits & 0x7FFFFFFF) if bits & 0x80000000 else ((~bits) & 0xFFFFFFFF)
    return float(np.array(raw, dtype=np.uint32).view(np.float32))


def auto_color_range(
    difference: bool, lo: float, hi: float, floor: float = 1e-9,
) -> tuple[float, float] | None:
    """Map a GPU min/max to the color endpoints.

    Dose keeps zero at the bottom. Difference is symmetric so 0 is the map's center,
    and never narrower than ``floor`` so float noise does not fill the scale.
    """
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return None
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    if not difference:
        if hi_f <= 0.0:
            return None
        return 0.0, hi_f
    reach = max(abs(lo_f), abs(hi_f), float(floor), 1e-9)
    return -reach, reach


def manual_color_limits(
    *,
    difference: bool,
    transparent: bool,
    integral: bool,
    gain: float,
    typical: float,
    ray_scale: float,
    error_scale: float,
    absolute: bool,
) -> tuple[float, float]:
    """Color endpoints before a GPU readback, matching the marcher."""
    gain_v = max(float(gain), 0.01)
    cell = max(float(typical), 1e-8)
    ray = max(float(ray_scale), 1e-8)
    if difference:
        scale = cell if transparent or not integral else ray
        window = float(error_scale) if absolute else float(error_scale) * scale
        return -max(window, 1e-8), max(window, 1e-8)
    if transparent:
        return 0.0, cell
    return 0.0, (ray if integral else cell) / gain_v


def suggest_abs_window(peak: float) -> float:
    """1-2-5 full scale near *peak*, in the ray's own units.

    A fixed 0.2 MU is enormous next to an MU/mm² line integral, and the
    middle of a divergent map is white, so the whole volume washes out.
    """
    peak_v = abs(float(peak))
    if not math.isfinite(peak_v) or peak_v <= 0.0:
        return 0.2
    exp = math.floor(math.log10(peak_v))
    frac = peak_v / 10.0 ** exp
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * 10.0 ** exp


_RAY_REDUCE = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 4, rg32f) readonly uniform image2D ray_image;
layout(std430, binding = 6) buffer Extrema { uint ext[2]; };

uint ordered(float v) {
    uint x = floatBitsToUint(v);
    return (x & 0x80000000u) != 0u ? ~x : (x | 0x80000000u);
}

shared uint smin;
shared uint smax;

void main() {
    if (gl_LocalInvocationIndex == 0u) {
        smin = 0xFF800000u;
        smax = 0x007FFFFFu;
    }
    barrier();
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(ray_image);
    if (gid.x < size.x && gid.y < size.y) {
        vec2 v = imageLoad(ray_image, gid).xy;
        if (v.x == v.x && v.y == v.y) {
            atomicMin(smin, ordered(v.x));
            atomicMax(smax, ordered(v.y));
        }
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u) {
        atomicMin(ext[0], smin);
        atomicMax(ext[1], smax);
    }
}
"""


def _link_compute(source: str) -> int:
    from OpenGL.GL import (
        GL_COMPILE_STATUS,
        GL_COMPUTE_SHADER,
        GL_LINK_STATUS,
        glAttachShader,
        glCompileShader,
        glCreateProgram,
        glCreateShader,
        glDeleteShader,
        glGetProgramInfoLog,
        glGetProgramiv,
        glGetShaderInfoLog,
        glGetShaderiv,
        glLinkProgram,
        glShaderSource,
    )

    sid = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(sid, source)
    glCompileShader(sid)
    if not glGetShaderiv(sid, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(sid)
        glDeleteShader(sid)
        raise RuntimeError(log.decode() if isinstance(log, bytes) else log)
    prog = glCreateProgram()
    glAttachShader(prog, sid)
    glLinkProgram(prog)
    glDeleteShader(sid)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        log = glGetProgramInfoLog(prog)
        raise RuntimeError(log.decode() if isinstance(log, bytes) else log)
    return int(prog)


def _ssbo(data: np.ndarray, binding: int) -> int:
    from OpenGL.GL import (
        GL_DYNAMIC_DRAW,
        GL_SHADER_STORAGE_BUFFER,
        glBindBuffer,
        glBindBufferBase,
        glBufferData,
        glGenBuffers,
    )

    buf = int(glGenBuffers(1))
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)
    return buf


def _delete_buffers(bufs) -> None:
    from OpenGL.GL import glDeleteBuffers

    live = [int(b) for b in bufs if b]
    if live:
        glDeleteBuffers(len(live), live)


class _ComputeLib:
    def __init__(self) -> None:
        self.count = _link_compute(_COUNT)
        self.scan = _link_compute(_SCAN)
        self.fill = _link_compute(_FILL)
        self.brick = _link_compute(_BRICK)

    def _u(self, prog, name, kind, *vals) -> None:
        from OpenGL.GL import (
            glGetUniformLocation,
            glUniform1f,
            glUniform1i,
            glUniform2i,
            glUniform1ui,
            glUniform3f,
            glUniform3i,
            glUseProgram,
        )

        glUseProgram(prog)
        loc = glGetUniformLocation(prog, name)
        if loc < 0:
            return
        if kind == "1i":
            glUniform1i(loc, int(vals[0]))
        elif kind == "1ui":
            glUniform1ui(loc, int(vals[0]) & 0xFFFFFFFF)
        elif kind == "1f":
            glUniform1f(loc, float(vals[0]))
        elif kind == "2i":
            glUniform2i(loc, int(vals[0]), int(vals[1]))
        elif kind == "3i":
            glUniform3i(loc, int(vals[0]), int(vals[1]), int(vals[2]))
        elif kind == "3f":
            glUniform3f(loc, float(vals[0]), float(vals[1]), float(vals[2]))


_GPU_FILL_FAILED = False
_GPU_GAMMA_FAILED = False


def _lib(canvas) -> _ComputeLib:
    """Compute programs for *canvas*'s GL context; program names do not cross contexts."""
    shared = canvas.context.shared
    lib = getattr(shared, "_scan_compute", None)
    if lib is None:
        lib = shared._scan_compute = _ComputeLib()
    return lib


def _spot_array(centers, sigmas, weights, energy=None, kernel=None) -> np.ndarray:
    """Spots as ``(n, 12)``: c, s, k vec4s (see ``_BRICK``). A layer kernel replaces the z Gaussian."""
    c = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    s = np.maximum(np.asarray(sigmas, dtype=np.float32).reshape(-1, 3), 1e-6)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    n = min(c.shape[0], s.shape[0], w.shape[0])
    out = np.zeros((n, 12), dtype=np.float32)
    out[:, 0:3] = c[:n]
    out[:, 3] = w[:n]
    out[:, 4:7] = s[:n]
    out[:, 8:10] = s[:n, 0:2]
    if kernel is None:
        out[:, 10] = c[:n, 2] - 4.0 * s[:n, 2]
        out[:, 11] = c[:n, 2] + 4.0 * s[:n, 2]
        return out
    e = np.asarray(energy, dtype=float).reshape(-1)[:n]
    sc, ss = kernel.span(c[:n], s[:n], e)
    out[:, 0:3] = sc
    out[:, 4:7] = ss
    layer = kernel.tables.layer_of(e)
    out[:, 7] = layer
    out[:, 10] = kernel.tables.zmin[layer]
    out[:, 11] = kernel.tables.zmax[layer]
    return out


def _layer_texture(tables) -> int:
    """Upload ``(cdf, mcs)`` as an RG32F 2D texture, one row per layer."""
    from OpenGL.GL import (
        GL_CLAMP_TO_EDGE,
        GL_FLOAT,
        GL_NEAREST,
        GL_RG,
        GL_RG32F,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_UNPACK_ALIGNMENT,
        glBindTexture,
        glGenTextures,
        glPixelStorei,
        glTexImage2D,
        glTexParameteri,
    )

    data = np.ascontiguousarray(np.stack([tables.cdf, tables.mcs], axis=-1), dtype=np.float32)
    layers, nodes = tables.cdf.shape
    tex = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, tex)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, nodes, layers, 0, GL_RG, GL_FLOAT, data)
    for pname, val in (
        (GL_TEXTURE_MIN_FILTER, GL_NEAREST),
        (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
        (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
        (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
    ):
        glTexParameteri(GL_TEXTURE_2D, pname, val)
    return tex


def _release_program(canvas) -> None:
    """Unbind a raw GL program and tell vispy, which caches the current one.

    vispy skips ``glUseProgram`` when its cache matches, so a stale cache sends
    the next uniform to the wrong program and the GL error aborts the frame.
    """
    from OpenGL.GL import glUseProgram

    glUseProgram(0)
    canvas.context.shared.parser.env["current_program"] = 0


def _gloo_handle(canvas, tex) -> int:
    canvas.set_current()
    if tex.glir is not canvas.context.glir:
        tex.glir.flush(canvas.context.shared.parser)
    canvas.context.flush_commands()
    ob = canvas.context.shared.parser.get_object(tex.id)
    handle = int(getattr(ob, "handle", 0) or 0) if ob is not None else 0
    if handle <= 0:
        raise RuntimeError("dose texture has no GL name")
    return handle


def _alloc_texture(shape_zyx):
    from vispy import gloo

    tex = gloo.Texture3D(
        shape=tuple(int(v) for v in shape_zyx),
        format="red",
        internalformat="r32f",
        interpolation="nearest",
        wrapping="clamp_to_edge",
    )
    return tex


def gpu_fill_texture(canvas, texture, spots: np.ndarray, grid: DoseGrid, tables=None) -> None:
    """Write *texture* from :func:`_spot_array` rows. Raises if compute cannot run."""
    from OpenGL.GL import (
        GL_ALL_BARRIER_BITS,
        GL_R32F,
        GL_RED,
        GL_SHADER_STORAGE_BUFFER,
        GL_TEXTURE_3D,
        GL_TRUE,
        GL_WRITE_ONLY,
        glBindBuffer,
        glBindImageTexture,
        glBindTexture,
        glDispatchCompute,
        glGetBufferSubData,
        glMemoryBarrier,
        glTexImage3D,
        glTexParameteri,
        GL_TEXTURE_MIN_FILTER,
        GL_TEXTURE_MAG_FILTER,
        GL_NEAREST,
        GL_FLOAT,
        GL_TEXTURE_WRAP_S,
        GL_TEXTURE_WRAP_T,
        GL_TEXTURE_WRAP_R,
        GL_CLAMP_TO_EDGE,
    )

    if not _gl_at_least(4, 3):
        raise RuntimeError("OpenGL 4.3 compute is unavailable")
    from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glDeleteTextures

    nspots = int(spots.shape[0])
    capacity = tile_slot_capacity(spots[:, 0:3], spots[:, 4:7], grid) + nspots
    if nspots == 0:
        spots = np.zeros((1, 12), dtype=np.float32)
    nx, ny, nz = grid.shape
    ntx, nty = tile_shape(grid)
    ntiles = ntx * nty
    if ntiles > 1024:
        raise RuntimeError("tile scan is one workgroup (1024)")
    handle = _gloo_handle(canvas, texture)
    voxel = float(grid.voxel)
    bufs: list[int] = []
    layer_tex = 0
    try:
        if tables is not None:
            glActiveTexture(GL_TEXTURE2)
            layer_tex = _layer_texture(tables)
            glActiveTexture(GL_TEXTURE0)
        # vispy may have allocated the float texture as the wrong type.
        glBindTexture(GL_TEXTURE_3D, handle)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, nx, ny, nz, 0, GL_RED, GL_FLOAT, None)
        for pname, val in (
            (GL_TEXTURE_MIN_FILTER, GL_NEAREST),
            (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
            (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
            (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_3D, pname, val)

        lib = _lib(canvas)
        origin = np.asarray(grid.origin, dtype=np.float32).reshape(3)
        counts = np.zeros(ntiles, dtype=np.uint32)
        bufs = [
            _ssbo(np.ascontiguousarray(spots), 0),
            _ssbo(counts, 1),
            _ssbo(np.zeros(ntiles, dtype=np.uint32), 2),
            _ssbo(np.zeros(capacity, dtype=np.uint32), 3),
            _ssbo(np.zeros(ntiles, dtype=np.uint32), 4),
        ]
        if nspots:
            lib._u(lib.count, "u_origin", "3f", origin[0], origin[1], origin[2])
            lib._u(lib.count, "u_ntiles", "2i", ntx, nty)
            lib._u(lib.count, "u_nspots", "1i", nspots)
            lib._u(lib.count, "u_voxel", "1f", voxel)
            glDispatchCompute((nspots + 63) // 64, 1, 1)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)
            lib._u(lib.scan, "u_n", "1i", ntiles)
            glDispatchCompute(1, 1, 1)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[1])
            raw_counts = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ntiles * 4)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufs[2])
            raw_offsets = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ntiles * 4)
            got_counts = np.frombuffer(bytes(raw_counts), dtype=np.uint32)
            got_offsets = np.frombuffer(bytes(raw_offsets), dtype=np.uint32)
            total = int(got_offsets[-1]) + int(got_counts[-1])
            if total > capacity:
                raise RuntimeError(f"tile list {total} exceeds {capacity}")
            lib._u(lib.fill, "u_origin", "3f", origin[0], origin[1], origin[2])
            lib._u(lib.fill, "u_ntiles", "2i", ntx, nty)
            lib._u(lib.fill, "u_nspots", "1i", nspots)
            lib._u(lib.fill, "u_capacity", "1i", capacity)
            lib._u(lib.fill, "u_voxel", "1f", voxel)
            glDispatchCompute((nspots + 63) // 64, 1, 1)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)
        glBindImageTexture(1, handle, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F)
        lib._u(lib.brick, "u_origin", "3f", origin[0], origin[1], origin[2])
        lib._u(lib.brick, "u_shape", "3i", nx, ny, nz)
        lib._u(lib.brick, "u_ntiles", "2i", ntx, nty)
        lib._u(lib.brick, "u_voxel", "1f", voxel)
        lib._u(lib.brick, "u_table", "1i", 1 if layer_tex else 0)
        lib._u(lib.brick, "u_layers", "1i", 2)
        lib._u(lib.brick, "u_nodes", "1i", tables.cdf.shape[1] if layer_tex else 1)
        glDispatchCompute((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)
    finally:
        glBindImageTexture(1, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glBindTexture(GL_TEXTURE_3D, 0)
        if layer_tex:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, 0)
            glDeleteTextures(1, [layer_tex])
            glActiveTexture(GL_TEXTURE0)
        _delete_buffers(bufs)
        _release_program(canvas)


def read_texture(canvas, texture, shape_zyx) -> np.ndarray:
    """Read an R32F 3D texture back as ``(nz, ny, nx)``."""
    from OpenGL.GL import GL_FLOAT, GL_PACK_ALIGNMENT, GL_RED, GL_TEXTURE_3D
    from OpenGL.GL import glBindTexture, glGetTexImage, glPixelStorei

    handle = _gloo_handle(canvas, texture)
    glBindTexture(GL_TEXTURE_3D, handle)
    try:
        glPixelStorei(GL_PACK_ALIGNMENT, 4)
        raw = glGetTexImage(GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT)
    finally:
        glBindTexture(GL_TEXTURE_3D, 0)
    return np.asarray(raw, dtype=np.float32).reshape(tuple(int(v) for v in shape_zyx))


def gamma_texture(canvas, ref_tex, evl_tex, out_tex, grid: DoseGrid, criteria: GammaCriteria):
    """Write the γ map of *ref_tex* against *evl_tex* into *out_tex*.

    Global normalization to the evaluated maximum. Returns ``(passed, evaluated)``.
    """
    global _GPU_GAMMA_FAILED
    canvas.set_current()
    nx, ny, nz = grid.shape
    evl = read_texture(canvas, evl_tex, (nz, ny, nx))
    ref = read_texture(canvas, ref_tex, (nz, ny, nx))
    vol = None
    if not _GPU_GAMMA_FAILED:
        from ..gpu.gamma import gamma_volume

        try:
            vol, passed, evaluated = gamma_volume(ref, evl, grid.voxel, criteria)
        except Exception as exc:
            _GPU_GAMMA_FAILED = True
            _log.warning("GPU gamma unavailable (%s); using the Python search", exc)
    if vol is None:
        vol, passed, evaluated = gamma_index(ref, evl, grid.voxel, criteria)
    out_tex.set_data(np.ascontiguousarray(vol))
    canvas.context.flush_commands()
    return passed, evaluated


def upload_volume(canvas, texture, volume: np.ndarray, voxel: float = 1.0) -> None:
    """CPU fallback. *volume* is ``(nz, ny, nx)`` MU per cell; the texture holds MU / mm³."""
    data = np.ascontiguousarray(volume / float(voxel) ** 3, dtype=np.float32)
    texture.set_data(data)
    canvas.context.flush_commands()


def fill_texture(
    canvas,
    texture,
    centers,
    sigmas,
    weights,
    energy,
    kernel,
    grid: DoseGrid,
) -> bool:
    """Fill *texture*. Returns True when the compute path did the write."""
    global _GPU_FILL_FAILED
    canvas.set_current()
    layered = isinstance(kernel, LayerDoseKernel)
    if (layered or isinstance(kernel, GaussianSmearKernel)) and not _GPU_FILL_FAILED:
        try:
            spots = _spot_array(centers, sigmas, weights, energy, kernel if layered else None)
            gpu_fill_texture(canvas, texture, spots, grid, kernel.tables if layered else None)
            return True
        except Exception as exc:
            _GPU_FILL_FAILED = True
            _log.warning("GPU dose fill unavailable (%s); using the Python integral", exc)
    volume = deposit_gaussians(centers, sigmas, weights, energy, kernel, grid)
    upload_volume(canvas, texture, volume, grid.voxel)
    return False


def make_dose_box_node():
    """Scene node: back faces store the ray exit, front faces march the volume."""
    from vispy import gloo
    from vispy.scene.visuals import create_visual_node
    from vispy.visuals import Visual

    class DoseBoxVisual(Visual):
        def __init__(self) -> None:
            Visual.__init__(self, vcode=_BOX_VERT, fcode=_box_frag())
            self._verts = gloo.VertexBuffer(np.zeros((36, 3), dtype=np.float32))
            self.shared_program["a_position"] = self._verts
            self._draw_mode = "triangles"
            self.shared_program["u_pass"] = 0
            self.shared_program["u_gain"] = 1.0
            self.shared_program["u_typical"] = 1.0
            self.shared_program["u_error_scale"] = 0.1
            self.shared_program["u_diff"] = 0.0
            self.shared_program["u_absolute"] = 0.0
            self.shared_program["u_ray"] = 0.0
            self.shared_program["u_ray_scale"] = 1.0
            self.shared_program["u_auto"] = 0.0
            self.shared_program["u_lo"] = 0.0
            self.shared_program["u_hi"] = 1.0
            self._auto = False
            self._marching = False
            self._cmap_name = "viridis"
            self._reduce_failed = False
            self._reduce_prog = 0
            self._ray_tex = 0
            self._ray_fbo = 0
            self._ray_size = (0, 0)
            self._ext_buf = 0
            from .dose_volume_fill import colormap_samples

            rgb = colormap_samples("viridis")
            self._cmap = gloo.Texture2D(
                np.ascontiguousarray(rgb.reshape(1, -1, 3), dtype=np.float32),
                interpolation="linear",
                wrapping="clamp_to_edge",
            )
            self.shared_program["u_cmap"] = self._cmap
            self.shared_program["u_origin"] = (0.0, 0.0, 0.0)
            self.shared_program["u_shape"] = (1.0, 1.0, 1.0)
            self.shared_program["u_voxel"] = 1.0
            self.shared_program["u_smooth"] = 1.0
            self.shared_program["u_viewport"] = (1.0, 1.0)
            self._exit = None
            self._fbo = None
            self._fbo_size = (0, 0)
            self._lo = np.zeros(3)
            self._hi = np.ones(3)
            self.set_gl_state(depth_test=False, blend=False, cull_face=False)

        def draw(self):
            # The scene pass must skip the box; draw_volume marches it after.
            # Toggling ``visible`` instead would queue a redraw every frame.
            if self._marching:
                Visual.draw(self)

        @staticmethod
        def _prepare_transforms(view):
            tr = view.transforms
            view.view_program.vert["visual_to_framebuffer"] = tr.get_transform(
                "visual", "framebuffer",
            )
            view.view_program.vert["framebuffer_to_render"] = tr.get_transform(
                "framebuffer", "render",
            )
            view.view_program.frag["visual_to_render"] = tr.get_transform("visual", "render")

        def _compute_bounds(self, axis, view):
            return float(self._lo[axis]), float(self._hi[axis])

        def set_box(self, origin, shape, voxel: float = 1.0) -> None:
            sh = np.asarray(shape, dtype=np.float32).reshape(3)
            extent = sh * np.float32(voxel)
            self._verts.set_data(box_triangles(origin, extent))
            o = np.asarray(origin, dtype=np.float32).reshape(3)
            self._lo = o
            self._hi = o + extent
            self.shared_program["u_origin"] = (float(o[0]), float(o[1]), float(o[2]))
            self.shared_program["u_shape"] = (float(sh[0]), float(sh[1]), float(sh[2]))
            self.shared_program["u_voxel"] = float(voxel)

        def set_interp(self, mode: str) -> None:
            code = {"nearest": 0.0, "cubic": 2.0}.get(mode, 1.0)
            self.shared_program["u_smooth"] = code

        def set_volumes(self, measured, plan) -> None:
            self.shared_program["u_meas"] = measured
            self.shared_program["u_plan"] = plan

        def set_display(
            self,
            *,
            gain: float,
            typical: float,
            error_scale: float,
            difference: bool,
            absolute: bool,
            ray: float = 0.0,
            ray_scale: float = 1.0,
            scale_name: str = "viridis",
            auto: bool = False,
            lo: float = 0.0,
            hi: float = 1.0,
        ) -> None:
            from .dose_volume_fill import colormap_samples

            self.shared_program["u_gain"] = float(gain)
            self.shared_program["u_typical"] = float(typical)
            self.shared_program["u_error_scale"] = float(error_scale)
            self.shared_program["u_diff"] = 1.0 if difference else 0.0
            self.shared_program["u_absolute"] = 1.0 if absolute else 0.0
            self.shared_program["u_ray"] = float(ray)
            self.shared_program["u_ray_scale"] = float(ray_scale)
            self._auto = bool(auto)
            self.shared_program["u_auto"] = 1.0 if auto else 0.0
            self.shared_program["u_lo"] = float(lo)
            self.shared_program["u_hi"] = float(hi)
            if scale_name != self._cmap_name:
                rgb = colormap_samples(scale_name)
                self._cmap.set_data(
                    np.ascontiguousarray(rgb.reshape(1, -1, 3), dtype=np.float32),
                )
                self._cmap_name = scale_name

        def _ensure_exit(self, canvas, width: int, height: int) -> None:
            if self._fbo is not None and self._fbo_size == (width, height):
                return
            self._fbo_size = (width, height)
            self._exit = gloo.Texture2D(
                shape=(height, width, 4),
                format="rgba",
                internalformat="rgba32f",
                interpolation="nearest",
            )
            self._fbo = gloo.FrameBuffer(color=self._exit)
            self.shared_program["u_exit"] = self._exit

        def _mark_reduce_failed(self, exc: Exception) -> None:
            if not self._reduce_failed:
                _log.warning("GPU ray range unavailable (%s); using the manual window", exc)
            self._reduce_failed = True

        def _prepare_ray_image(self, canvas, width: int, height: int) -> None:
            import ctypes

            from OpenGL.GL import (
                GL_CLAMP_TO_EDGE,
                GL_COLOR,
                GL_COLOR_ATTACHMENT0,
                GL_DRAW_FRAMEBUFFER,
                GL_DRAW_FRAMEBUFFER_BINDING,
                GL_FALSE,
                GL_NEAREST,
                GL_RG32F,
                GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                GL_TEXTURE_MIN_FILTER,
                GL_TEXTURE_WRAP_S,
                GL_TEXTURE_WRAP_T,
                GL_WRITE_ONLY,
                glBindFramebuffer,
                glBindImageTexture,
                glBindTexture,
                glClearBufferfv,
                glDeleteFramebuffers,
                glDeleteTextures,
                glDrawBuffer,
                glFramebufferTexture2D,
                glGenFramebuffers,
                glGenTextures,
                glGetIntegerv,
                glTexParameteri,
                glTexStorage2D,
            )

            # The exit-pass pop_fbo is still queued; flush so the saved binding is real.
            canvas.context.flush_commands()
            prev = int(np.asarray(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)).reshape(-1)[0])
            try:
                if self._reduce_prog == 0:
                    self._reduce_prog = _link_compute(_RAY_REDUCE)
                    self._ext_buf = _ssbo(
                        np.array([ORDERED_POS_INF, ORDERED_NEG_INF], dtype=np.uint32),
                        6,
                    )
                if self._ray_size != (width, height):
                    if self._ray_tex:
                        glDeleteTextures(1, [int(self._ray_tex)])
                    if self._ray_fbo:
                        glDeleteFramebuffers(1, [int(self._ray_fbo)])
                    tex = int(glGenTextures(1))
                    glBindTexture(GL_TEXTURE_2D, tex)
                    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, int(width), int(height))
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                    fbo = int(glGenFramebuffers(1))
                    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo)
                    glFramebufferTexture2D(
                        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0,
                    )
                    glBindTexture(GL_TEXTURE_2D, 0)
                    self._ray_tex = tex
                    self._ray_fbo = fbo
                    self._ray_size = (width, height)
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, int(self._ray_fbo))
                glDrawBuffer(GL_COLOR_ATTACHMENT0)
                nan = (ctypes.c_float * 4)(
                    float("nan"), float("nan"), float("nan"), float("nan"),
                )
                glClearBufferfv(GL_COLOR, 0, nan)
            finally:
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prev)
            glBindImageTexture(
                4, int(self._ray_tex), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG32F,
            )

        def _reduce_ray_image(self, canvas, width: int, height: int) -> tuple[float, float]:
            from OpenGL.GL import (
                GL_BUFFER_UPDATE_BARRIER_BIT,
                GL_FALSE,
                GL_READ_ONLY,
                GL_RG32F,
                GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                GL_SHADER_STORAGE_BUFFER,
                glBindBuffer,
                glBindBufferBase,
                glBindImageTexture,
                glBufferSubData,
                glDispatchCompute,
                glGetBufferSubData,
                glMemoryBarrier,
                glUseProgram,
            )

            # The march draw may still sit in vispy's queue.
            canvas.context.flush_commands()
            init = np.array([ORDERED_POS_INF, ORDERED_NEG_INF], dtype=np.uint32)
            try:
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, int(self._ext_buf))
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, init.nbytes, init)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, int(self._ext_buf))
                glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
                glBindImageTexture(
                    4, int(self._ray_tex), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F,
                )
                glUseProgram(int(self._reduce_prog))
                glDispatchCompute((int(width) + 15) // 16, (int(height) + 15) // 16, 1)
                glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, int(self._ext_buf))
                # ponytail: synchronous 8-byte readback stalls one frame of GPU work;
                # a fenced PBO ring would hide it if drags ever stutter.
                raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8)
                vals = np.frombuffer(bytes(raw), dtype=np.uint32)
            finally:
                glBindImageTexture(4, 0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, 0)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
                _release_program(canvas)
            return float_from_ordered(int(vals[0])), float_from_ordered(int(vals[1]))

        def draw_volume(self, canvas) -> tuple[float, float] | None:
            w, h = canvas.physical_size
            width, height = int(w), int(h)
            if width < 2 or height < 2:
                return None
            self._ensure_exit(canvas, width, height)
            self.shared_program["u_viewport"] = (float(width), float(height))
            self._marching = True
            try:
                self.set_gl_state(depth_test=False, blend=False, cull_face="front")
                self.shared_program["u_pass"] = 0
                canvas.push_fbo(self._fbo, (0, 0), (width, height))
                try:
                    canvas.context.clear(color=(0.0, 0.0, 0.0, 0.0), depth=True)
                    self.draw()
                finally:
                    canvas.pop_fbo()
                ready = False
                if self._auto and not self._reduce_failed:
                    try:
                        self._prepare_ray_image(canvas, width, height)
                        ready = True
                    except Exception as exc:
                        self._mark_reduce_failed(exc)
                if self._auto and self._reduce_failed:
                    self.shared_program["u_auto"] = 0.0
                # Always pass, but write the dose depth for the lines drawn next.
                self.set_gl_state(
                    depth_test=True,
                    depth_func="always",
                    blend=True,
                    blend_equation="func_add",
                    blend_func=("one", "one_minus_src_alpha"),
                    cull_face="back",
                )
                self.shared_program["u_pass"] = 1
                self.draw()
            finally:
                self._marching = False
            span = None
            if ready:
                try:
                    span = self._reduce_ray_image(canvas, width, height)
                except Exception as exc:
                    self._mark_reduce_failed(exc)
            return span

    return create_visual_node(DoseBoxVisual)
