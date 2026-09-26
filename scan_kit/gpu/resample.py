"""Trilinear resampling of one voxel grid onto another on the GPU."""

from __future__ import annotations

import functools

import numpy as np

from .core import Kernel, Params, device, read, storage, uniform

_P = Params("R", [
    ("snx", "i"), ("sny", "i"), ("snz", "i"), ("tnx", "i"), ("tny", "i"), ("tnz", "i"),
    ("m00", "f"), ("m01", "f"), ("m02", "f"), ("c0", "f"),
    ("m10", "f"), ("m11", "f"), ("m12", "f"), ("c1", "f"),
    ("m20", "f"), ("m21", "f"), ("m22", "f"), ("c2", "f"),
])

_SOURCE = _P.wgsl + """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> P: R;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) g: vec3u) {
    let t = vec3i(g);
    if (any(t >= vec3i(P.tnx, P.tny, P.tnz))) { return; }
    let q = vec3f(t);
    let p = vec3f(
        P.m00 * q.x + P.m01 * q.y + P.m02 * q.z + P.c0,
        P.m10 * q.x + P.m11 * q.y + P.m12 * q.z + P.c1,
        P.m20 * q.x + P.m21 * q.y + P.m22 * q.z + P.c2,
    );
    let b = floor(p);
    let f = p - b;
    let i = vec3i(b);
    var acc: f32 = 0.0;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let s = i + vec3i(dx, dy, dz);
                if (any(s < vec3i(0)) || any(s >= vec3i(P.snx, P.sny, P.snz))) { continue; }
                let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
                acc += w * src[s.x + P.snx * (s.y + P.sny * s.z)];
            }
        }
    }
    dst[t.x + P.tnx * (t.y + P.tny * t.z)] = acc;
}
"""


@functools.cache
def _kernel() -> Kernel:
    return Kernel(_SOURCE, label="resample")


def resample_affine(src: np.ndarray, shape_zyx, matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """``dst[k, j, i] = src`` trilinearly at source index ``matrix @ (i, j, k) + offset`` (0 outside)."""
    src = np.ascontiguousarray(src, dtype=np.float32)
    snz, sny, snx = src.shape
    tnz, tny, tnx = (int(v) for v in shape_zyx)
    m = np.asarray(matrix, dtype=float)
    c = np.asarray(offset, dtype=float)
    kernel = _kernel()
    params = uniform(_P.nbytes)
    device().queue.write_buffer(params, 0, _P.pack(
        snx=snx, sny=sny, snz=snz, tnx=tnx, tny=tny, tnz=tnz,
        **{f"m{r}{k}": m[r, k] for r in range(3) for k in range(3)}, **{f"c{r}": c[r] for r in range(3)},
    ))
    out = storage(size=4 * tnx * tny * tnz)
    kernel.run(kernel.bind(storage(src), out, params), (tnx + 3) // 4, (tny + 3) // 4, (tnz + 3) // 4)
    return read(out, np.float32).reshape(tnz, tny, tnx)
