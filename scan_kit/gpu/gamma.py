"""3D global gamma index on the GPU."""

from __future__ import annotations

import functools

import numpy as np

from ..views.dose_volume_physics import GammaCriteria, gamma_offsets
from .core import Kernel, Params, device, read, storage, uniform

_P = Params("G", [("nx", "i"), ("ny", "i"), ("nz", "i"), ("noffs", "i"), ("dd", "f"), ("cut", "f"), ("cap2", "f")])

_SOURCE = _P.wgsl + """
@group(0) @binding(0) var<storage, read> refd: array<f32>;
@group(0) @binding(1) var<storage, read> evl: array<f32>;
@group(0) @binding(2) var<storage, read> offs: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> gam: array<f32>;
@group(0) @binding(4) var<storage, read_write> tally: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> P: G;

fn evl_at(p: vec3f) -> f32 {
    let b = floor(p);
    let f = p - b;
    let i = vec3i(b);
    var acc: f32 = 0.0;
    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let q = i + vec3i(dx, dy, dz);
                if (any(q < vec3i(0)) || any(q >= vec3i(P.nx, P.ny, P.nz))) { continue; }
                let w = select(1.0 - f.x, f.x, dx == 1) * select(1.0 - f.y, f.y, dy == 1) * select(1.0 - f.z, f.z, dz == 1);
                acc += w * evl[q.x + P.nx * (q.y + P.ny * q.z)];
            }
        }
    }
    return acc;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) g: vec3u) {
    let v = vec3i(g);
    if (any(v >= vec3i(P.nx, P.ny, P.nz))) { return; }
    let i = v.x + P.nx * (v.y + P.ny * v.z);
    let r = refd[i];
    // Empty voxels are never scored, or a 0 % cutoff would pass all the air.
    if (r <= 0.0 || r < P.cut) {
        gam[i] = 0.0;
        return;
    }
    // Offsets are sorted by distance, so once distance alone loses, stop.
    var best = P.cap2;
    for (var k = 0; k < P.noffs; k++) {
        let o = offs[k];
        if (o.w >= best) { break; }
        let d = (evl_at(vec3f(v) + o.xyz) - r) / P.dd;
        best = min(best, o.w + d * d);
    }
    let gv = sqrt(best);
    gam[i] = gv;
    atomicAdd(&tally[1], 1u);
    if (gv <= 1.0) { atomicAdd(&tally[0], 1u); }
}
"""


@functools.cache
def _kernel() -> Kernel:
    return Kernel(_SOURCE, label="gamma")


def gamma_volume(ref, evl, spacing_mm, criteria: GammaCriteria, norm: float | None = None):
    """Global gamma of *ref* points searched in *evl*, both ``(nz, ny, nx)``.

    Normalized to *norm*, by default the evaluated maximum. *spacing_mm* is one
    spacing or (x, y, z). Returns ``(gamma, passed, evaluated)`` like ``gamma_index``.
    """
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    evl = np.ascontiguousarray(evl, dtype=np.float32)
    if ref.shape != evl.shape:
        raise ValueError(f"gamma needs matching grids, got {ref.shape} and {evl.shape}")
    nz, ny, nx = ref.shape
    norm = float(evl.max()) if norm is None and evl.size else float(norm or 0.0)
    if norm <= 0.0:
        return np.zeros(ref.shape, dtype=np.float32), 0, 0
    offs = gamma_offsets(spacing_mm, criteria)
    kernel = _kernel()
    params = uniform(_P.nbytes)
    device().queue.write_buffer(params, 0, _P.pack(
        nx=nx, ny=ny, nz=nz, noffs=len(offs), dd=criteria.dose_pct / 100.0 * norm,
        cut=criteria.cutoff_pct / 100.0 * norm, cap2=criteria.cap**2,
    ))
    out = storage(size=ref.nbytes)
    tally = storage(np.zeros(2, dtype=np.uint32))
    kernel.run(kernel.bind(storage(ref), storage(evl), storage(offs), out, tally, params),
               (nx + 3) // 4, (ny + 3) // 4, (nz + 3) // 4)
    passed, evaluated = read(tally, np.uint32)
    return read(out, np.float32).reshape(ref.shape), int(passed), int(evaluated)
