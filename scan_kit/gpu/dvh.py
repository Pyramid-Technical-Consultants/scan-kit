"""Dose-volume histograms of many structures in one GPU pass."""

from __future__ import annotations

import functools

import numpy as np

from .core import GpuUnavailable, Kernel, Params, device, read, storage, uniform, workgroups_1d

_P = Params("H", [("n", "u"), ("nroi", "u"), ("nbins", "u"), ("scale", "f")])

_SOURCE = _P.wgsl + """
@group(0) @binding(0) var<storage, read> dose: array<f32>;
@group(0) @binding(1) var<storage, read> bits: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;
// Per structure, the max then (after nroi) the min dose, as f32 bits: non-negative floats order like u32.
@group(0) @binding(3) var<storage, read_write> ext: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> P: H;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= P.n) { return; }
    let m = bits[i];
    if (m == 0u) { return; }
    let d = max(dose[i], 0.0);
    let b = min(u32(d * P.scale), P.nbins - 1u);
    let db = bitcast<u32>(d);
    for (var r = 0u; r < P.nroi; r++) {
        if (((m >> r) & 1u) == 0u) { continue; }
        atomicAdd(&hist[r * P.nbins + b], 1u);
        atomicMax(&ext[r], db);
        atomicMin(&ext[P.nroi + r], db);
    }
}
"""


@functools.cache
def _kernel() -> Kernel:
    return Kernel(_SOURCE, label="dvh")


def histograms(dose, bits, nroi: int, nbins: int, top: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel counts ``(nroi, nbins)`` over ``[0, top)`` Gy (the last bin takes everything above),
    and each structure's exact (max, min) dose; empty structures read 0 and +inf."""
    dose = np.ascontiguousarray(dose, dtype=np.float32).reshape(-1)
    bits = np.ascontiguousarray(bits, dtype=np.uint32).reshape(-1)
    if dose.size != bits.size:
        raise ValueError("dose and structure bits must be the same grid")
    scale = nbins / top if top > 0.0 else 0.0
    try:
        kernel = _kernel()
    except GpuUnavailable:
        return _histograms_cpu(dose, bits, nroi, nbins, scale)
    params = uniform(_P.nbytes)
    device().queue.write_buffer(params, 0, _P.pack(n=dose.size, nroi=nroi, nbins=nbins, scale=scale))
    hist = storage(np.zeros(nroi * nbins, dtype=np.uint32))
    ext = storage(np.r_[np.zeros(nroi, np.uint32), np.full(nroi, 0xFFFFFFFF, np.uint32)])
    kernel.run(kernel.bind(storage(dose), storage(bits), hist, ext, params), *workgroups_1d(dose.size))
    e = read(ext, np.uint32)
    empty = e[nroi:] == 0xFFFFFFFF
    dmin = np.where(empty, np.inf, e[nroi:].view(np.float32).astype(np.float64))
    return read(hist, np.uint32).reshape(nroi, nbins).astype(np.int64), e[:nroi].view(np.float32).astype(np.float64), dmin


def _histograms_cpu(dose, bits, nroi, nbins, scale):
    d = np.maximum(dose, 0.0)
    b = np.minimum((d * scale).astype(np.int64), nbins - 1)
    hist = np.zeros((nroi, nbins), dtype=np.int64)
    dmax, dmin = np.zeros(nroi), np.full(nroi, np.inf)
    for r in range(nroi):
        inside = (bits >> np.uint32(r)) & np.uint32(1) == 1
        if inside.any():
            hist[r] = np.bincount(b[inside], minlength=nbins)
            dmax[r], dmin[r] = float(d[inside].max()), float(d[inside].min())
    return hist, dmax, dmin
