"""GPU Monte Carlo proton dose: a compute-shader port of MCsquare's transport.

The shader (``assets/dose_mc_transport.comp``) runs on MCsquare's own material
tables (``assets/mc_materials.npz``, built by ``scripts/build_mc_tables.py``).
There is no CPU transport: without OpenGL 4.3 compute, :func:`mc_fill_texture`
raises.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from importlib import resources

import numpy as np

from .dose_volume_fill import DoseGrid
from .dose_volume_raycast import (
    _ComputeLib,
    _delete_buffers,
    _gl_at_least,
    _gloo_handle,
    _link_compute,
    _release_program,
    _ssbo,
)

BATCHES = 10  # MCsquare MIN_NUM_BATCH
DISPATCH_HISTORIES = 32768
QUANTUM_MEV = 1e-4
MEV_TO_J = 1.602176634e-13
LEDGER_KEYS = ("incident", "grid", "off_grid", "leaked", "lost")
TYPE_MIXTURE, TYPE_ICRU, TYPE_PP = 0, 1, 2

_FOLD = """
#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(std430, binding = 3) buffer TallyLo { uint tally_lo[]; };
layout(std430, binding = 4) buffer TallyHi { uint tally_hi[]; };
layout(std430, binding = 6) buffer Sum { float dose_sum[]; };
layout(std430, binding = 7) buffer Sq { float dose_sq[]; };
layout(r32f, binding = 1) uniform writeonly image3D u_out;
uniform ivec3 u_dims;
uniform float u_scale;
uniform int u_mode;  // 0 fold a batch, 1 fold the last batch and store, 2 store a preview
uniform float u_gain;

void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(c, u_dims))) return;
    int i = c.x + u_dims.x * (c.y + u_dims.y * c.z);
    double q = double(int(tally_hi[i])) * 4294967296.0LF + double(tally_lo[i]);
    float b = float(q * double(u_scale));
    if (u_mode == 2) {
        imageStore(u_out, c, vec4((dose_sum[i] + b) * u_gain, 0.0, 0.0, 0.0));
        return;
    }
    tally_lo[i] = 0u;
    tally_hi[i] = 0u;
    float s = dose_sum[i] + b;
    dose_sum[i] = s;
    dose_sq[i] += b * b;
    if (u_mode == 1) imageStore(u_out, c, vec4(s, 0.0, 0.0, 0.0));
}
"""


@dataclass(frozen=True)
class McResult:
    histories: int
    # Mean relative batch uncertainty over voxels above half the maximum (MCsquare Process_batch).
    uncertainty: float
    # Ledger energies in MeV per history, keyed by LEDGER_KEYS.
    ledger: dict
    overflow: int
    deepest: int

    def fraction(self, key: str) -> float:
        inc = self.ledger["incident"]
        return self.ledger[key] / inc if inc > 0 else 0.0

    @property
    def closure(self) -> float:
        """(incident - everything accounted for) / incident."""
        inc = self.ledger["incident"]
        if inc <= 0:
            return 0.0
        return (inc - sum(self.ledger[k] for k in LEDGER_KEYS[1:])) / inc


@functools.cache
def load_tables() -> dict:
    path = resources.files("scan_kit") / "assets" / "mc_materials.npz"
    with resources.as_file(path) as p, np.load(p) as z:
        return {k: z[k] for k in z.files}


def mc_media() -> tuple[str, ...]:
    return tuple(str(m) for m in load_tables()["media"])


def pack_tables(t: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flatten the npz into (float32 F, int32 I, #define offsets) for the shader's two tables."""
    nm = len(t["media"])
    stop_bins = t["m_stop"].shape[1]
    nuc_bins = t["m_nuc"].shape[1]
    m_stop = 7
    m_nuc = m_stop + stop_bins
    m_stride = m_nuc + nuc_bins
    media = np.zeros((nm, m_stride))
    media[:, 0:3] = t["m_props"]
    media[:, 3:7] = t["m_frac"]
    media[:, m_stop:m_nuc] = t["m_stop"]
    media[:, m_nuc:] = t["m_nuc"]

    parts = [("M_BASE", media.ravel()), ("F_EA", t["e_A"]), ("F_ELE", t["el_E"]), ("F_ELS", t["el_sigma"]),
             ("F_ELC", t["el_cdf"].ravel()), ("F_INE", t["in_E"]), ("F_INS", t["in_sigma"]),
             ("F_INM", t["in_mult"].ravel()), ("F_INR", t["in_recoil"]), ("F_SE", t["sec_E"]),
             ("F_SD", t["sec_cD"]), ("F_SDD", t["sec_cDD"].ravel())]
    defines = {}
    at = 0
    for name, arr in parts:
        defines[name] = at
        at += arr.size
    F = np.concatenate([a for _, a in parts]).astype(np.float32)

    ne = len(t["elements"])
    m_int = np.zeros((nm, 6), dtype=np.int32)
    m_int[:, 0] = t["m_type"]
    m_int[:, 1] = (t["m_comp"] >= 0).sum(axis=1)
    m_int[:, 2:6] = t["m_comp"]
    el = np.zeros((ne, 5), dtype=np.int32)
    el[:, 0] = t["e_type"]
    el[:, 1] = t["el_start"][:-1]
    el[:, 2] = np.diff(t["el_start"])
    el[:, 3] = t["in_start"][:-1]
    el[:, 4] = np.diff(t["in_start"])
    iparts = [("I_M", m_int.ravel()), ("I_EL", el.ravel()),
              ("I_RS", t["in_rows_start"].ravel()), ("I_RN", t["in_rows_n"].ravel())]
    at = 0
    for name, arr in iparts:
        defines[name] = at
        at += arr.size
    ints = np.concatenate([a for _, a in iparts]).astype(np.int32)

    defines.update(
        M_STRIDE=m_stride, M_RHO=0, M_NEL=1, M_X0=2, M_FRAC=3, M_STOP=m_stop, M_NUC=m_nuc,
        STOP_BINS=stop_bins, NUC_BINS=nuc_bins, TYPE_ICRU=TYPE_ICRU, TYPE_PP=TYPE_PP,
        WATER_MEDIUM=list(t["media"]).index("water"),
    )
    return F, ints, defines


def transport_source(defines: dict) -> str:
    text = (resources.files("scan_kit") / "assets" / "dose_mc_transport.comp").read_text(encoding="utf-8")
    head, _, body = text.partition("\n")
    lines = "".join(f"#define {k} {v}\n" for k, v in defines.items())
    return f"{head}\n{lines}{body}"


class _McLib(_ComputeLib):
    def __init__(self) -> None:
        self.F, self.ints, defines = pack_tables(load_tables())
        self.transport = _link_compute(transport_source(defines))
        self.fold = _link_compute(_FOLD)


def _mc_lib(canvas) -> _McLib:
    shared = canvas.context.shared
    lib = getattr(shared, "_scan_mc", None)
    if lib is None:
        lib = shared._scan_mc = _McLib()
    return lib


def batch_uncertainty(dose_sum: np.ndarray, dose_sq: np.ndarray, batches: int = BATCHES) -> float:
    """MCsquare ``Process_batch``: mean relative σ over voxels above half the maximum."""
    s = np.asarray(dose_sum, dtype=float)
    q = np.asarray(dose_sq, dtype=float)
    peak = float(s.max()) if s.size else 0.0
    if peak <= 0.0:
        return 0.0
    # ponytail: plain 0.5 x max; MCsquare eases its max upward to damp noise.
    keep = s > 0.5 * peak
    n = float(batches)
    rel = np.sqrt(np.maximum(n * (q[keep] * n / (s[keep] ** 2) - 1.0), 0.0)) / n
    return float(rel.mean())


def _read_buffer(buf: int, count: int, dtype) -> np.ndarray:
    from OpenGL.GL import GL_SHADER_STORAGE_BUFFER, glBindBuffer, glGetBufferSubData

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf)
    raw = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * np.dtype(dtype).itemsize)
    return np.frombuffer(bytes(raw), dtype=dtype).copy()


class McRun:
    """Transport *histories* protons into *tex* (``grid``-shaped r32f Gy), a slice at a time.

    Spots are nozzle values in scene mm and MeV: centers (x, y), sigmas (sx, sy),
    energy, and delivered protons. The phantom is *medium_key* from ``z = 0`` to
    ``-depth`` mm, laterally infinite, behind *wet* mm of water.

    :meth:`step` transports for about a time budget and :meth:`preview` writes the
    running estimate, so the GUI can draw in between. The finished texture and
    :attr:`result` don't depend on how the run was sliced.
    """

    def __init__(
        self,
        canvas,
        tex,
        x,
        y,
        sx,
        sy,
        energy,
        protons,
        medium_key: str,
        grid: DoseGrid,
        *,
        depth: float,
        wet: float,
        spread_pct: float,
        histories: int,
        seed: int,
    ) -> None:
        from OpenGL.GL import (
            GL_CLAMP_TO_EDGE,
            GL_FLOAT,
            GL_MAX_SHADER_STORAGE_BLOCK_SIZE,
            GL_NEAREST,
            GL_R32F,
            GL_RED,
            GL_TEXTURE_3D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_R,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            glBindTexture,
            glGetIntegerv,
            glTexImage3D,
            glTexParameteri,
        )

        self.result: McResult | None = None
        self._bufs: list[int] = []
        self._canvas = canvas
        canvas.set_current()
        if not _gl_at_least(4, 3):
            raise RuntimeError("OpenGL 4.3 compute is unavailable")
        t = load_tables()
        media = [str(m) for m in t["media"]]
        if medium_key not in media:
            raise ValueError(f"no MCsquare material data for {medium_key!r}")
        medium = media.index(medium_key)
        nx, ny, nz = (int(v) for v in grid.shape)
        self._dims = (nx, ny, nz)
        nvox = self._nvox = nx * ny * nz
        limit = int(glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE))
        if nvox * 4 > limit:
            raise RuntimeError(f"grid needs {nvox * 4} bytes per buffer; the GPU allows {limit}")

        e = np.asarray(energy, dtype=float).reshape(-1)
        w = np.asarray(protons, dtype=float).reshape(-1)
        cols = [np.asarray(v, dtype=float).reshape(-1) for v in (x, y, sx, sy)]
        ok = np.isfinite(e) & np.isfinite(w) & (w > 0.0) & (e > 0.0)
        for c in cols:
            ok &= np.isfinite(c)
        total = float(w[ok].sum())
        self.histories = max(int(histories) // BATCHES, 1) * BATCHES
        self._per_batch = self.histories // BATCHES
        self._next = 0  # histories dispatched so far
        self._chunk = DISPATCH_HISTORIES

        self._handle = _gloo_handle(canvas, tex)
        glBindTexture(GL_TEXTURE_3D, self._handle)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, nx, ny, nz, 0, GL_RED, GL_FLOAT, None)
        for pname, val in (
            (GL_TEXTURE_MIN_FILTER, GL_NEAREST),
            (GL_TEXTURE_MAG_FILTER, GL_NEAREST),
            (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE),
            (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
            (GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE),
        ):
            glTexParameteri(GL_TEXTURE_3D, pname, val)
        glBindTexture(GL_TEXTURE_3D, 0)
        if total <= 0.0:
            tex.set_data(np.zeros((nz, ny, nx), dtype=np.float32))
            canvas.context.flush_commands()
            self.result = McResult(0, 0.0, {k: 0.0 for k in LEDGER_KEYS}, 0, 0)
            return

        n = int(ok.sum())
        spots = np.zeros((n, 8), dtype=np.float32)
        for k, c in enumerate(cols):
            spots[:, k] = c[ok] / 10.0
        spots[:, 2:4] = np.maximum(spots[:, 2:4], 0.0)
        spots[:, 4] = e[ok]
        spots[:, 5] = max(float(spread_pct), 0.0) / 100.0 * e[ok]
        spots[:, 6] = np.cumsum(w[ok]) / total
        spots[-1, 6] = 1.0

        medium_rho = float(t["m_props"][medium, 0])
        voxel_cm = float(grid.voxel) / 10.0
        per_history = total / self.histories
        gy_per_quantum = QUANTUM_MEV * per_history * MEV_TO_J / (medium_rho * voxel_cm**3 * 1e-3)
        origin = np.asarray(grid.origin, dtype=float) / 10.0
        # The programs are shared by every run, so each slice sets its own uniforms.
        self._tp_uniforms = (
            ("u_origin", "3f", *origin), ("u_voxel", "1f", voxel_cm), ("u_dims", "3i", nx, ny, nz),
            ("u_depth", "1f", max(float(depth), 0.0) / 10.0), ("u_wet", "1f", max(float(wet), 0.0) / 10.0),
            ("u_medium", "1i", medium), ("u_nspots", "1i", n), ("u_seed", "1ui", seed),
        )
        self._fp_uniforms = (("u_dims", "3i", nx, ny, nz), ("u_scale", "1f", gy_per_quantum))

        self._lib = lib = _mc_lib(canvas)
        try:
            for binding, data in enumerate((
                spots, lib.F, lib.ints,
                np.zeros(nvox, dtype=np.uint32), np.zeros(nvox, dtype=np.uint32), np.zeros(12, dtype=np.uint32),
                np.zeros(nvox, dtype=np.float32), np.zeros(nvox, dtype=np.float32),
            )):
                self._bufs.append(_ssbo(data, binding))
        except Exception:
            self.close()
            raise
        finally:
            self._unbind()

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def progress(self) -> float:
        """Share of the histories transported."""
        return 1.0 if self.done else self._next / self.histories

    def _bind(self) -> None:
        from OpenGL.GL import GL_R32F, GL_SHADER_STORAGE_BUFFER, GL_TRUE, GL_WRITE_ONLY, glBindBufferBase, glBindImageTexture

        self._canvas.set_current()
        for binding, buf in enumerate(self._bufs):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf)
        glBindImageTexture(1, self._handle, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F)
        lib = self._lib
        for prog, uniforms in ((lib.transport, self._tp_uniforms), (lib.fold, self._fp_uniforms)):
            for name, kind, *vals in uniforms:
                lib._u(prog, name, kind, *vals)

    def _unbind(self) -> None:
        from OpenGL.GL import GL_R32F, GL_SHADER_STORAGE_BUFFER, GL_TRUE, GL_WRITE_ONLY, glBindBuffer, glBindImageTexture

        glBindImageTexture(1, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        _release_program(self._canvas)

    def _fold(self, mode: int, gain: float = 1.0) -> None:
        from OpenGL.GL import GL_ALL_BARRIER_BITS, glDispatchCompute, glMemoryBarrier

        lib = self._lib
        lib._u(lib.fold, "u_mode", "1i", mode)
        lib._u(lib.fold, "u_gain", "1f", gain)
        nx, ny, nz = self._dims
        glMemoryBarrier(GL_ALL_BARRIER_BITS)
        glDispatchCompute((nx + 7) // 8, (ny + 7) // 8, (nz + 7) // 8)
        glMemoryBarrier(GL_ALL_BARRIER_BITS)

    def step(self, budget_s: float = math.inf) -> bool:
        """Transport for about *budget_s* seconds; True once the run is finished."""
        from OpenGL.GL import glDispatchCompute, glFinish

        if self.done:
            return True
        lib = self._lib
        end = time.perf_counter() + budget_s
        self._bind()
        try:
            while True:
                start = self._next % self._per_batch
                count = min(self._chunk, self._per_batch - start)
                t0 = time.perf_counter()
                lib._u(lib.transport, "u_hist_base", "1ui", self._next)
                lib._u(lib.transport, "u_nhist", "1ui", count)
                glDispatchCompute((count + 63) // 64, 1, 1)
                # Short submissions keep each one well under the Windows GPU timeout.
                glFinish()
                self._next += count
                if math.isfinite(budget_s):
                    # Integer tallies add in any order, so the slice size never changes the dose.
                    took = time.perf_counter() - t0
                    if took > 0.6 * budget_s:
                        self._chunk = max(self._chunk // 2, 1024)
                    elif took < 0.2 * budget_s:
                        self._chunk = min(self._chunk * 2, DISPATCH_HISTORIES)
                if self._next % self._per_batch == 0:
                    last = self._next == self.histories
                    self._fold(1 if last else 0)
                    if last:
                        self._finish()
                        return True
                if time.perf_counter() >= end:
                    return False
        finally:
            self._unbind()

    def preview(self) -> None:
        """Write the dose so far, scaled up to the full run, into the texture."""
        if self.done or self._next == 0:
            return
        self._bind()
        try:
            self._fold(2, self.histories / self._next)
        finally:
            self._unbind()

    def _finish(self) -> None:
        raw = _read_buffer(self._bufs[5], 12, np.uint32)
        dose_sum = _read_buffer(self._bufs[6], self._nvox, np.float32)
        dose_sq = _read_buffer(self._bufs[7], self._nvox, np.float32)
        self.close()
        pairs = raw[:10].view(np.int64)  # (lo, hi) little-endian pairs
        ledger = {k: float(pairs[i]) * QUANTUM_MEV / self.histories for i, k in enumerate(LEDGER_KEYS)}
        self.result = McResult(
            histories=self.histories,
            uncertainty=batch_uncertainty(dose_sum, dose_sq),
            ledger=ledger,
            overflow=int(raw[10]),
            deepest=int(raw[11]),
        )

    def close(self) -> None:
        """Free the GPU buffers; a run closed before it finishes leaves its last preview."""
        if self._bufs:
            self._canvas.set_current()
            _delete_buffers(self._bufs)
            self._bufs = []


def mc_fill_texture(canvas, tex, *args, **kwargs) -> McResult:
    """:class:`McRun` to completion in one call."""
    run = McRun(canvas, tex, *args, **kwargs)
    run.step()
    return run.result


def mc_note(result: McResult) -> str:
    """`` · MC 1.0M · ±0.8 % · 0.4 % off-grid`` for the volume label."""
    h = result.histories
    count = f"{h / 1e6:.1f}M" if h >= 1e6 else f"{h / 1e3:.0f}k"
    return f" · MC {count} · ±{100.0 * result.uncertainty:.1f} % · {100.0 * result.fraction('off_grid'):.1f} % off-grid"
