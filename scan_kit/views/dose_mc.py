"""GPU Monte Carlo proton dose: a WebGPU port of MCsquare's transport.

The WGSL kernel (``assets/mc_transport.wgsl``) runs on MCsquare's own material
tables (``assets/mc_materials.npz``, built by ``scripts/build_mc_tables.py``).
There is no CPU transport: without a WebGPU adapter, :class:`McRun` raises
:class:`~scan_kit.gpu.GpuUnavailable`.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from importlib import resources

import numpy as np

from ..gpu import Kernel, Params, device, read, storage, uniform, wait, wgsl_asset, workgroups_1d
from .dose_volume_fill import DoseGrid

BATCHES = 10  # MCsquare MIN_NUM_BATCH
# Batches stay this size as a run grows, so its uncertainty estimate survives a raised target.
BATCH_HISTORIES = 100_000
# Whole batches: every dispatch waits on its slowest history, so a batch split in two pays that twice.
DISPATCH_HISTORIES = BATCH_HISTORIES
QUANTUM_MEV = 1e-4
MEV_TO_J = 1.602176634e-13
LEDGER_KEYS = ("incident", "grid", "off_grid", "leaked", "lost", "beamline")
TYPE_MIXTURE, TYPE_ICRU, TYPE_PP = 0, 1, 2
MODE_SLAB, MODE_PATIENT = 0, 1
FLAG_DOSE_TO_WATER, FLAG_LET = 1, 2
SIMPLE_STRIDE, BDL_STRIDE, BEAM_STRIDE = 8, 40, 16

_TRANSPORT = Params("Params", [
    ("mode", "i"), ("flags", "u"), ("ox", "f"), ("oy", "f"), ("oz", "f"), ("vx", "f"), ("vy", "f"), ("vz", "f"),
    ("nx", "i"), ("ny", "i"), ("nz", "i"), ("depth", "f"), ("wet", "f"), ("medium", "i"), ("nspots", "i"),
    ("hist_base", "u"), ("nhist", "u"), ("seed", "u"),
])
_FOLD_P = Params("Fold", [("n", "u"), ("mode", "i"), ("scale", "f"), ("gain", "f")])
_FOLD = _FOLD_P.wgsl + """
@group(0) @binding(0) var<storage, read_write> tally: array<u32>;
@group(0) @binding(1) var<storage, read_write> dose_sum: array<f32>;
@group(0) @binding(2) var<storage, read_write> dose_sq: array<f32>;
@group(0) @binding(3) var<storage, read_write> dose_out: array<f32>;
@group(0) @binding(4) var<uniform> P: Fold;

// mode 0 folds a batch, 1 folds the last batch and stores, 2 stores a preview.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = g.x + g.y * nwg.x * 256u;
    if (i >= P.n) { return; }
    let b = (f32(bitcast<i32>(tally[2u * i + 1u])) * 4294967296.0 + f32(tally[2u * i])) * P.scale;
    if (P.mode == 2) {
        dose_out[i] = (dose_sum[i] + b) * P.gain;
        return;
    }
    tally[2u * i] = 0u;
    tally[2u * i + 1u] = 0u;
    let s = dose_sum[i] + b;
    dose_sum[i] = s;
    dose_sq[i] += b * b;
    if (P.mode == 1) { dose_out[i] = s * P.gain; }
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
    """Flatten the npz into (float32 F, int32 I, offsets) for the kernel's two tables."""
    nm = len(t["media"])
    width = t["m_comp"].shape[1]
    stop_bins = t["m_stop"].shape[1]
    nuc_bins = t["m_nuc"].shape[1]
    m_stop = 3 + width
    m_nuc = m_stop + stop_bins
    m_stride = m_nuc + nuc_bins
    media = np.zeros((nm, m_stride))
    media[:, 0:3] = t["m_props"]
    media[:, 3:m_stop] = t["m_frac"]
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
    m_int = np.zeros((nm, 2 + width), dtype=np.int32)
    m_int[:, 0] = t["m_type"]
    m_int[:, 1] = (t["m_comp"] >= 0).sum(axis=1)
    m_int[:, 2:] = t["m_comp"]
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
        M_STRIDE=m_stride, M_RHO=0, M_NEL=1, M_X0=2, M_FRAC=3, M_STOP=m_stop, M_NUC=m_nuc, I_MSTRIDE=2 + width,
        STOP_BINS=stop_bins, NUC_BINS=nuc_bins, TYPE_ICRU=TYPE_ICRU, TYPE_PP=TYPE_PP,
        WATER_MEDIUM=list(t["media"]).index("water"),
    )
    return F, ints, defines


def transport_source(defines: dict) -> str:
    consts = "".join(f"const {k}: i32 = {int(v)};\n" for k, v in defines.items())
    return consts + _TRANSPORT.wgsl + wgsl_asset("mc_transport.wgsl")


class _McLib:
    def __init__(self) -> None:
        F, ints, defines = pack_tables(load_tables())
        self.F = storage(F, label="mc tables")
        self.I = storage(ints, label="mc index")
        self.transport = Kernel(transport_source(defines), label="mc transport")
        self.fold = Kernel(_FOLD, label="mc fold")


@functools.cache
def _mc_lib() -> _McLib:
    return _McLib()


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


def _pack64(raw: np.ndarray) -> np.ndarray:
    return raw.view(np.int64)  # (lo, hi) little-endian pairs


class McRun:
    """Transport *histories* protons through a phantom or CT, a slice at a time.

    *spots* are kernel spot records with the proton CDF in place; *protons* is the
    total they carry. The dose grid is ``shape`` (nx, ny, nz) voxels of *spacing_cm*
    from *origin_cm*. :meth:`step` transports for about a time budget and
    :meth:`preview` returns the running estimate, so the GUI can draw in between.
    The finished :attr:`dose` (``(nz, ny, nx)`` Gy) and :attr:`result` don't depend
    on how the run was sliced. Use :meth:`slab` or :meth:`patient` to build one.
    """

    def __init__(
        self,
        spots: np.ndarray,
        protons: float,
        *,
        mode: int,
        origin_cm,
        spacing_cm,
        shape,
        histories: int,
        seed: int,
        medium: int = 0,
        depth_cm: float = 0.0,
        wet_cm: float = 0.0,
        material: np.ndarray | None = None,
        density: np.ndarray | None = None,
        beams: np.ndarray | None = None,
        flags: int = 0,
    ) -> None:
        self.result: McResult | None = None
        self.dose: np.ndarray | None = None
        # LET_d in keV/µm per voxel, when the run scores it.
        self.let: np.ndarray | None = None
        self._bufs: dict = {}
        nx, ny, nz = (int(v) for v in shape)
        self._shape_zyx = (nz, ny, nx)
        nvox = self._nvox = nx * ny * nz
        self._flags = int(flags)
        histories = max(int(histories), BATCHES)
        self._per_batch = min(BATCH_HISTORIES, histories // BATCHES)
        self.histories = histories // self._per_batch * self._per_batch
        self._next = 0  # histories dispatched so far
        self._chunk = DISPATCH_HISTORIES
        spots = np.ascontiguousarray(spots, dtype=np.float32)
        if protons <= 0.0 or len(spots) == 0:
            self.dose = np.zeros(self._shape_zyx, dtype=np.float32)
            self.result = McResult(0, 0.0, {k: 0.0 for k in LEDGER_KEYS}, 0, 0)
            return

        dev = device()
        limit = dev.limits["max-storage-buffer-binding-size"]
        if nvox * 16 > limit:
            raise RuntimeError(f"grid needs {nvox * 16} bytes for its LET tally; the GPU allows {limit}")
        spacing = np.broadcast_to(np.asarray(spacing_cm, dtype=float), (3,))
        origin = np.asarray(origin_cm, dtype=float).reshape(3)
        # Tallies hold E / rho per history weight; storing divides by the histories run.
        self._gy_per_quantum = QUANTUM_MEV * float(protons) * MEV_TO_J / (float(np.prod(spacing)) * 1e-3)
        self._params = dict(
            mode=int(mode), flags=self._flags, ox=origin[0], oy=origin[1], oz=origin[2],
            vx=spacing[0], vy=spacing[1], vz=spacing[2], nx=nx, ny=ny, nz=nz,
            depth=max(float(depth_cm), 0.0), wet=max(float(wet_cm), 0.0), medium=int(medium),
            nspots=len(spots), seed=int(seed) & 0xFFFFFFFF,
        )
        if material is not None:
            words = np.zeros(-(-nvox // 4) * 4, dtype=np.uint8)
            words[:nvox] = np.asarray(material, dtype=np.uint8).reshape(-1)
            material = words.view(np.uint32)
        lib = self._lib = _mc_lib()
        b = self._bufs
        b["spots"] = storage(spots)
        b["tally"] = storage(size=8 * nvox)
        b["ledger"] = storage(np.zeros(15, dtype=np.uint32))
        b["params"] = uniform(_TRANSPORT.nbytes)
        b["material"] = storage(material if material is not None else np.zeros(1, dtype=np.uint32))
        b["density"] = storage(np.asarray(density, dtype=np.float32).reshape(-1) if density is not None else np.zeros(1, dtype=np.float32))
        b["let"] = storage(size=16 * nvox if self._flags & FLAG_LET else 4)
        b["beams"] = storage(beams if beams is not None else np.zeros(BEAM_STRIDE, dtype=np.float32))
        b["sum"] = storage(size=4 * nvox)
        b["sq"] = storage(size=4 * nvox)
        b["out"] = storage(size=4 * nvox)
        b["fold"] = uniform(_FOLD_P.nbytes)
        self._transport = lib.transport.bind(
            b["spots"], lib.F, lib.I, b["tally"], b["ledger"], b["params"],
            b["material"], b["density"], b["let"], b["beams"],
        )
        self._fold_group = lib.fold.bind(b["tally"], b["sum"], b["sq"], b["out"], b["fold"])

    @classmethod
    def slab(
        cls, x, y, sx, sy, energy, protons, medium_key: str, grid: DoseGrid, *,
        depth: float, wet: float, spread_pct: float, histories: int, seed: int,
    ) -> McRun:
        """Spots at the nozzle in scene mm and MeV: centers (x, y), sigmas (sx, sy), energy,
        and delivered protons, into *medium_key* from ``z = 0`` to ``-depth`` mm, laterally
        infinite, behind *wet* mm of water, scored on *grid*.
        """
        media = mc_media()
        if medium_key not in media:
            raise ValueError(f"no MCsquare material data for {medium_key!r}")
        e = np.asarray(energy, dtype=float).reshape(-1)
        w = np.asarray(protons, dtype=float).reshape(-1)
        cols = [np.asarray(v, dtype=float).reshape(-1) for v in (x, y, sx, sy)]
        ok = np.isfinite(e) & np.isfinite(w) & (w > 0.0) & (e > 0.0)
        for c in cols:
            ok &= np.isfinite(c)
        total = float(w[ok].sum())
        spots = np.zeros((int(ok.sum()), SIMPLE_STRIDE), dtype=np.float32)
        for k, c in enumerate(cols):
            spots[:, k] = c[ok] / 10.0
        spots[:, 2:4] = np.maximum(spots[:, 2:4], 0.0)
        spots[:, 4] = e[ok]
        spots[:, 5] = max(float(spread_pct), 0.0) / 100.0 * e[ok]
        if len(spots):
            spots[:, 6] = np.cumsum(w[ok]) / total
            spots[-1, 6] = 1.0
        return cls(
            spots, total, mode=MODE_SLAB, origin_cm=np.asarray(grid.origin, dtype=float) / 10.0,
            spacing_cm=float(grid.voxel) / 10.0, shape=grid.shape, histories=histories, seed=seed,
            medium=media.index(medium_key), depth_cm=float(depth) / 10.0, wet_cm=float(wet) / 10.0,
        )

    @classmethod
    def patient(
        cls, spots: np.ndarray, protons, beams: np.ndarray, material: np.ndarray, density: np.ndarray,
        spacing_mm, *, histories: int, seed: int, dose_to_water: bool = True, let: bool = False,
    ) -> McRun:
        """BDL spot records (:meth:`~scan_kit.qa.beam_model.BeamModel.spot_records`) carrying
        *protons* each, through a ``(nz, ny, nx)`` volume of per-voxel medium and density,
        scored on its own grid.

        *beams* rows are :func:`beam_record`\\ s in the volume's frame: mm along its axes from
        the corner of voxel 0.
        """
        w = np.asarray(protons, dtype=float).reshape(-1)
        keep = np.isfinite(w) & (w > 0.0)
        spots = np.array(spots, dtype=np.float32).reshape(-1, BDL_STRIDE)[keep]
        w = w[keep]
        if len(w):
            spots[:, 5] = np.cumsum(w) / w.sum()
            spots[-1, 5] = 1.0
        material = np.asarray(material)
        if material.shape != np.shape(density) or material.ndim != 3:
            raise ValueError("material and density must be the same (nz, ny, nx) volume")
        if len(mc_media()) > 256 or (material.size and int(material.max()) >= len(mc_media())):
            raise ValueError("material indices must address the Monte Carlo tables")
        flags = (FLAG_DOSE_TO_WATER if dose_to_water else 0) | (FLAG_LET if let else 0)
        return cls(
            spots, float(w.sum()), mode=MODE_PATIENT, origin_cm=(0.0, 0.0, 0.0),
            spacing_cm=np.asarray(spacing_mm, dtype=float) / 10.0, shape=material.shape[::-1],
            histories=histories, seed=seed, material=material, density=density,
            beams=np.ascontiguousarray(beams, dtype=np.float32).reshape(-1), flags=flags,
        )

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def progress(self) -> float:
        """Share of the histories transported."""
        return 1.0 if self.done else self._next / self.histories

    def extend(self, histories: int) -> bool:
        """Retarget to *histories*, keeping what's transported; False when that's already past it."""
        target = max(int(histories) // self._per_batch, 1) * self._per_batch
        if not self._bufs or target < self._next:
            return False
        if target != self.histories:
            self.histories = target
            self.result = None
        return True

    def _fold(self, mode: int, gain: float) -> None:
        device().queue.write_buffer(self._bufs["fold"], 0, _FOLD_P.pack(
            n=self._nvox, mode=mode, scale=self._gy_per_quantum, gain=gain,
        ))
        self._lib.fold.run(self._fold_group, *workgroups_1d(self._nvox))

    def step(self, budget_s: float = math.inf) -> bool:
        """Transport for about *budget_s* seconds; True once the run is finished."""
        if self.done:
            return True
        queue = device().queue
        end = time.perf_counter() + budget_s
        while True:
            start = self._next % self._per_batch
            count = min(self._chunk, self._per_batch - start)
            t0 = time.perf_counter()
            queue.write_buffer(self._bufs["params"], 0, _TRANSPORT.pack(
                **self._params, hist_base=self._next, nhist=count,
            ))
            queue.write_buffer(self._bufs["ledger"], 56, bytes(4))
            self._lib.transport.run(self._transport, (count + 63) // 64)
            # Short submissions keep each one well under the Windows GPU timeout.
            wait()
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
                self._fold(1 if last else 0, 1.0 / self._next)
                if last:
                    self._finish()
                    return True
            if time.perf_counter() >= end:
                return False

    def preview(self) -> np.ndarray | None:
        """The mean dose of the histories so far, ``(nz, ny, nx)`` Gy; None before any."""
        if self.done:
            return self.dose
        if self._next == 0:
            return None
        self._fold(2, 1.0 / self._next)
        return read(self._bufs["out"], np.float32).reshape(self._shape_zyx)

    def _finish(self) -> None:
        b = self._bufs
        raw = read(b["ledger"], np.uint32)
        pairs = _pack64(raw[:12])
        ledger = {k: float(pairs[i]) * QUANTUM_MEV / self.histories for i, k in enumerate(LEDGER_KEYS)}
        dose_sum = read(b["sum"], np.float32)
        self.dose = read(b["out"], np.float32).reshape(self._shape_zyx)
        if self._flags & FLAG_LET:
            q = _pack64(read(b["let"], np.uint32)).reshape(-1, 2).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                let = np.where(q[:, 1] > 0, q[:, 0] / q[:, 1], 0.0)
            self.let = let.astype(np.float32).reshape(self._shape_zyx)
        self.result = McResult(
            histories=self.histories,
            uncertainty=batch_uncertainty(dose_sum, read(b["sq"], np.float32), self.histories // self._per_batch),
            ledger=ledger,
            overflow=int(raw[12]),
            deepest=int(raw[13]),
        )

    def close(self) -> None:
        """Free the GPU buffers, after which the run can't :meth:`extend`; :attr:`dose` stays."""
        for buf in self._bufs.values():
            buf.destroy()
        self._bufs = {}


def beam_record(rotation, isocenter_mm, nozzle_to_iso_mm: float, smx_mm: float, smy_mm: float) -> np.ndarray:
    """A kernel beam record: *rotation* takes the beam frame (IEC gantry, Z toward the source)
    to the volume frame, where the isocenter sits at *isocenter_mm*."""
    rec = np.zeros(BEAM_STRIDE, dtype=np.float32)
    rec[0:9] = np.asarray(rotation, dtype=float).reshape(9)
    rec[9:12] = np.asarray(isocenter_mm, dtype=float).reshape(3) / 10.0
    rec[12:15] = nozzle_to_iso_mm, smx_mm, smy_mm
    return rec


def mc_dose(*args, **kwargs) -> tuple[np.ndarray, McResult]:
    """:meth:`McRun.slab` to completion in one call: ``(dose, result)``."""
    run = McRun.slab(*args, **kwargs)
    run.step()
    run.close()
    return run.dose, run.result


def mc_note(result: McResult) -> str:
    """`` · MC 1.0M · ±0.8 % · 0.4 % off-grid`` for the volume label."""
    h = result.histories
    count = f"{h / 1e6:.1f}M" if h >= 1e6 else f"{h / 1e3:.0f}k"
    return f" · MC {count} · ±{100.0 * result.uncertainty:.1f} % · {100.0 * result.fraction('off_grid'):.1f} % off-grid"
