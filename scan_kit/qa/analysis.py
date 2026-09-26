"""QA analysis: DVHs, clinical goals, gamma against the TPS dose, and provenance for export."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from ..dicom.ct import DicomError
from ..dicom.dose import DoseImage
from ..dicom.grid import VolumeGrid
from ..dicom.structures import MAX_MASK_BITS, mask_bits
from ..views.dose_volume_physics import GammaCriteria

DVH_BINS = 4096
# DoseSummationTypes covering every fraction (PS3.3 C.8.8.3); the *_SESSION ones and
# CONTROL_POINT are one fraction.
COURSE_SUMMATIONS = {"PLAN", "MULTI_PLAN", "PLAN_OVERVIEW", "FRACTION", "BEAM", "BRACHY"}
BEAM_SUMMATIONS = {"BEAM", "BEAM_SESSION", "CONTROL_POINT"}
# Constant proton RBE: TPS EFFECTIVE doses and prescriptions are in Gy(RBE).
RBE = 1.1


@dataclass(frozen=True)
class Dvh:
    name: str
    edges: np.ndarray  # Gy, bins + 1
    counts: np.ndarray  # voxels per bin
    voxel_cc: float
    dmax: float
    dmin: float

    @property
    def voxels(self) -> int:
        return int(self.counts.sum())

    @property
    def volume_cc(self) -> float:
        return self.voxels * self.voxel_cc

    @property
    def cumulative(self) -> np.ndarray:
        """Volume fraction receiving at least ``edges[i]``."""
        n = max(self.voxels, 1)
        return np.r_[np.cumsum(self.counts[::-1])[::-1], 0] / n

    @property
    def mean(self) -> float:
        centers = 0.5 * (self.edges[1:] + self.edges[:-1])
        return float((centers * self.counts).sum() / max(self.voxels, 1))

    def dose_at(self, fraction: float) -> float:
        """D at a volume *fraction* (0.95 for D95%): the dose that much of the structure reaches."""
        if fraction <= 0.0:
            return self.dmax
        if fraction >= 1.0:
            return self.dmin
        c = self.cumulative
        return float(np.interp(fraction, c[::-1], self.edges[::-1]))

    def dose_at_cc(self, cc: float) -> float:
        return self.dose_at(cc / self.volume_cc) if self.volume_cc > 0 else 0.0

    def volume_at(self, dose: float) -> float:
        """V at *dose* Gy, as a fraction of the structure."""
        return float(np.interp(dose, self.edges, self.cumulative))


def dvhs(dose: np.ndarray, grid: VolumeGrid, rois, *, bins: int = DVH_BINS, top: float | None = None,
         bits=None) -> dict:
    """Name -> :class:`Dvh` for *rois* on *dose* (``(nz, ny, nx)`` Gy on *grid*), voxel-center inclusion.

    *bits* reuses :func:`mask_bits` volumes already made for *rois*, one per ``MAX_MASK_BITS`` of them.

    ponytail: whole voxels in or out; small structures want supersampled partial volumes.
    """
    from ..gpu.dvh import histograms

    rois = list(rois)
    out = {}
    peak = float(np.max(dose)) if dose.size else 0.0
    hi = float(top) if top else (peak * (1.0 + 1e-6) or 1.0)
    edges = np.linspace(0.0, hi, bins + 1)
    cc = float(np.prod(grid.spacing)) / 1000.0
    for c, start in enumerate(range(0, len(rois), MAX_MASK_BITS)):
        chunk = rois[start:start + MAX_MASK_BITS]
        mask = bits[c] if bits is not None else mask_bits(chunk, grid)
        counts, dmax, dmin = histograms(dose, mask, len(chunk), bins, hi)
        for r, roi in enumerate(chunk):
            out[roi.name] = Dvh(roi.name, edges, counts[r], cc, float(dmax[r]), float(dmin[r]) if counts[r].any() else 0.0)
    return out


_GOAL = re.compile(
    r"^\s*(?P<roi>.+?)\s*:\s*(?P<metric>Dmean|Dmax|Dmin|D(?P<dv>\d+(?:\.\d+)?)(?P<du>%|cc)|V(?P<vd>\d+(?:\.\d+)?)(?P<vu>Gy|%))"
    r"\s*(?P<op><=|>=|<|>)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>Gy|%|cc)?\s*$",
    re.IGNORECASE,
)
_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


@dataclass(frozen=True)
class Goal:
    """A clinical goal like ``PTV: D95% >= 95%``, ``Cord: Dmax < 45 Gy`` or ``Lung: V20Gy < 30%``.

    Dose values in % are of the prescription; V goals in % are of the structure's volume.
    """

    text: str
    roi: str
    metric: str
    op: str
    value: float
    unit: str

    @classmethod
    def parse(cls, text: str) -> Goal:
        m = _GOAL.match(text)
        if m is None:
            raise ValueError(f"not a goal: {text!r} (try 'PTV: D95% >= 95%' or 'Cord: Dmax < 45 Gy')")
        metric = m["metric"]
        volume = metric.upper().startswith("V")
        unit = {"gy": "Gy", "%": "%", "cc": "cc"}[(m["unit"] or ("%" if volume else "Gy")).lower()]
        if unit == ("Gy" if volume else "cc"):
            raise ValueError(f"{text!r}: a {'volume' if volume else 'dose'} goal can't be in {unit}")
        return cls(text.strip(), m["roi"], metric, m["op"], float(m["value"]), unit)

    def measure(self, dvh: Dvh, rx: float | None = None) -> float:
        """The metric in the goal's unit."""
        def need_rx() -> float:
            if not rx:
                raise ValueError(f"{self.text!r} needs the prescription dose")
            return float(rx)

        def pct_of_rx(gy: float) -> float:
            return 100.0 * gy / need_rx()

        m = self.metric.lower()
        if m.startswith("v"):
            level = float(m[1:-2] if m.endswith("gy") else m[1:-1])
            frac = dvh.volume_at(level if m.endswith("gy") else level / 100.0 * need_rx())
            return frac * dvh.volume_cc if self.unit == "cc" else 100.0 * frac
        if m == "dmean":
            gy = dvh.mean
        elif m == "dmax":
            gy = dvh.dmax
        elif m == "dmin":
            gy = dvh.dmin
        elif m.endswith("cc"):
            gy = dvh.dose_at_cc(float(m[1:-2]))
        else:
            gy = dvh.dose_at(float(m[1:-1]) / 100.0)
        return pct_of_rx(gy) if self.unit == "%" else gy

    def check(self, dvh: Dvh, rx: float | None = None) -> tuple[float, bool]:
        v = self.measure(dvh, rx)
        return v, bool(_OPS[self.op](v, self.value))


def resample(dose: np.ndarray, grid: VolumeGrid, target: VolumeGrid) -> np.ndarray:
    """*dose* on *grid* trilinearly at *target*'s voxel centers, 0 outside."""
    from ..gpu import GpuUnavailable
    from ..gpu.resample import resample_affine

    if grid.frame_uid and target.frame_uid and grid.frame_uid != target.frame_uid:
        raise DicomError("dose grids are in different frames of reference")
    # Target voxel index -> source voxel index is affine.
    rot = (grid.axes.T @ target.axes) * target.spacing[None, :] / grid.spacing[:, None]
    off = grid.axes.T @ (target.origin - grid.origin) / grid.spacing
    try:
        return resample_affine(dose, target.shape_zyx, rot, off)
    except GpuUnavailable:
        return _resample_cpu(dose, grid, target)


def _resample_cpu(dose: np.ndarray, grid: VolumeGrid, target: VolumeGrid) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    nx, ny, nz = target.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    out = np.empty(target.shape_zyx, dtype=np.float32)
    src = np.asarray(dose, dtype=np.float32)
    for k in range(nz):
        ijk = np.stack([ii, jj, np.full_like(ii, k)], axis=-1)
        idx = grid.to_index(target.to_patient(ijk))
        out[k] = map_coordinates(src, [idx[..., 2], idx[..., 1], idx[..., 0]], order=1, mode="constant", cval=0.0)
    return out


def tps_per_fraction(tps: DoseImage, fractions: int) -> np.ndarray:
    """The TPS dose for one fraction."""
    return tps.dose / max(int(fractions), 1) if tps.summation.upper() in COURSE_SUMMATIONS else tps.dose


@dataclass(frozen=True)
class GammaReport:
    gamma: np.ndarray  # on the TPS grid, 0 where not evaluated
    passed: int
    evaluated: int
    criteria: GammaCriteria

    @property
    def rate(self) -> float:
        return self.passed / self.evaluated if self.evaluated else float("nan")


def gamma_vs_tps(
    tps: DoseImage, dose: np.ndarray, grid: VolumeGrid, criteria: GammaCriteria, *, fractions: int = 1,
    mask: np.ndarray | None = None, effective: bool = False,
) -> GammaReport:
    """Global gamma of the TPS dose (reference, one fraction) searched in *dose* resampled to its grid.

    *dose* is physical, or Gy(RBE) when *effective*; it is compared in the TPS dose's DoseType.
    *mask* (bool on the TPS grid, e.g. BODY) limits both doses; the norm is the TPS maximum there.
    """
    from ..gpu.gamma import gamma_volume

    ref = tps_per_fraction(tps, fractions).astype(np.float32)
    scale = (RBE if tps.dose_type.upper() == "EFFECTIVE" else 1.0) / (RBE if effective else 1.0)
    evl = resample(dose, grid, tps.grid) * np.float32(scale)
    if mask is not None:
        ref, evl = np.where(mask, ref, 0.0).astype(np.float32), np.where(mask, evl, 0.0).astype(np.float32)
    g, passed, evaluated = gamma_volume(ref, evl, tps.grid.spacing, criteria, norm=float(ref.max()) if ref.size else 0.0)
    return GammaReport(g, passed, evaluated, criteria)


def provenance(
    *, kind: str, plan, calibration, model, result, histories: int, seed: int, deliveries=(), case=None,
) -> dict:
    """What an exported dose came from, for its RTDOSE ImageComments; *case* adds the input UIDs."""
    from .. import __version__
    from ..gpu import adapter_info

    try:
        gpu = adapter_info()
    except Exception:  # noqa: BLE001 - provenance never blocks an export
        gpu = {}
    table = hashlib.sha256(np.stack([model.table[c] for c in sorted(model.table)]).tobytes()).hexdigest()[:16]
    inputs = {}
    if case is not None:
        inputs = {
            "frame": case.frame_uid, "ct_series": case.ct.series_uid, "ct_images": len(case.ct.sop_uids),
            "ct_sop_digest": hashlib.sha256("\\".join(case.ct.sop_uids).encode()).hexdigest()[:16],
            "structures": case.structures.sop_uid if case.structures is not None else "",
        }
    return {
        "inputs": inputs,
        "kind": kind,
        "engine": f"scan-kit {__version__} GPU Monte Carlo (WebGPU port of MCsquare)",
        "gpu": " / ".join(str(gpu.get(k, "")) for k in ("device", "backend_type") if gpu.get(k)),
        "plan": {"uid": plan.sop_uid, "label": plan.label, "fractions": plan.fractions},
        "calibration": {"name": calibration.name, "digest": calibration.digest},
        "beam_model": {"name": model.name, "digest": table},
        "histories": int(histories),
        "seed": int(seed),
        "uncertainty": float(result.uncertainty) if result is not None else None,
        "deliveries": list(deliveries),
        "dose": "dose-to-water, Gy per fraction",
    }
