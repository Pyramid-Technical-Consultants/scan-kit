"""RTSTRUCT contours and their voxel masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ct import DicomError
from .grid import VolumeGrid

MAX_MASK_BITS = 32


@dataclass(frozen=True)
class Roi:
    number: int
    name: str
    color: tuple[int, int, int]
    kind: str  # RTROIInterpretedType: EXTERNAL, PTV, CTV, ORGAN, ...
    contours: tuple[np.ndarray, ...]  # closed planar polygons, (n, 3) patient mm


@dataclass(frozen=True)
class StructureSet:
    sop_uid: str
    frame_uid: str
    label: str
    rois: tuple[Roi, ...]

    def roi(self, name: str) -> Roi:
        for r in self.rois:
            if r.name == name:
                return r
        raise KeyError(name)


def structures_from_dataset(ds) -> StructureSet:
    frames = {str(r.ReferencedFrameOfReferenceUID) for r in ds.get("StructureSetROISequence", [])
              if "ReferencedFrameOfReferenceUID" in r}
    if len(frames) > 1:
        raise DicomError("RTSTRUCT ROIs span several frames of reference")
    names = {int(r.ROINumber): str(r.ROIName) for r in ds.get("StructureSetROISequence", [])}
    kinds = {int(o.ReferencedROINumber): str(o.get("RTROIInterpretedType", "") or "")
             for o in ds.get("RTROIObservationsSequence", [])}
    rois = []
    for item in ds.get("ROIContourSequence", []):
        number = int(item.ReferencedROINumber)
        contours = []
        for c in item.get("ContourSequence", []):
            if str(c.get("ContourGeometricType", "CLOSED_PLANAR")) not in ("CLOSED_PLANAR", "CLOSEDPLANAR_XOR"):
                continue
            pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            if len(pts) >= 3:
                contours.append(pts)
        color = tuple(int(v) for v in item.get("ROIDisplayColor", (255, 255, 0)))
        name = names.get(number, f"ROI {number}")
        # Names key DVHs and goals, so a repeated one gets its ROI number.
        if any(r.name == name for r in rois):
            name = f"{name} ({number})"
        rois.append(Roi(number, name, color, kinds.get(number, ""), tuple(contours)))
    return StructureSet(
        sop_uid=str(ds.SOPInstanceUID), frame_uid=next(iter(frames), str(ds.get("FrameOfReferenceUID", ""))),
        label=str(ds.get("StructureSetLabel", "") or ""), rois=tuple(rois),
    )


def load_structures(path: str | Path) -> StructureSet:
    import pydicom

    return structures_from_dataset(pydicom.dcmread(str(path), force=True))


def rasterize(roi: Roi, grid: VolumeGrid) -> np.ndarray:
    """Voxels of *grid* whose centers lie inside *roi*, as ``(nz, ny, nx)`` bool.

    Contours on one plane combine even-odd, so an inner contour is a hole. Each slice takes
    the nearest contour plane within half the contour spacing (half a slice for a single
    plane), so contours finer or coarser than the grid neither cancel nor leave gaps.
    """
    out = np.zeros(grid.shape_zyx, dtype=bool)
    box = _raster(roi, grid)
    if box is not None:
        out[box[0], box[1]] = box[2]
    return out


def _raster(roi: Roi, grid: VolumeGrid):
    """(z slice, y slice, bool mask there) bounding *roi* on *grid*, or None when it covers nothing."""
    nx, ny, nz = grid.shape
    rows = np.arange(ny, dtype=float)
    planes: dict[float, list] = {}
    for pts in roi.contours:
        ijk = grid.to_index(pts)
        if np.ptp(ijk[:, 2]) > 0.5:
            continue
        x0, y0 = ijk[:, 0], ijk[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        # Edges crossing each row of voxel centers, half-open so shared vertices count once.
        cross = (y0[:, None] <= rows) != (y1[:, None] <= rows)
        e, r = np.nonzero(cross)
        t = (rows[r] - y0[e]) / (y1[e] - y0[e])
        x = x0[e] + t * (x1[e] - x0[e])
        i = np.clip(np.ceil(x), 0, nx).astype(np.intp)
        planes.setdefault(round(float(ijk[:, 2].mean()), 2), []).append((r, i))
    hit = [r for c in planes.values() for r, _i in c if r.size]
    if not hit:
        return None
    z = np.array(sorted(planes))
    reach = 0.5 * max(float(np.diff(z).min()) if len(z) > 1 else 1.0, 1.0)
    ks = [k for k in range(max(int(np.floor(z[0] - reach)), 0), min(int(np.ceil(z[-1] + reach)) + 1, nz))
          if np.abs(z - k).min() <= reach]
    if not ks:
        return None
    r0, r1 = min(int(r.min()) for r in hit), max(int(r.max()) for r in hit) + 1
    toggles = np.zeros((ks[-1] + 1 - ks[0], r1 - r0, nx + 1), dtype=np.uint8)
    for k in ks:
        for r, i in planes[z[int(np.abs(z - k).argmin())]]:
            np.bitwise_xor.at(toggles[k - ks[0]], (r - r0, i), 1)
    mask = np.bitwise_xor.accumulate(toggles, axis=2)[:, :, :nx].astype(bool)
    return slice(ks[0], ks[-1] + 1), slice(r0, r1), mask


def mask_bits(rois, grid: VolumeGrid) -> np.ndarray:
    """``(nz, ny, nx)`` uint32 with bit *n* set inside ``rois[n]``, for GPU DVH and display."""
    rois = list(rois)
    if len(rois) > MAX_MASK_BITS:
        raise ValueError(f"at most {MAX_MASK_BITS} structures per mask volume, got {len(rois)}")
    bits = np.zeros(grid.shape_zyx, dtype=np.uint32)
    for n, roi in enumerate(rois):
        box = _raster(roi, grid)
        if box is not None:
            bits[box[0], box[1]] |= box[2].astype(np.uint32) << np.uint32(n)
    return bits
