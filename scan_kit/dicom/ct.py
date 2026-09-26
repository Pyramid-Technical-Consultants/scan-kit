"""CT series to a HU volume on a :class:`VolumeGrid`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grid import VolumeGrid

# Slice gaps may drift this much (relative) before the series counts as non-uniform.
SPACING_TOLERANCE = 0.01


class DicomError(ValueError):
    """Input DICOM that can't be used safely."""


@dataclass(frozen=True)
class CtImage:
    grid: VolumeGrid
    hu: np.ndarray  # (nz, ny, nx) float32
    study_uid: str
    series_uid: str
    sop_uids: tuple[str, ...]  # in slice order
    patient_position: str  # HFS, HFP, FFS, FFP, ...
    patient_id: str = ""
    patient_name: str = ""


def _read(path):
    import pydicom

    return pydicom.dcmread(str(path), force=True)


def ct_from_datasets(slices: Iterable) -> CtImage:
    """Stack CT *slices* (pydicom datasets, any order) into one volume."""
    slices = list(slices)
    if not slices:
        raise DicomError("no CT slices")
    first = slices[0]
    series = {str(s.SeriesInstanceUID) for s in slices}
    if len(series) != 1:
        raise DicomError(f"CT slices come from {len(series)} series")
    iop = np.asarray(first.ImageOrientationPatient, dtype=float)
    for s in slices:
        if not np.allclose(np.asarray(s.ImageOrientationPatient, dtype=float), iop, atol=1e-4):
            raise DicomError("CT slices disagree on ImageOrientationPatient")
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    order = sorted(slices, key=lambda s: float(np.dot(np.asarray(s.ImagePositionPatient, dtype=float), normal)))
    pos = np.array([np.asarray(s.ImagePositionPatient, dtype=float) for s in order])
    along = pos @ normal
    if len(order) > 1:
        gaps = np.diff(along)
        dz = float(np.median(gaps))
        if dz <= 0.0 or np.any(np.abs(gaps - dz) > SPACING_TOLERANCE * dz):
            raise DicomError(f"CT slice spacing is not uniform ({gaps.min():.3f} to {gaps.max():.3f} mm)")
        lateral = pos - pos[0] - np.outer(along - along[0], normal)
        if np.abs(lateral).max() > 1e-2:
            raise DicomError("CT slices are not stacked along their normal")
    else:
        dz = float(first.get("SliceThickness", 1.0) or 1.0)
    rows, cols = int(first.Rows), int(first.Columns)
    hu = np.empty((len(order), rows, cols), dtype=np.float32)
    for k, s in enumerate(order):
        if (int(s.Rows), int(s.Columns)) != (rows, cols):
            raise DicomError("CT slices differ in size")
        slope = float(s.get("RescaleSlope", 1.0) or 1.0)
        intercept = float(s.get("RescaleIntercept", 0.0) or 0.0)
        try:
            pixels = s.pixel_array
        except Exception as exc:
            raise DicomError(f"cannot decode CT pixels ({s.file_meta.TransferSyntaxUID.name}): {exc}") from exc
        hu[k] = pixels.astype(np.float32) * slope + intercept
    # PixelSpacing is (between rows, between columns): y then x.
    dy, dx = (float(v) for v in first.PixelSpacing)
    grid = VolumeGrid(
        origin=pos[0], spacing=(dx, dy, dz), shape=(cols, rows, len(order)),
        axes=np.column_stack([row, col, normal]), frame_uid=str(first.get("FrameOfReferenceUID", "")),
    )
    return CtImage(
        grid=grid, hu=hu, study_uid=str(first.get("StudyInstanceUID", "")), series_uid=str(first.SeriesInstanceUID),
        sop_uids=tuple(str(s.SOPInstanceUID) for s in order),
        patient_position=str(first.get("PatientPosition", "") or "HFS"),
        patient_id=str(first.get("PatientID", "")), patient_name=str(first.get("PatientName", "")),
    )


def load_ct(paths: Iterable[str | Path]) -> CtImage:
    return ct_from_datasets(_read(p) for p in paths)
