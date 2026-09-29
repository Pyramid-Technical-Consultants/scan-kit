"""RTDOSE in and out."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ct import SPACING_TOLERANCE, CtImage, DicomError
from .grid import VolumeGrid

RT_DOSE = "1.2.840.10008.5.1.4.1.1.481.2"
DISCLAIMER = "scan-kit research dose, not for clinical use"


@dataclass(frozen=True)
class DoseImage:
    grid: VolumeGrid
    dose: np.ndarray  # (nz, ny, nx) float32 Gy
    sop_uid: str
    plan_uid: str  # ReferencedRTPlanSequence, "" if none
    summation: str  # PLAN, BEAM, FRACTION, ...
    dose_type: str  # PHYSICAL, EFFECTIVE, ...
    beam_number: int | None = None
    comment: str = ""


def dose_from_dataset(ds) -> DoseImage:
    if str(ds.get("SOPClassUID", "")) != RT_DOSE:
        raise DicomError("not an RT Dose")
    if str(ds.get("DoseUnits", "GY")).upper() != "GY":
        raise DicomError(f"RTDOSE units {ds.DoseUnits!r} are not Gy")
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    pixels = np.asarray(ds.pixel_array, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = pixels[None]
    offsets = np.asarray(ds.get("GridFrameOffsetVector", [0.0]), dtype=float)[: pixels.shape[0]]
    origin = np.asarray(ds.ImagePositionPatient, dtype=float)
    # Offsets are relative to ImagePositionPatient when the first is 0, else absolute z (PS3.3 C.8.8.3.2).
    if offsets.size and offsets[0] != 0.0:
        offsets = offsets - float(np.dot(origin, normal))
    if offsets.size > 1:
        gaps = np.diff(offsets)
        dz = float(np.median(gaps))
        if dz == 0.0 or np.any(np.abs(gaps - dz) > SPACING_TOLERANCE * abs(dz)):
            raise DicomError("RTDOSE frame spacing is not uniform")
        if dz < 0.0:
            pixels, dz = pixels[::-1], -dz
            origin = origin + normal * offsets[-1]
    else:
        dz = float(ds.get("SliceThickness", 1.0) or 1.0)
    dose = pixels * float(ds.get("DoseGridScaling", 1.0) or 1.0)
    dy, dx = (float(v) for v in ds.PixelSpacing)
    grid = VolumeGrid(
        origin=origin, spacing=(dx, dy, dz), shape=(dose.shape[2], dose.shape[1], dose.shape[0]),
        axes=np.column_stack([row, col, normal]), frame_uid=str(ds.get("FrameOfReferenceUID", "")),
    )
    plan = next(iter(ds.get("ReferencedRTPlanSequence", [])), None)
    beam = None
    if plan is not None:
        for fg in plan.get("ReferencedFractionGroupSequence", []):
            for rb in fg.get("ReferencedBeamSequence", []):
                beam = int(rb.ReferencedBeamNumber)
    return DoseImage(
        grid=grid, dose=np.ascontiguousarray(dose, dtype=np.float32), sop_uid=str(ds.SOPInstanceUID),
        plan_uid=str(plan.ReferencedSOPInstanceUID) if plan is not None else "",
        summation=str(ds.get("DoseSummationType", "") or ""), dose_type=str(ds.get("DoseType", "") or ""),
        beam_number=beam, comment=str(ds.get("DoseComment", "") or ""),
    )


def load_dose(path: str | Path) -> DoseImage:
    import pydicom

    return dose_from_dataset(pydicom.dcmread(str(path), force=True))


def write_dose(
    path: str | Path,
    dose: np.ndarray,
    grid: VolumeGrid,
    ct: CtImage,
    *,
    plan_uid: str = "",
    summation: str = "PLAN",
    description: str = "scan-kit MC",
    provenance: dict | None = None,
) -> Path:
    """Write *dose* (``(nz, ny, nx)`` Gy on *grid*) as an RTDOSE in *ct*'s study and frame.

    *provenance* (inputs, engine, seed, ...) is stored as JSON in the dose comment.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, generate_uid

    from .. import __version__

    dose = np.asarray(dose, dtype=np.float64)
    if dose.shape != grid.shape_zyx:
        raise ValueError(f"dose is {dose.shape}, grid wants {grid.shape_zyx}")
    if grid.frame_uid and grid.frame_uid != ct.grid.frame_uid:
        raise DicomError("dose grid and CT are in different frames of reference")
    peak = float(np.nanmax(dose)) if dose.size else 0.0
    scale = peak / 4.0e9 if peak > 0.0 else 1.0
    pixels = np.rint(np.clip(np.nan_to_num(dose), 0.0, None) / scale).astype(np.uint32)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = RT_DOSE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    now = datetime.datetime.now()
    ds.SOPClassUID = RT_DOSE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.Manufacturer = "Pyramid Technical Consultants"
    ds.SoftwareVersions = f"scan-kit {__version__}"
    ds.ContentDate = ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.ContentTime = ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.PatientID = ct.patient_id
    ds.PatientName = ct.patient_name
    ds.StudyInstanceUID = ct.study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesDescription = f"{description} (research only)"[:64]  # LO; DoseComment carries the disclaimer
    ds.FrameOfReferenceUID = ct.grid.frame_uid
    ds.ImagePositionPatient = [float(v) for v in grid.origin]
    ds.ImageOrientationPatient = [float(v) for v in np.concatenate([grid.axes[:, 0], grid.axes[:, 1]])]
    ds.PixelSpacing = [float(grid.spacing[1]), float(grid.spacing[0])]
    ds.SliceThickness = float(grid.spacing[2])
    nx, ny, nz = grid.shape
    ds.NumberOfFrames = nz
    ds.FrameIncrementPointer = pydicom.tag.Tag("GridFrameOffsetVector")
    ds.GridFrameOffsetVector = [float(k * grid.spacing[2]) for k in range(nz)]
    ds.Rows, ds.Columns = ny, nx
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = summation
    ds.DoseGridScaling = f"{scale:.10e}"
    ds.DoseComment = DISCLAIMER
    ds.ImageComments = json.dumps(provenance or {}, sort_keys=True, default=str)
    if plan_uid:
        ref = pydicom.Dataset()
        ref.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.8"
        ref.ReferencedSOPInstanceUID = plan_uid
        ds.ReferencedRTPlanSequence = [ref]
    ds.PixelData = pixels.tobytes()
    out = Path(path)
    ds.save_as(str(out), enforce_file_format=True)
    return out
