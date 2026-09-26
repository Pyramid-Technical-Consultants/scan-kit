"""A small synthetic patient written as real DICOM: CT, RTSTRUCT, RT Ion Plan (and RTDOSE).

For tests and demos; it carries no patient data. The phantom is a water box in air with a
bone and a lung insert; the PTV is a cube behind them and a ring ROI has a hole.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .plan import RT_ION_PLAN
from .study import CT_IMAGE as CT_CLASS
from .study import RT_STRUCT

BOX_MM = 120.0  # water box edge
BONE = ((-60.0, 60.0), (-20.0, -10.0), (-30.0, 30.0))  # x, y, z extents (mm) of the bone slab
LUNG = ((-60.0, 60.0), (-5.0, 10.0), (-30.0, 30.0))
PTV = ((-15.0, 15.0), (20.0, 50.0), (-15.0, 15.0))


@dataclass(frozen=True)
class Phantom:
    folder: Path
    frame_uid: str
    study_uid: str
    ct_files: tuple[Path, ...]
    struct_file: Path
    plan_file: Path
    plan_uid: str


def _base(sop_class: str, modality: str, study: str, frame: str, series: str | None = None):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.PatientName = "Phantom^Synthetic"
    ds.PatientID = "SYNTH-0001"
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series or generate_uid()
    ds.FrameOfReferenceUID = frame
    ds.StudyDate = ds.SeriesDate = datetime.date.today().strftime("%Y%m%d")
    return ds


def phantom_hu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """HU at patient points (broadcast arrays, mm)."""

    def inside(box):
        (x0, x1), (y0, y1), (z0, z1) = box
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)

    half = BOX_MM / 2
    hu = np.where(inside(((-half, half), (-half, half), (-half, half))), 0.0, -1000.0)
    hu = np.where(inside(BONE), 700.0, hu)
    return np.where(inside(LUNG), -750.0, hu)


def _square(x0, x1, y0, y1, z) -> list[float]:
    return [v for p in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)) for v in p]


def write_phantom(
    folder: str | Path,
    *,
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    size: tuple[int, int, int] = (80, 80, 70),
    position: str = "HFS",
    gantry: float = 0.0,
    couch: float = 0.0,
    energies: tuple[float, ...] = (120.0, 130.0, 140.0),
    spot_pitch: float = 6.0,
    mu_per_spot: float = 0.02,
    range_shifter_wet: float = 0.0,
) -> Phantom:
    """Write the phantom into *folder*; the CT slices are shuffled on disk on purpose."""
    import pydicom
    from pydicom.uid import generate_uid

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    study, frame, series = generate_uid(), generate_uid(), generate_uid()
    dx, dy, dz = spacing
    nx, ny, nz = size
    # HFS keeps the image axes on +L, +P; HFP flips both, as scanners do.
    flip = -1.0 if position.upper().endswith("P") else 1.0
    row, col = np.array([flip, 0.0, 0.0]), np.array([0.0, flip, 0.0])
    x0 = -flip * (nx - 1) * dx / 2
    y0 = -flip * (ny - 1) * dy / 2
    zs = (np.arange(nz) - (nz - 1) / 2) * dz
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    px, py = x0 + row[0] * ii * dx, y0 + col[1] * jj * dy
    ct_files = []
    for k in np.random.default_rng(0).permutation(nz):
        ds = _base(CT_CLASS, "CT", study, frame, series)
        ds.PatientPosition = position
        ds.ImageOrientationPatient = [*row, *col]
        ds.ImagePositionPatient = [x0, y0, float(zs[k])]
        ds.PixelSpacing = [dy, dx]
        ds.SliceThickness = dz
        ds.InstanceNumber = int(k) + 1
        ds.Rows, ds.Columns = ny, nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, -1024.0
        hu = phantom_hu(px, py, np.full_like(px, zs[k]))
        ds.PixelData = np.rint(hu + 1024.0).astype(np.int16).tobytes()
        path = folder / f"CT_{ds.SOPInstanceUID[-12:]}.dcm"
        ds.save_as(str(path), enforce_file_format=True)
        ct_files.append(path)

    rs = _base(RT_STRUCT, "RTSTRUCT", study, frame)
    rs.StructureSetLabel = "Synthetic"
    ref = pydicom.Dataset()
    ref.FrameOfReferenceUID = frame
    ref_series = pydicom.Dataset()
    ref_series.SeriesInstanceUID = series
    ref_study = pydicom.Dataset()
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"  # Detached Study Management, as TPSs write
    ref_study.ReferencedSOPInstanceUID = study
    ref_study.RTReferencedSeriesSequence = [ref_series]
    ref.RTReferencedStudySequence = [ref_study]
    rs.ReferencedFrameOfReferenceSequence = [ref]
    half = BOX_MM / 2
    rois = [
        ("BODY", "EXTERNAL", (0, 255, 0), [[_square(-half, half, -half, half, z)] for z in zs if abs(z) < half]),
        ("PTV", "PTV", (255, 0, 0), [[_square(PTV[0][0], PTV[0][1], PTV[1][0], PTV[1][1], z)]
                                     for z in zs if PTV[2][0] <= z < PTV[2][1]]),
        # A 40 mm square with a 20 mm hole: even-odd must leave the hole empty.
        ("RING", "ORGAN", (0, 0, 255), [[_square(-50, -10, 20, 60, z), _square(-40, -20, 30, 50, z)]
                                        for z in zs if -10 <= z < 10]),
    ]
    rs.StructureSetROISequence, rs.ROIContourSequence, rs.RTROIObservationsSequence = [], [], []
    for n, (name, kind, color, slices) in enumerate(rois, start=1):
        roi = pydicom.Dataset()
        roi.ROINumber, roi.ROIName, roi.ReferencedFrameOfReferenceUID = n, name, frame
        rs.StructureSetROISequence.append(roi)
        item = pydicom.Dataset()
        item.ReferencedROINumber, item.ROIDisplayColor = n, list(color)
        item.ContourSequence = []
        for polys in slices:
            for poly in polys:
                c = pydicom.Dataset()
                c.ContourGeometricType = "CLOSED_PLANAR"
                c.NumberOfContourPoints = len(poly) // 3
                c.ContourData = poly
                item.ContourSequence.append(c)
        rs.ROIContourSequence.append(item)
        obs = pydicom.Dataset()
        obs.ObservationNumber, obs.ReferencedROINumber, obs.RTROIInterpretedType = n, n, kind
        rs.RTROIObservationsSequence.append(obs)
    struct_file = folder / "RS_synthetic.dcm"
    rs.save_as(str(struct_file), enforce_file_format=True)

    plan = _base(RT_ION_PLAN, "RTPLAN", study, frame)
    plan.RTPlanLabel = "SYNTH"
    plan.RTPlanGeometry = "PATIENT"
    setup = pydicom.Dataset()
    setup.PatientSetupNumber, setup.PatientPosition = 1, position
    plan.PatientSetupSequence = [setup]
    grid = np.arange(-12.0, 12.01, spot_pitch)
    sx, sy = (a.ravel() for a in np.meshgrid(grid, grid))
    beam = pydicom.Dataset()
    beam.BeamNumber, beam.BeamName, beam.TreatmentMachineName = 1, "B1", "SYNTH"
    beam.TreatmentDeliveryType, beam.ScanMode, beam.RadiationType = "TREATMENT", "MODULATED", "PROTON"
    beam.ReferencedPatientSetupNumber = 1
    beam.VirtualSourceAxisDistances = [2000.0, 2000.0]
    if range_shifter_wet > 0.0:
        shifter = pydicom.Dataset()
        shifter.RangeShifterNumber, shifter.RangeShifterID, shifter.RangeShifterType = 1, "RS1", "BINARY"
        beam.RangeShifterSequence = [shifter]
        beam.NumberOfRangeShifters = 1
    iso = [0.0, (PTV[1][0] + PTV[1][1]) / 2, 0.0]
    cps, cum = [], 0.0
    for n, energy in enumerate(energies):
        for last in (False, True):
            cp = pydicom.Dataset()
            cp.ControlPointIndex = len(cps)
            cp.NominalBeamEnergy = energy
            cp.ScanSpotTuneID = "3.0"
            cp.NumberOfScanSpotPositions = len(sx)
            cp.ScanSpotPositionMap = [float(v) for p in zip(sx, sy) for v in p]
            cp.ScanSpotMetersetWeights = [0.0 if last else 1.0] * len(sx)
            cp.CumulativeMetersetWeight = cum
            if not last:
                cum += float(len(sx))
            if not cps:
                cp.GantryAngle, cp.PatientSupportAngle = gantry, couch
                cp.IsocenterPosition = iso
                cp.SnoutPosition = 300.0
                if range_shifter_wet > 0.0:
                    s = pydicom.Dataset()
                    s.ReferencedRangeShifterNumber, s.RangeShifterSetting = 1, "IN"
                    s.IsocenterToRangeShifterDistance = 300.0
                    s.RangeShifterWaterEquivalentThickness = range_shifter_wet
                    cp.RangeShifterSettingsSequence = [s]
            cps.append(cp)
    beam.IonControlPointSequence = cps
    beam.NumberOfControlPoints = len(cps)
    beam.FinalCumulativeMetersetWeight = cum
    plan.IonBeamSequence = [beam]
    fg = pydicom.Dataset()
    fg.FractionGroupNumber, fg.NumberOfFractionsPlanned, fg.NumberOfBeams = 1, 1, 1
    rb = pydicom.Dataset()
    rb.ReferencedBeamNumber, rb.BeamMeterset = 1, cum * mu_per_spot
    fg.ReferencedBeamSequence = [rb]
    plan.FractionGroupSequence = [fg]
    ref = pydicom.Dataset()
    ref.DoseReferenceNumber, ref.DoseReferenceStructureType, ref.DoseReferenceType = 1, "SITE", "TARGET"
    ref.TargetPrescriptionDose = 2.0
    plan.DoseReferenceSequence = [ref]
    plan_file = folder / "RP_synthetic.dcm"
    plan.save_as(str(plan_file), enforce_file_format=True)
    return Phantom(folder, frame, study, tuple(ct_files), struct_file, plan_file, str(plan.SOPInstanceUID))
