"""Index a folder of DICOM and load one patient's CT, structures, plans and doses together."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .ct import CtImage, DicomError, load_ct
from .dose import RT_DOSE, DoseImage, load_dose
from .plan import RT_ION_PLAN, IonPlan, load_ion_plan
from .structures import StructureSet, load_structures

CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
ENHANCED_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2.1"
RT_STRUCT = "1.2.840.10008.5.1.4.1.1.481.3"
_HEADER_TAGS = ("SOPClassUID", "SOPInstanceUID", "Modality", "StudyInstanceUID", "SeriesInstanceUID",
                "FrameOfReferenceUID", "ReferencedFrameOfReferenceSequence", "StructureSetROISequence")


@dataclass(frozen=True)
class DicomEntry:
    path: Path
    sop_class: str
    sop_uid: str
    modality: str
    study_uid: str
    series_uid: str
    frame_uid: str
    ref_series: str = ""  # an RTSTRUCT's contoured CT series


def _ref_series(ds) -> str:
    for fr in ds.get("ReferencedFrameOfReferenceSequence", []):
        for st in fr.get("RTReferencedStudySequence", []):
            for se in st.get("RTReferencedSeriesSequence", []):
                return str(se.get("SeriesInstanceUID", "") or "")
    return ""


@dataclass
class PatientCase:
    """Everything on one frame of reference; inputs are only ever read."""

    frame_uid: str
    ct: CtImage
    structures: StructureSet | None = None
    plans: list[IonPlan] = field(default_factory=list)
    doses: list[DoseImage] = field(default_factory=list)

    def plan(self, sop_uid: str | None = None) -> IonPlan:
        if not self.plans:
            raise DicomError("no RT Ion Plan on this frame of reference")
        if sop_uid is None:
            return self.plans[0]
        return next(p for p in self.plans if p.sop_uid == sop_uid)

    def doses_for(self, plan: IonPlan) -> list[DoseImage]:
        return [d for d in self.doses if d.plan_uid == plan.sop_uid]


def _frame_of(ds) -> str:
    uid = str(ds.get("FrameOfReferenceUID", "") or "")
    if uid:
        return uid
    for ref in ds.get("ReferencedFrameOfReferenceSequence", []):
        return str(ref.FrameOfReferenceUID)
    for roi in ds.get("StructureSetROISequence", []):
        if "ReferencedFrameOfReferenceUID" in roi:
            return str(roi.ReferencedFrameOfReferenceUID)
    return ""


class StudyIndex:
    """Headers of every readable DICOM file under a folder (pixels are not read)."""

    def __init__(self, entries: list[DicomEntry]) -> None:
        self.entries = entries

    @classmethod
    def scan(cls, root: str | Path) -> StudyIndex:
        import pydicom
        from pydicom.errors import InvalidDicomError

        entries = []
        root = Path(root)
        for path in sorted(p for p in (root.rglob("*") if root.is_dir() else [root]) if p.is_file()):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=list(_HEADER_TAGS))
            except (InvalidDicomError, OSError, ValueError, TypeError, AttributeError):
                continue
            if "SOPClassUID" not in ds:
                continue
            entries.append(DicomEntry(
                path=path, sop_class=str(ds.SOPClassUID), sop_uid=str(ds.get("SOPInstanceUID", "")),
                modality=str(ds.get("Modality", "")), study_uid=str(ds.get("StudyInstanceUID", "")),
                series_uid=str(ds.get("SeriesInstanceUID", "")), frame_uid=_frame_of(ds), ref_series=_ref_series(ds),
            ))
        return cls(entries)

    def frames(self) -> dict[str, list[DicomEntry]]:
        """Entries by frame of reference UID; an entry without one files under ``""``."""
        out: dict[str, list[DicomEntry]] = defaultdict(list)
        for e in self.entries:
            out[e.frame_uid].append(e)
        return dict(out)

    def ct_series(self, frame_uid: str) -> dict[str, list[DicomEntry]]:
        out: dict[str, list[DicomEntry]] = defaultdict(list)
        for e in self.entries:
            if e.frame_uid == frame_uid and e.sop_class == CT_IMAGE:
                out[e.series_uid].append(e)
        return dict(out)

    def load(self, frame_uid: str | None = None) -> PatientCase:
        """The case on *frame_uid* (the only one with a CT when None).

        The planning CT is the series the RTSTRUCT was contoured on, else the one largest
        series. A plan or dose on another frame of reference is refused; a plan without a
        frame UID is accepted, its dose checks it.
        """
        with_ct = sorted({e.frame_uid for e in self.entries if e.sop_class == CT_IMAGE})
        if frame_uid is None:
            if len(with_ct) != 1:
                raise DicomError(f"{len(with_ct)} frames of reference have a CT; pick one")
            frame_uid = with_ct[0]
        series = self.ct_series(frame_uid)
        if not series:
            raise DicomError("no CT on this frame of reference")
        struct = next((e for e in self.entries if e.sop_class == RT_STRUCT and e.frame_uid == frame_uid), None)
        chosen = series.get(struct.ref_series) if struct is not None else None
        if chosen is None:
            top = max(len(s) for s in series.values())
            largest = [s for s in series.values() if len(s) == top]
            if len(largest) > 1:
                raise DicomError(f"{len(largest)} CT series of {top} slices on this frame and no RTSTRUCT names one")
            chosen = largest[0]
        ct = load_ct(e.path for e in chosen)
        case = PatientCase(frame_uid, ct)
        if struct is not None:
            case.structures = load_structures(struct.path)
        for e in self.entries:
            if e.sop_class == RT_ION_PLAN and e.frame_uid in (frame_uid, ""):
                case.plans.append(load_ion_plan(e.path))
        for e in self.entries:
            if e.sop_class == RT_DOSE and e.frame_uid == frame_uid:
                case.doses.append(load_dose(e.path))
        _check_frames(case)
        return case


def _check_frames(case: PatientCase) -> None:
    f = case.frame_uid
    if case.structures is not None and case.structures.frame_uid not in (f, ""):
        raise DicomError("RTSTRUCT is on another frame of reference than the CT")
    for p in case.plans:
        if p.frame_uid not in (f, ""):
            raise DicomError(f"plan {p.label!r} is on another frame of reference than the CT")
    for d in case.doses:
        if d.grid.frame_uid not in (f, ""):
            raise DicomError("an RTDOSE is on another frame of reference than the CT")
