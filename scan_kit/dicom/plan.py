"""RT Ion Plan geometry: beams, energy layers, spots, range shifters, IEC 61217 frames."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ct import DicomError

RT_ION_PLAN = "1.2.840.10008.5.1.4.1.1.481.8"

# IEC 61217 patient support (X_s, Y_s, Z_s) to patient LPS, per PatientPosition.
_SUPPORT_TO_LPS = {
    "HFS": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "HFP": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "FFS": ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "FFP": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}


@dataclass(frozen=True)
class IonLayer:
    energy: float  # nominal MeV
    x: np.ndarray  # spot positions at the isocenter plane, IEC beam limiting device X_b (mm)
    y: np.ndarray
    mu: np.ndarray  # MU per spot
    tune_id: str = ""
    range_shifter: str = ""  # ReferencedRangeShifterNumber's ID when IN, else ""
    # mm water equivalent from RangeShifterWaterEquivalentThickness; NaN when IN without one
    range_shifter_wet: float = 0.0
    range_shifter_distance: float = math.nan  # IsocenterToRangeShifterDistance (mm)
    snout_position: float = math.nan
    spot_size: tuple[float, float] | None = None  # ScanningSpotSize (FWHM mm) when the plan gives it


@dataclass(frozen=True)
class IonBeam:
    number: int
    name: str
    gantry: float  # degrees, IEC 61217
    couch: float  # patient support angle, degrees
    isocenter: np.ndarray  # patient LPS mm
    patient_position: str
    meterset: float  # BeamMeterset MU for one fraction
    machine: str
    layers: tuple[IonLayer, ...]
    range_shifters: tuple[str, ...] = ()
    virtual_sad: tuple[float, float] | None = None  # VirtualSourceAxisDistances (mm)

    @property
    def spots(self) -> int:
        return sum(len(layer.mu) for layer in self.layers)

    def gantry_to_patient(self) -> np.ndarray:
        return gantry_to_patient(self.gantry, self.couch, self.patient_position)


@dataclass(frozen=True)
class IonPlan:
    sop_uid: str
    label: str
    frame_uid: str
    fractions: int
    beams: tuple[IonBeam, ...]
    study_uid: str = ""
    patient_id: str = ""
    prescription: float = math.nan  # course Gy, the largest TargetPrescriptionDose

    def beam(self, number: int) -> IonBeam:
        for b in self.beams:
            if b.number == number:
                return b
        raise KeyError(number)


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gantry_to_patient(gantry: float, couch: float, patient_position: str = "HFS") -> np.ndarray:
    """Rotation from the IEC 61217 gantry frame (Z_g toward the source) to patient LPS.

    Gantry about Y_f, then the support's rotation about Z_f, then PatientPosition.
    """
    try:
        m = np.array(_SUPPORT_TO_LPS[patient_position.upper()], dtype=float)
    except KeyError:
        raise DicomError(f"patient position {patient_position!r} is not supported (HFS, HFP, FFS, FFP)") from None
    return m @ _rot_z(-couch) @ _rot_y(gantry)


def _f(item, key, default=math.nan) -> float:
    v = item.get(key, None)
    return default if v is None or v == "" else float(v)


def plan_from_dataset(ds) -> IonPlan:
    if str(ds.get("SOPClassUID", "")) != RT_ION_PLAN:
        raise DicomError("not an RT Ion Plan")
    setups = {int(s.PatientSetupNumber): str(s.get("PatientPosition", "") or "HFS")
              for s in ds.get("PatientSetupSequence", [])}
    fraction = next(iter(ds.get("FractionGroupSequence", [])), None)
    fractions = int(fraction.get("NumberOfFractionsPlanned", 1) or 1) if fraction is not None else 1
    metersets = {}
    if fraction is not None:
        for ref in fraction.get("ReferencedBeamSequence", []):
            metersets[int(ref.ReferencedBeamNumber)] = _f(ref, "BeamMeterset", 0.0)
    beams = []
    for b in ds.get("IonBeamSequence", []):
        if str(b.get("TreatmentDeliveryType", "TREATMENT")) != "TREATMENT":
            continue
        if str(b.get("ScanMode", "MODULATED")) not in ("MODULATED", "MODULATED_SPEC"):
            raise DicomError(f"beam {b.BeamNumber}: only pencil-beam scanning (MODULATED) plans are supported")
        if fraction is not None and int(b.BeamNumber) not in metersets:
            continue  # not delivered in this fraction group
        shifters = {int(r.RangeShifterNumber): str(r.get("RangeShifterID", "") or f"RS{r.RangeShifterNumber}")
                    for r in b.get("RangeShifterSequence", [])}
        final = _f(b, "FinalCumulativeMetersetWeight", 0.0)
        meterset = metersets.get(int(b.BeamNumber), 0.0)
        if not meterset > 0.0 or not final > 0.0:
            raise DicomError(f"beam {b.BeamNumber} has no BeamMeterset or FinalCumulativeMetersetWeight")
        mu_per_weight = meterset / final
        # Control point attributes persist until a later control point changes them; shifters_in
        # holds (WET, isocenter distance) of each shifter number while it is IN.
        state = dict(energy=math.nan, gantry=0.0, couch=0.0, iso=None, tune="", snout=math.nan, shifters_in={})
        layers = []
        for cp in b.get("IonControlPointSequence", []):
            state["energy"] = _f(cp, "NominalBeamEnergy", state["energy"])
            state["gantry"] = _f(cp, "GantryAngle", state["gantry"])
            state["couch"] = _f(cp, "PatientSupportAngle", state["couch"])
            state["snout"] = _f(cp, "SnoutPosition", state["snout"])
            if "IsocenterPosition" in cp:
                state["iso"] = np.asarray(cp.IsocenterPosition, dtype=float)
            if "ScanSpotTuneID" in cp:
                state["tune"] = str(cp.ScanSpotTuneID)
            for rs in cp.get("RangeShifterSettingsSequence", []):
                number = int(rs.ReferencedRangeShifterNumber)
                if number not in shifters:
                    raise DicomError(f"beam {b.BeamNumber}: range shifter {number} is not in its RangeShifterSequence")
                if str(rs.get("RangeShifterSetting", "OUT")) == "IN":
                    wet, dist = state["shifters_in"].get(number, (math.nan, math.nan))
                    state["shifters_in"][number] = (_f(rs, "RangeShifterWaterEquivalentThickness", wet),
                                                    _f(rs, "IsocenterToRangeShifterDistance", dist))
                else:
                    state["shifters_in"].pop(number, None)
            n = int(cp.get("NumberOfScanSpotPositions", 0) or 0)
            if not n or "ScanSpotPositionMap" not in cp:
                continue
            xy = np.asarray(cp.ScanSpotPositionMap, dtype=float).reshape(-1, 2)[:n]
            w = np.atleast_1d(np.asarray(cp.get("ScanSpotMetersetWeights", np.zeros(n)), dtype=float))[:n]
            keep = np.isfinite(w) & (w > 0.0)
            if not keep.any():
                continue
            if not math.isfinite(state["energy"]):
                raise DicomError(f"beam {b.BeamNumber}: spots before any NominalBeamEnergy")
            if len(state["shifters_in"]) > 1:
                raise DicomError(f"beam {b.BeamNumber}: more than one range shifter IN is not supported")
            rs_number, (wet, rs_dist) = next(iter(state["shifters_in"].items()), (None, (0.0, math.nan)))
            size = cp.get("ScanningSpotSize", None)
            layers.append(IonLayer(
                energy=state["energy"], x=xy[keep, 0], y=xy[keep, 1], mu=w[keep] * mu_per_weight,
                tune_id=state["tune"], range_shifter=shifters.get(rs_number, ""), range_shifter_wet=wet,
                range_shifter_distance=rs_dist, snout_position=state["snout"],
                spot_size=tuple(float(v) for v in size) if size is not None else None,
            ))
        if state["iso"] is None:
            raise DicomError(f"beam {b.BeamNumber} has no IsocenterPosition")
        sad = b.get("VirtualSourceAxisDistances", None)
        beams.append(IonBeam(
            number=int(b.BeamNumber), name=str(b.get("BeamName", "") or f"Beam {b.BeamNumber}"),
            gantry=state["gantry"], couch=state["couch"], isocenter=state["iso"],
            patient_position=setups.get(int(b.get("ReferencedPatientSetupNumber", 1) or 1), "HFS"),
            meterset=meterset, machine=str(b.get("TreatmentMachineName", "") or ""), layers=tuple(layers),
            range_shifters=tuple(shifters.values()),
            virtual_sad=tuple(float(v) for v in sad) if sad is not None else None,
        ))
    rx = [_f(r, "TargetPrescriptionDose") for r in ds.get("DoseReferenceSequence", [])]
    return IonPlan(
        sop_uid=str(ds.SOPInstanceUID), label=str(ds.get("RTPlanLabel", "") or ""),
        frame_uid=str(ds.get("FrameOfReferenceUID", "") or ""), fractions=fractions, beams=tuple(beams),
        study_uid=str(ds.get("StudyInstanceUID", "")), patient_id=str(ds.get("PatientID", "")),
        prescription=max((v for v in rx if math.isfinite(v)), default=math.nan),
    )


def load_ion_plan(path: str | Path) -> IonPlan:
    import pydicom

    return plan_from_dataset(pydicom.dcmread(str(path), force=True))
