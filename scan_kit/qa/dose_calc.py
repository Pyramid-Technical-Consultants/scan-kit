"""Patient dose on the GPU Monte Carlo: planned or delivered spots through the planning CT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dicom.calibration import CtCalibration
from ..dicom.ct import CtImage
from ..dicom.grid import VolumeGrid
from ..dicom.plan import IonPlan
from ..views.dose_mc import McRun, beam_record
from .beam_model import BeamModel

BODY_HU = -900.0  # voxels above this bound the transported volume
CROP_MARGIN = 2  # voxels kept around them


@dataclass(frozen=True)
class SpotSet:
    """Spots in plan terms, one entry per spot: IEC beam-limiting-device X/Y (mm) at the
    isocenter plane, nominal energy (MeV), MU, and the range shifter each passes."""

    beam: np.ndarray  # beam number
    energy: np.ndarray
    x: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    rs_id: np.ndarray  # str, "" when out
    rs_wet: np.ndarray  # mm
    rs_distance: np.ndarray  # mm, isocenter to the shifter's downstream face

    @classmethod
    def concat(cls, parts) -> SpotSet:
        parts = list(parts)
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in cls.__dataclass_fields__))

    def select(self, keep) -> SpotSet:
        return SpotSet(*(getattr(self, f)[keep] for f in self.__dataclass_fields__))

    def __len__(self) -> int:
        return len(self.mu)


def plan_spots(plan: IonPlan, beams=None) -> SpotSet:
    """Every planned spot of *beams* (numbers; all by default) for one fraction."""
    parts = []
    for b in plan.beams:
        if beams is not None and b.number not in beams:
            continue
        for layer in b.layers:
            n = len(layer.mu)
            parts.append(SpotSet(
                np.full(n, b.number), np.full(n, layer.energy), np.asarray(layer.x, float), np.asarray(layer.y, float),
                np.asarray(layer.mu, float), np.full(n, layer.range_shifter, dtype=object),
                np.full(n, layer.range_shifter_wet), np.full(n, layer.range_shifter_distance),
            ))
    if not parts:
        raise ValueError("no spots in the selected beams")
    return SpotSet.concat(parts)


def body_box(hu: np.ndarray) -> tuple[slice, slice, slice]:
    """(z, y, x) slices bounding the voxels above :data:`BODY_HU`, plus a margin."""
    body = hu > BODY_HU
    if not body.any():
        return tuple(slice(0, n) for n in hu.shape)
    out = []
    for axis, n in enumerate(hu.shape):
        i = np.flatnonzero(body.any(axis=tuple(a for a in range(3) if a != axis)))
        out.append(slice(max(int(i[0]) - CROP_MARGIN, 0), min(int(i[-1]) + 1 + CROP_MARGIN, n)))
    return tuple(out)


def sub_grid(grid: VolumeGrid, box) -> VolumeGrid:
    lo = np.array([box[2].start, box[1].start, box[0].start], dtype=float)
    shape = (box[2].stop - box[2].start, box[1].stop - box[1].start, box[0].stop - box[0].start)
    return VolumeGrid(grid.to_patient(lo), grid.spacing, shape, grid.axes, grid.frame_uid)


def patient_run(
    ct: CtImage, calibration: CtCalibration, model: BeamModel, plan: IonPlan, spots: SpotSet, *,
    histories: int, seed: int, dose_to_water: bool = True, let: bool = False, crop: bool = True,
) -> tuple[McRun, VolumeGrid]:
    """A Monte Carlo run of *spots* on *ct*, and the grid its dose lands on (the CT, or
    the CT cropped to the patient). Dose is Gy for the MU given."""
    box = body_box(ct.hu) if crop else tuple(slice(0, n) for n in ct.hu.shape)
    grid = sub_grid(ct.grid, box)
    material, density = calibration.voxels(ct.hu[box])
    numbers = sorted({int(n) for n in spots.beam})
    beams = np.stack([
        beam_record(grid.axes.T @ plan.beam(n).gantry_to_patient(), grid.to_local(plan.beam(n).isocenter),
                    model.nozzle_to_iso, model.smx_to_iso, model.smy_to_iso)
        for n in numbers
    ])
    index = np.searchsorted(numbers, spots.beam.astype(int))
    records, protons = [], []
    for rs in sorted({str(r) for r in spots.rs_id}):
        sel = np.array([str(r) == rs for r in spots.rs_id])
        records.append(model.spot_records(
            spots.energy[sel], spots.x[sel], spots.y[sel], beam=index[sel], rs_id=rs,
            rs_wet=spots.rs_wet[sel] if rs else 0.0, rs_distance=spots.rs_distance[sel],
        ))
        protons.append(spots.mu[sel] * model.protons_per_mu(spots.energy[sel]))
    run = McRun.patient(
        np.concatenate(records), np.concatenate(protons), beams, material, density, grid.spacing,
        histories=histories, seed=seed, dose_to_water=dose_to_water, let=let,
    )
    return run, grid
