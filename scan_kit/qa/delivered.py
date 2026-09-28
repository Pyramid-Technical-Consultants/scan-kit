"""Delivered dose: measured spots matched to the plan's layers, transported like the plan.

A :class:`Delivery` is one beam as the ionization chambers saw it: per spot, the position
projected to the isocenter plane, the layer energy and the MU the chambers counted.
:func:`match_delivery` snaps each spot to its planned layer so it inherits that layer's
beam, range shifter and nominal energy, and reports how far the delivery strayed.
:func:`fraction_runs` then transports the delivered and the planned spots of the same
beams on the same CT grid, so their difference is the delivery alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dicom.calibration import CtCalibration
from ..dicom.ct import CtImage
from ..dicom.grid import VolumeGrid
from ..dicom.plan import IonPlan
from ..views.dose_mc import McRun
from .beam_model import BeamModel
from .dose_calc import SpotSet, patient_run, plan_spots

ENERGY_TOL_MEV = 0.5


@dataclass(frozen=True)
class Delivery:
    x: np.ndarray  # mm at the isocenter plane, IEC beam limiting device X
    y: np.ndarray
    energy: np.ndarray  # MeV
    mu: np.ndarray  # MU the chambers counted
    label: str = ""


@dataclass(frozen=True)
class DeliveryMatch:
    beam: int
    spots: SpotSet  # the measured spots, in plan terms
    planned_mu: float
    delivered_mu: float
    unmatched: int  # spots no layer's energy was within tolerance of (dropped)
    energy_error: float  # MeV, worst |measured - layer| of the matched spots
    position_rms: float  # mm, measured spot to its nearest planned spot in the layer
    layer_mu_ratio: np.ndarray  # delivered / planned MU per layer, NaN for a layer not delivered


def delivery_from_session(session_id: str, base_dir: str, label: str = "") -> Delivery | None:
    """The spots of a logged session, positions on the IC2 -> IC1 ray at the isocenter."""
    from ..views.dose_volume_catalog import GRAIN_SPOT, XY_ISO_RAY
    from ..views.dose_volume_data import load_session_splat_source, measured_cloud

    source = load_session_splat_source(session_id, base_dir, GRAIN_SPOT)
    cloud = None if source is None else measured_cloud(source, XY_ISO_RAY, base_dir)
    if cloud is None or cloud.dose is None:
        return None
    return Delivery(cloud.x, cloud.y, cloud.energy, cloud.dose, label or session_id)


def _layer_energies(plan: IonPlan, beam: int) -> np.ndarray:
    return np.array([layer.energy for layer in plan.beam(beam).layers])


def pick_beam(plan: IonPlan, delivery: Delivery, tol: float = ENERGY_TOL_MEV) -> int:
    """The beam whose layers explain the most delivered spots (then the closest total MU)."""
    def score(b):
        e = _layer_energies(plan, b.number)
        hit = np.abs(delivery.energy[:, None] - e[None, :]).min(axis=1) <= tol
        return int(hit.sum()), -abs(float(sum(ly.mu.sum() for ly in b.layers)) - float(np.nansum(delivery.mu)))

    return max(plan.beams, key=score).number


def match_delivery(plan: IonPlan, delivery: Delivery, beam: int | None = None, tol: float = ENERGY_TOL_MEV) -> DeliveryMatch:
    beam = pick_beam(plan, delivery, tol) if beam is None else beam
    b = plan.beam(beam)
    energies = _layer_energies(plan, beam)
    ok = np.isfinite(delivery.x) & np.isfinite(delivery.y) & np.isfinite(delivery.mu) & (delivery.mu > 0.0)
    gap = np.abs(np.asarray(delivery.energy, float)[:, None] - energies[None, :])
    layer = gap.argmin(axis=1)
    tie = gap <= gap.min(axis=1, keepdims=True) + 1e-6
    split = tie.sum(axis=1) > 1
    if split.any():
        # Layers sharing an energy (split or repainted): a spot goes to the one with the nearest planned spot.
        # ponytail: repaints of one spot map tie again and go to the first; the log's order would part them.
        near = np.full(gap.shape, np.inf)
        for i in np.flatnonzero(tie[split].any(axis=0)):
            ly = b.layers[i]
            near[split, i] = np.hypot(delivery.x[split, None] - ly.x, delivery.y[split, None] - ly.y).min(axis=1)
        layer[split] = np.where(tie[split], near[split], np.inf).argmin(axis=1)
    ok &= gap.min(axis=1) <= tol
    idx = np.flatnonzero(ok)
    layers = [b.layers[i] for i in layer[idx]]
    spots = SpotSet(
        np.full(idx.size, beam), energies[layer[idx]], delivery.x[idx].astype(float), delivery.y[idx].astype(float),
        delivery.mu[idx].astype(float), np.array([ly.range_shifter for ly in layers], dtype=object),
        np.array([ly.range_shifter_wet for ly in layers]), np.array([ly.range_shifter_distance for ly in layers]),
    )
    miss = []
    ratio = np.full(len(b.layers), np.nan)
    for i, ly in enumerate(b.layers):
        mine = idx[layer[idx] == i]
        if not mine.size:
            continue
        ratio[i] = float(delivery.mu[mine].sum() / ly.mu.sum())
        d = np.hypot(delivery.x[mine, None] - ly.x[None, :], delivery.y[mine, None] - ly.y[None, :])
        miss.append(d.min(axis=1))
    miss = np.concatenate(miss) if miss else np.zeros(0)
    return DeliveryMatch(
        beam=beam, spots=spots, planned_mu=float(sum(ly.mu.sum() for ly in b.layers)),
        delivered_mu=float(spots.mu.sum()), unmatched=int((~ok).sum()),
        energy_error=float(gap.min(axis=1)[idx].max()) if idx.size else float("nan"),
        position_rms=float(np.sqrt(np.mean(miss**2))) if miss.size else float("nan"),
        layer_mu_ratio=ratio,
    )


def group_fractions(matches: list[DeliveryMatch]) -> list[list[DeliveryMatch]]:
    """Beam deliveries in log order, split into fractions wherever a beam comes round again.

    ponytail: a fraction that repeats a beam (an interrupted field) splits in two; group
    by session date once the logs carry the treatment record.
    """
    out: list[list[DeliveryMatch]] = []
    for m in matches:
        if not out or any(prev.beam == m.beam for prev in out[-1]):
            out.append([])
        out[-1].append(m)
    return out


def fraction_runs(
    ct: CtImage, calibration: CtCalibration, model: BeamModel, plan: IonPlan, matches: list[DeliveryMatch], *,
    histories: int, seed: int, dose_to_water: bool = True, let: bool = False,
) -> tuple[McRun, McRun, VolumeGrid]:
    """(delivered, planned) runs of the matched beam deliveries, on one grid; a beam
    delivered twice (two fractions) counts its planned spots twice."""
    if not matches:
        raise ValueError("no delivered beams to transport")
    delivered = SpotSet.concat(m.spots for m in matches)
    planned = SpotSet.concat(plan_spots(plan, beams={m.beam}) for m in matches)
    kw = dict(histories=histories, seed=seed, dose_to_water=dose_to_water, let=let)
    run_d, grid = patient_run(ct, calibration, model, plan, delivered, **kw)
    run_p, _ = patient_run(ct, calibration, model, plan, planned, **kw)
    return run_d, run_p, grid
