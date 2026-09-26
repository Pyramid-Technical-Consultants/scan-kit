"""Patient QA: beam model, the IEC 61217 chain against MCsquare's, and patient transport on a synthetic CT."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scan_kit.dicom import StudyIndex, gantry_to_patient
from scan_kit.dicom.calibration import CtCalibration
from scan_kit.dicom.structures import rasterize
from scan_kit.dicom.synthetic import write_phantom
from scan_kit.qa import BeamModel, Delivery, fraction_runs, match_delivery, patient_run, plan_spots
from scan_kit.qa.beam_model import _phase_space

BDL = "BDL_default_DN_RangeShifter"


def test_beam_model_interpolates_like_mcsquare() -> None:
    m = BeamModel.read(BDL)
    t = m.table
    assert m.nozzle_to_iso == 410.0 and m.range_shifters[0].material == 64
    assert m.at(t["NominalEnergy"][3])["MeanEnergy"] == pytest.approx(t["MeanEnergy"][3])
    mid = 0.5 * (t["NominalEnergy"][3] + t["NominalEnergy"][4])
    assert m.protons_per_mu(mid) == pytest.approx(0.5 * (t["ProtonsMU"][3] + t["ProtonsMU"][4]))
    T, s = _phase_space([4.0, 3.0], [0.006, 0.004], [0.3, 0.0])
    for k, (size, div, corr) in enumerate(((4.0, 0.006, 0.3), (3.0, 0.004, 0.0))):
        A = np.array([[size**2, corr * size * div], [corr * size * div, div**2]])
        assert np.allclose(T[k] @ np.diag(s[k] ** 2) @ T[k].T, A, rtol=1e-9, atol=1e-15)
        assert s[k, 0] >= s[k, 1]


@pytest.mark.parametrize("gantry,couch", [(0, 0), (45, 10), (270, -30)])
def test_iec_chain_is_mcsquares_on_a_flipped_y_ct(gantry, couch) -> None:
    g, c = math.radians(gantry), math.radians(-couch)
    ry = np.array([[math.cos(g), 0, math.sin(g)], [0, 1, 0], [-math.sin(g), 0, math.cos(g)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    swap = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # BEV_to_CT_frame's XYZ -> XZY
    assert np.allclose(np.diag([1, -1, 1]) @ gantry_to_patient(gantry, couch, "HFS"), swap @ rz @ ry)


@pytest.fixture(scope="module")
def phantom(tmp_path_factory):
    def load(**kw):
        case = StudyIndex.scan(write_phantom(tmp_path_factory.mktemp("qa"), **kw).folder).load()
        return case, case.plan()

    return load


def _run(case, plan, histories=200_000, seed=3, **kw):
    return patient_run(case.ct, CtCalibration.mcsquare("default"), BeamModel.read(BDL), plan, plan_spots(plan),
                       histories=histories, seed=seed, **kw)


def test_patient_transport_closes_and_hits_the_target(gpu, phantom) -> None:
    case, plan = phantom()
    run, grid = _run(case, plan, let=True)
    run.step()
    r = run.result
    assert abs(r.closure) < 1e-6 and r.fraction("beamline") < 0.01 and r.overflow == 0
    ptv = rasterize(case.structures.roi("PTV"), grid)
    assert run.dose[ptv].mean() > 5.0 * run.dose[~ptv].mean()
    assert 1.0 < float(run.let[ptv].mean()) < 10.0
    # The run is cropped to the patient.
    assert grid.shape[0] < case.ct.grid.shape[0]


def test_patient_transport_is_deterministic_in_any_slicing(gpu, phantom) -> None:
    case, plan = phantom(gantry=45.0, couch=10.0)
    whole, _ = _run(case, plan, histories=100_000)
    whole.step()
    sliced, _ = _run(case, plan, histories=100_000)
    while not sliced.step(0.002):
        pass
    assert np.array_equal(whole.dose, sliced.dose)
    assert whole.result.ledger == sliced.result.ledger


def _as_delivered(plan):
    s = plan_spots(plan)
    return Delivery(s.x.copy(), s.y.copy(), s.energy.copy(), s.mu.copy())


def test_delivery_matches_plan_layers_and_reports_errors(phantom) -> None:
    _case, plan = phantom()
    d = _as_delivered(plan)
    exact = match_delivery(plan, d)
    assert exact.beam == plan.beams[0].number and exact.unmatched == 0
    assert exact.position_rms == 0.0 and np.allclose(exact.layer_mu_ratio, 1.0)
    first = d.energy == d.energy[0]
    off = Delivery(d.x + 2.0, d.y, d.energy + 0.2 * first, np.where(first, 1.05, 1.0) * d.mu)
    stray = Delivery(np.r_[off.x, 0.0], np.r_[off.y, 0.0], np.r_[off.energy, 60.0], np.r_[off.mu, 1.0])
    m = match_delivery(plan, stray)
    assert m.unmatched == 1 and m.energy_error == pytest.approx(0.2)
    assert m.position_rms == pytest.approx(2.0, abs=1e-9)
    assert m.layer_mu_ratio[0] == pytest.approx(1.05) and np.allclose(m.layer_mu_ratio[1:], 1.0)
    assert np.array_equal(m.spots.energy, plan_spots(plan).energy)
    # A layer split in two at one energy: each spot goes to the half that planned it.
    import dataclasses

    b = plan.beams[0]
    ly = b.layers[0]
    left = ly.x < 0
    halves = [dataclasses.replace(ly, x=ly.x[k], y=ly.y[k], mu=ly.mu[k]) for k in (left, ~left)]
    split = dataclasses.replace(plan, beams=(dataclasses.replace(b, layers=(*halves, *b.layers[1:])),))
    assert np.allclose(match_delivery(split, d).layer_mu_ratio, 1.0)


def test_sessions_split_into_fractions_when_a_beam_repeats() -> None:
    from types import SimpleNamespace

    from scan_kit.qa import group_fractions

    logs = [SimpleNamespace(beam=b) for b in (1, 2, 1, 2, 3, 2)]
    assert [[m.beam for m in f] for f in group_fractions(logs)] == [[1, 2], [1, 2, 3], [2]]


def test_delivered_as_planned_is_the_planned_dose(gpu, phantom) -> None:
    case, plan = phantom()
    runs = fraction_runs(case.ct, CtCalibration.mcsquare("default"), BeamModel.read(BDL), plan,
                         [match_delivery(plan, _as_delivered(plan))], histories=50_000, seed=5)
    for run in runs[:2]:
        run.step()
    assert np.array_equal(runs[0].dose, runs[1].dose) and runs[0].dose.shape == runs[2].shape_zyx


def test_range_shifter_takes_energy_upstream(gpu, phantom) -> None:
    case, plan = phantom(range_shifter_wet=40.0)
    run, _ = _run(case, plan, histories=100_000)
    run.step()
    assert 0.1 < run.result.fraction("beamline") < 0.4 and abs(run.result.closure) < 1e-6


def test_dvh_and_goals_on_a_ramp(gpu, phantom) -> None:
    from scan_kit.gpu.dvh import _histograms_cpu, histograms
    from scan_kit.qa.analysis import Goal, dvhs

    case, _plan = phantom()
    grid = case.ct.grid
    dose = np.broadcast_to(np.arange(grid.shape[0], dtype=np.float32) * 0.1, grid.shape_zyx).copy()
    ptv = case.structures.roi("PTV")
    inside = dose[rasterize(ptv, grid)]
    h = dvhs(dose, grid, [ptv, case.structures.roi("BODY")])["PTV"]
    assert h.voxels == inside.size and h.dmax == pytest.approx(inside.max()) and h.dmin == pytest.approx(inside.min())
    assert h.mean == pytest.approx(inside.mean(), abs=2 * dose.max() / 4096)
    assert h.dose_at(0.5) == pytest.approx(np.median(inside), abs=0.1)
    assert h.volume_at(float(np.median(inside))) == pytest.approx(0.5, abs=0.1)
    from scan_kit.dicom.structures import mask_bits

    bits = mask_bits([ptv], grid)
    got, cpu = histograms(dose, bits, 1, 256, 10.0), _histograms_cpu(dose.reshape(-1), bits.reshape(-1), 1, 256, 25.6)
    assert np.array_equal(got[0], cpu[0]) and np.allclose(got[1], cpu[1])
    rx = float(np.median(inside))
    assert Goal.parse("PTV: D50% >= 99%").check(h, rx)[1]
    assert not Goal.parse("PTV : Dmax < 1 Gy").check(h)[1]
    assert Goal.parse("PTV: V50% >= 90%").measure(h, rx) > 90.0
    assert Goal.parse("PTV: V0Gy > 1 cc").measure(h) == pytest.approx(h.volume_cc)
    assert Goal.parse("PTV: V0Gy > 1 CC").unit == "cc"
    for bad in ("PTV D95 big", "PTV: V20Gy < 30 Gy", "PTV: D95% > 3 cc"):
        with pytest.raises(ValueError):
            Goal.parse(bad)


def test_resample_is_exact_for_linear_dose(gpu, phantom) -> None:
    from scan_kit.dicom.grid import VolumeGrid
    from scan_kit.qa.analysis import _resample_cpu, resample

    case, _plan = phantom()
    grid = case.ct.grid
    lps = grid.to_patient(np.stack(np.meshgrid(*(np.arange(n) for n in grid.shape), indexing="ij"), -1))
    field = lambda p: 2.0 + 0.01 * p[..., 0] - 0.02 * p[..., 1] + 0.03 * p[..., 2]  # noqa: E731
    dose = field(lps).transpose(2, 1, 0).astype(np.float32)
    target = VolumeGrid(grid.origin + 7.3, (3.0, 2.5, 3.5), (30, 30, 25), frame_uid=grid.frame_uid)
    got = resample(dose, grid, target)
    t = target.to_patient(np.stack(np.meshgrid(*(np.arange(n) for n in target.shape), indexing="ij"), -1))
    assert np.allclose(got, field(t).transpose(2, 1, 0), atol=1e-4)
    assert np.allclose(got, _resample_cpu(dose, grid, target), atol=1e-4)


def test_gamma_against_exported_tps_dose(gpu, phantom, tmp_path) -> None:
    import json

    import pydicom

    from scan_kit.dicom import load_dose, write_dose
    from scan_kit.qa.analysis import gamma_vs_tps, provenance
    from scan_kit.views.dose_volume_physics import GammaCriteria

    case, plan = phantom()
    cal, model = CtCalibration.mcsquare("default"), BeamModel.read(BDL)
    run, grid = _run(case, plan, histories=200_000)
    run.step()
    prov = provenance(kind="planned", plan=plan, calibration=cal, model=model, result=run.result,
                      histories=run.histories, seed=3, case=case)
    assert prov["inputs"]["ct_images"] == len(case.ct.sop_uids) and prov["inputs"]["structures"]
    path = write_dose(tmp_path / "tps.dcm", run.dose * plan.fractions, grid, case.ct, plan_uid=plan.sop_uid,
                      provenance=prov)
    assert json.loads(pydicom.dcmread(path).ImageComments)["calibration"]["digest"] == cal.digest
    tps = load_dose(path)
    same = gamma_vs_tps(tps, run.dose, grid, GammaCriteria(2.0, 2.0, 10.0), fractions=plan.fractions)
    assert same.rate > 0.999
    shifted = type(grid)(grid.origin + np.array([0.0, 4.0, 0.0]), grid.spacing, grid.shape, grid.axes, grid.frame_uid)
    moved = gamma_vs_tps(tps, run.dose, shifted, GammaCriteria(2.0, 2.0, 10.0), fractions=plan.fractions)
    assert moved.rate < same.rate - 0.05
    # An EFFECTIVE TPS dose is Gy(RBE): physical and Gy(RBE) Monte Carlo both compare in it.
    import dataclasses

    from scan_kit.qa.analysis import RBE, tps_per_fraction

    eff = dataclasses.replace(tps, dose_type="EFFECTIVE", dose=tps.dose * np.float32(RBE))
    for dose, effective in ((run.dose, False), (run.dose * np.float32(RBE), True)):
        g = gamma_vs_tps(eff, dose, grid, GammaCriteria(2.0, 2.0, 10.0), fractions=plan.fractions, effective=effective)
        assert g.rate > 0.999
    # FRACTION is the whole fraction group (PS3.3); FRACTION_SESSION is one session.
    assert np.allclose(tps_per_fraction(dataclasses.replace(tps, summation="FRACTION"), 30), tps.dose / 30)
    assert np.array_equal(tps_per_fraction(dataclasses.replace(tps, summation="FRACTION_SESSION"), 30), tps.dose)
