"""GPU Monte Carlo dose: tables, energy bookkeeping, geometry and range, checked against the engine itself.

Agreement with MCsquare lives in ``validation/`` and is not run by pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scan_kit.views.dose_mc import load_tables, mc_media
from scan_kit.views.dose_volume_catalog import MC_MEDIA

ROOT = Path(__file__).resolve().parents[1]


def _grid(lateral, depth):
    from scan_kit.views.dose_volume_fill import DoseGrid

    return DoseGrid(np.array([-lateral / 2, -lateral / 2, -float(depth)]), (lateral, lateral, depth), False, 1.0)


def _dose(medium, spots, *, sigma=3.0, lateral=40, depth=60, wet=0.0, histories=100_000, seed=1):
    """(nz, ny, nx) Gy per proton on a 1 mm grid centred on the beam axis, z index 0 deepest."""
    from scan_kit.views.dose_mc import mc_dose

    x, y, e, w = (np.array(c, dtype=float) for c in zip(*spots))
    return mc_dose(
        x, y, np.full_like(x, sigma), np.full_like(x, sigma), e, w, medium, _grid(lateral, depth),
        depth=float(depth), wet=float(wet), spread_pct=0.7, histories=histories, seed=seed,
    )


def test_tables_match_mcsquare_data() -> None:
    t = load_tables()
    n = len(mc_media())
    assert mc_media()[: len(MC_MEDIA)] == MC_MEDIA and n < 256
    assert t["m_stop"].shape == (n, 801) and t["m_nuc"].shape == (n, 250)
    water = list(t["media"]).index("water")
    g4 = ROOT / "third_party" / "MCsquare" / "Materials" / "Water" / "G4_Stop_Pow.dat"
    if g4.is_file():
        row = np.loadtxt(g4)[200]
        assert row[0] == 100.0 and t["m_stop"][water, 200] == pytest.approx(row[1] * 1e6, rel=1e-6)
    # Water's total nuclear cross section at 150 MeV is its H (p-p) and O (ICRU) parts.
    elements = [str(e) for e in t["elements"]]
    comp = [elements[c] for c in t["m_comp"][water] if c >= 0]
    frac = dict(zip(comp, t["m_frac"][water]))
    o = elements.index("oxygen")
    el = slice(t["el_start"][o], t["el_start"][o + 1])
    inel = slice(t["in_start"][o], t["in_start"][o + 1])
    oxygen = np.interp(150.0, t["el_E"][el], t["el_sigma"][el]) + np.interp(150.0, t["in_E"][inel], t["in_sigma"][inel])
    pp = (0.315 * 150.0**-1.126 + 3.78e-6 * 150.0) / 0.1119
    assert t["m_nuc"][water, 150] == pytest.approx(frac["hydrogen"] * pp + frac["oxygen"] * oxygen, rel=1e-5)


def test_energy_ledger_closes(gpu) -> None:
    _dose_vol, res = _dose("water", ((0.0, 0.0, 150.0, 1.0),), sigma=4.0, lateral=100, depth=180, wet=30.0)
    assert res.overflow == 0 and res.deepest >= 1
    assert abs(res.closure) < 1e-4
    assert res.ledger["incident"] == pytest.approx(150.0, rel=1e-3)
    # 30 mm of entrance water sits outside the grid and takes ~12 % of 150 MeV.
    assert 0.1 < res.fraction("off_grid") < 0.14 and res.fraction("grid") > 0.8 and res.ledger["lost"] > 0.0


def test_grazing_particles_cross_voxel_faces_in_air(gpu) -> None:
    """A particle on a face moving nearly along it must still cross: in air nothing scatters it off,
    and a 2 µm nudge below float32 resolution left it crawling to the 100k step cap."""
    from scan_kit.dicom import gantry_to_patient
    from scan_kit.qa import BeamModel
    from scan_kit.views.dose_mc import McRun, beam_record

    model = BeamModel.read("BDL_default_DN_RangeShifter")
    n, sp = 100, 2.0
    air = np.full((n, n, n), mc_media().index("water"), np.uint8)
    beam = beam_record(gantry_to_patient(0.0, 0.0, "HFS"), (n * sp / 2,) * 3, model.nozzle_to_iso,
                       model.smx_to_iso, model.smy_to_iso)
    spots = model.spot_records(150.0, np.zeros(1), np.zeros(1), beam=0, rs_id="", rs_wet=0.0, rs_distance=0.0)
    run = McRun.patient(spots, np.ones(1), beam[None], air, np.full(air.shape, 0.0012, np.float32), (sp,) * 3,
                        histories=50_000, seed=1, dose_to_water=False)
    run.step()
    run.close()
    # 200 mm at 2 mm voxels is ~100 face crossings; the crawl took 100000 steps.
    assert 0 < run.result.longest < 2_000


def test_same_seed_is_bit_identical(gpu) -> None:
    spots = ((0.0, 0.0, 100.0, 1.0),)
    a, _ = _dose("pmma", spots, histories=50_000, depth=90, seed=7)
    b, _ = _dose("pmma", spots, histories=50_000, depth=90, seed=7)
    c, _ = _dose("pmma", spots, histories=50_000, depth=90, seed=8)
    assert a.max() > 0.0 and np.array_equal(a, b) and not np.array_equal(a, c)


TWO_SPOTS = ((0.0, 0.0, 100.0, 1.0), (6.0, -4.0, 120.0, 2.0))


def _two_spot_run(histories):
    """McRun of TWO_SPOTS on the grid ``_dose(..., depth=110)`` uses."""
    from scan_kit.views.dose_mc import McRun

    x, y, e, w = (np.array(c, dtype=float) for c in zip(*TWO_SPOTS))
    return McRun.slab(
        x, y, np.full(2, 3.0), np.full(2, 3.0), e, w, "water", _grid(40, 110),
        depth=110.0, wet=0.0, spread_pct=0.7, histories=histories, seed=1,
    )


def test_sliced_run_matches_one_shot(gpu) -> None:
    one, whole = _dose("water", TWO_SPOTS, depth=110, histories=60_000)
    run = _two_spot_run(60_000)
    run.step(1e-4)
    partial = run.preview()
    assert 0.0 < run.progress < 1.0 and partial.max() > 0.0
    while not run.step(1e-4):
        run.preview()
    assert np.array_equal(run.dose, one) and run.result == whole
    run.close()


def test_raised_target_resumes_a_finished_run(gpu) -> None:
    one, whole = _dose("water", TWO_SPOTS, depth=110, histories=60_000)
    run = _two_spot_run(30_000)
    run.step()
    assert run.done and not run.extend(20_000)
    assert run.extend(60_000) and run.progress == 0.5
    run.step()
    # Same histories as the one-shot run, folded in smaller batches.
    np.testing.assert_allclose(run.dose, one, rtol=1e-4, atol=1e-6 * one.max())
    assert run.result.histories == 60_000 and run.result.ledger == pytest.approx(whole.ledger, rel=1e-6)
    run.close()
    assert not run.extend(90_000)


def test_spot_lands_where_planned(gpu) -> None:
    vol, _ = _dose("water", ((12.0, -7.0, 100.0, 1.0),), lateral=60, depth=90)
    c = np.arange(60) + 0.5 - 30.0
    px, py = vol.sum(axis=(0, 1)), vol.sum(axis=(0, 2))
    x, y = (px * c).sum() / px.sum(), (py * c).sum() / py.sum()
    assert x == pytest.approx(12.0, abs=0.3) and y == pytest.approx(-7.0, abs=0.3)
    sx, sy = np.sqrt((px * (c - x) ** 2).sum() / px.sum()), np.sqrt((py * (c - y) ** 2).sum() / py.sum())
    assert sx == pytest.approx(sy, rel=0.05)


def test_range_matches_its_own_stopping_powers(gpu) -> None:
    t = load_tables()
    m = list(t["media"]).index("water")
    stop = t["m_stop"][m] * float(t["m_props"][m, 0]) / 1e6  # MeV/cm
    fine = np.linspace(0.5, 150.0, 2000)
    csda = np.trapezoid(1.0 / np.interp(fine, np.arange(stop.size) * 0.5, stop), fine) * 10.0
    vol, _ = _dose("water", ((0.0, 0.0, 150.0, 1.0),), depth=180)
    idd = vol[::-1].sum(axis=(1, 2))
    peak = int(np.argmax(idd))
    level = 0.8 * idd[peak]
    i = peak + int(np.argmax(idd[peak:] < level)) - 1
    r80 = i + 0.5 + (idd[i] - level) / (idd[i] - idd[i + 1])
    assert r80 == pytest.approx(csda, rel=0.01)
