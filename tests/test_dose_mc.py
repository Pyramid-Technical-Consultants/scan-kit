"""GPU Monte Carlo dose: MCsquare tables, energy bookkeeping, and agreement with MCsquare goldens."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from scan_kit.views.dose_mc import load_tables, mc_media
from scan_kit.views.dose_volume_catalog import MC_MEDIA, MEDIUM_POLYETHYLENE, MODEL_ANALYTIC, MODEL_MC, WEIGHT_DOSE

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mcsquare_validate", ROOT / "scripts" / "mcsquare_validate.py")
mv = sys.modules.setdefault("mcsquare_validate", importlib.util.module_from_spec(_spec))
_spec.loader.exec_module(mv)

GOLDENS = sorted(mv.GOLDEN_DIR.glob("*.npz"))
# Everything else runs with -m slow.
QUICK = {"water_150_s3", "water_150_s4_wet30", "copper_100_s4"}
HISTORIES = 1_000_000


@pytest.fixture(scope="module")
def canvas(qapp):
    from scan_kit.views.dose_volume_raycast import _gl_at_least
    from scan_kit.views.vispy_plot import make_scene_canvas

    try:
        c = make_scene_canvas(size=(64, 64), show=False, gl="gl+")
        c.set_current()
    except Exception as exc:
        pytest.skip(f"visPy canvas unavailable: {exc}")
    if not _gl_at_least(4, 3):
        pytest.skip("OpenGL 4.3 compute unavailable")
    return c


def test_tables_match_mcsquare_data() -> None:
    t = load_tables()
    assert mc_media() == MC_MEDIA
    assert t["m_stop"].shape == (len(MC_MEDIA), 801) and t["m_nuc"].shape == (len(MC_MEDIA), 250)
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


def test_energy_ledger_closes(canvas) -> None:
    case = mv.Case("ledger", "water", 4.0, ((0.0, 0.0, 150.0, 1.0),), wet=30)
    _dose, _tex, res = mv.run_gpu(canvas, case, 100_000)
    assert res.overflow == 0 and res.deepest >= 1
    assert abs(res.closure) < 1e-4
    assert res.ledger["incident"] == pytest.approx(150.0, rel=1e-3)
    # 30 mm of entrance water sits outside the grid and takes ~12 % of 150 MeV.
    assert 0.1 < res.fraction("off_grid") < 0.14 and res.fraction("grid") > 0.8 and res.ledger["lost"] > 0.0


def test_same_seed_is_bit_identical(canvas) -> None:
    case = mv.Case("seed", "pmma", 3.0, ((0.0, 0.0, 100.0, 1.0),), lateral=40)
    a, _, _ = mv.run_gpu(canvas, case, 50_000, seed=7)
    b, _, _ = mv.run_gpu(canvas, case, 50_000, seed=7)
    c, _, _ = mv.run_gpu(canvas, case, 50_000, seed=8)
    assert a.max() > 0.0 and np.array_equal(a, b) and not np.array_equal(a, c)


def _golden_params():
    for path in GOLDENS:
        marks = () if path.stem in QUICK else (pytest.mark.slow,)
        yield pytest.param(path, id=path.stem, marks=marks)


def _assert_close(ref: dict, gpu: dict, tol: dict) -> None:
    diff = mv.compare(ref, gpu)
    assert diff["idd"] <= tol["idd"], diff
    assert abs(diff["r80"]) <= tol["r80"], diff
    assert not diff["sigma"] > tol["sigma"], diff
    assert abs(diff["energy"]) <= tol["energy"], diff


GOLDEN_TOL = {"idd": 0.015, "r80": 0.3, "sigma": 0.03, "energy": 0.01}


@pytest.mark.skipif(not GOLDENS, reason="no MCsquare goldens; run scripts/mcsquare_validate.py --write-goldens")
@pytest.mark.parametrize("path", list(_golden_params()))
def test_gpu_matches_mcsquare_golden(canvas, path) -> None:
    with np.load(path) as g:
        case = mv.Case(
            path.stem, str(g["medium"]), float(g["sigma"]), tuple(map(tuple, g["spots"])),
            int(g["wet"]), int(g["lateral"]),
        )
        assert case.depth == int(g["depth"]) and float(g["spread_pct"]) == mv.SPREAD_PCT
        ref = {k: g[k] for k in ("idd", "r80", "r90", "sigma_depths", "sigmas", "energy")}
    gpu_vol, _tex, _res = mv.run_gpu(canvas, case, HISTORIES)
    _assert_close(ref, mv.summary(gpu_vol, case), GOLDEN_TOL)


@pytest.mark.skipif(not os.environ.get("MCSQUARE_DIR"), reason="MCSQUARE_DIR not set")
def test_gpu_matches_live_mcsquare(canvas, tmp_path) -> None:
    case = mv.Case("live", "water", 4.0, ((0.0, 0.0, 120.0, 1.0),), lateral=80)
    ref = mv.run_mcsquare(case, mv.mcsquare_exe(), HISTORIES, tmp_path)
    gpu, _tex, _res = mv.run_gpu(canvas, case, HISTORIES)
    _assert_close(mv.summary(ref, case), mv.summary(gpu, case), GOLDEN_TOL)


def test_plan_vs_plan_monte_carlo_shares_random_numbers(canvas) -> None:
    from scan_kit.views.dose_volume_data import SplatBatch, range_axis_for_medium
    from scan_kit.views.dose_volume_raycast import read_texture
    from scan_kit.views.dose_volume_vispy import DoseScene

    rng = np.random.default_rng(3)
    n = 20
    plan = SplatBatch(
        center=np.column_stack([rng.uniform(-15, 15, (n, 2)), np.zeros(n)]).astype(np.float32),
        sigma=np.full((n, 3), 3.5, dtype=np.float32),
        weight=rng.uniform(0.5, 2.0, n).astype(np.float32),
        energy_mev=rng.choice([90.0, 120.0], n).astype(np.float32),
        dose_mu=rng.uniform(0.5, 2.0, n).astype(np.float32),
        k_mu=2.6e-8,
    )
    scene = DoseScene(canvas)
    scene.render(
        plan, plan, range_axis_for_medium("water"), gain=1.0, gantry_deg=0.0, weight_mode=WEIGHT_DOSE,
        voxel_mm=2.0, gamma=True, model=MODEL_MC, mc_histories=100_000,
    )
    shape = tuple(scene._meas_tex.shape[:3])
    meas = read_texture(canvas, scene._meas_tex, shape)
    assert meas.max() > 0.0 and np.array_equal(meas, read_texture(canvas, scene._plan_tex, shape))
    passed, total = scene.gamma_pass
    assert total > 0 and passed == total
    assert "MC 100k" in scene.volume_note


def test_model_controls_follow_medium_and_weight(qapp, tmp_path) -> None:
    from scan_kit.views.dose_volume_catalog import WEIGHT_MU
    from scan_kit.views.dose_volume_window import DoseVolumeWindow

    window = DoseVolumeWindow([], str(tmp_path))
    try:
        assert not window._model_row.isHidden() and window._histories_row.isHidden()
        window._model_combo.set_current(MODEL_MC)
        window._sync_model_controls()
        assert not window._histories_row.isHidden() and not window._scatter_check.isEnabled()
        config = window._read_config()
        assert config.dose_model == MODEL_MC and config.mc_histories == 1_000_000
        window._set_combo(window._medium_combo, MEDIUM_POLYETHYLENE)
        assert not window._model_combo.isEnabled() and window._choice(window._model_combo) == MODEL_ANALYTIC
        assert window._histories_row.isHidden() and window._scatter_check.isEnabled()
        window._set_combo(window._weight_combo, WEIGHT_MU)
        window._sync_model_controls()
        assert window._model_row.isHidden()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
