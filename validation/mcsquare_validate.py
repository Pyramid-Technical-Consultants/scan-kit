"""Check the GPU Monte Carlo against MCsquare, the reference engine.

Not part of ``pytest``: run it after changing Monte Carlo physics, not after UI work.
Checks only run the GPU; MCsquare's dose is cached in ``validation/goldens/``.
Each golden holds the depth dose, R80, lateral sigma, energy and centroid over the
whole grid, plus the dose around the beam for a 3D gamma. Run from the repo root::

    python validation/mcsquare_validate.py fast                  # tricky small cases, ~30 s
    python validation/mcsquare_validate.py full                  # every case, ~10 min
    python validation/mcsquare_validate.py full --case water_150_s3
    python validation/mcsquare_validate.py fast --write-goldens  # rerun MCsquare, replace the cache

``--write-goldens`` needs ``MCSQUARE_DIR`` pointing at an MCsquare build (the
folder or the executable). Exits nonzero if any case misses its suite's tolerance.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "third_party" / "MCsquare" / "Materials"
GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
GOLDEN_PRIMARIES = 1e7
EV_PER_G_TO_GY = 1.602176e-16
MEV_TO_J = 1.602176634e-13
SPREAD_PCT = 0.7
SEED = 1
LATERAL_MM = 100
SIGMA_DEPTHS = (0.25, 0.5, 0.9)  # fractions of R80
# Gamma only scores reference dose >= 10 % of the maximum, searching up to 2 mm.
CROP_FRAC, CROP_MARGIN = 0.05, 3
# fast runs 10x fewer GPU histories than the goldens, so its noise allows less.
SUITES = {
    "fast": dict(histories=1e6, idd=0.02, r80=0.25, sigma=0.03, energy=0.01, centroid=0.2, gamma_pct=2.0, gamma=0.98),
    "full": dict(histories=1e7, idd=0.01, r80=0.2, sigma=0.02, energy=0.005, centroid=0.1, gamma_pct=1.0, gamma=0.99),
}


@dataclass(frozen=True)
class Case:
    name: str
    medium: str
    sigma: float  # mm, both axes, at the nozzle
    spots: tuple  # ((x mm, y mm, energy MeV, protons), ...)
    wet: int = 0  # mm of water in front of the phantom
    lateral: int = LATERAL_MM
    scale: float = 1.0  # x histories; a field spreads them over far more voxels than a pencil
    fast: bool = False  # also in the fast suite

    @property
    def pencil(self) -> bool:
        return len(self.spots) == 1

    @property
    def depth(self) -> int:
        e_max = max(s[2] for s in self.spots)
        return int(math.ceil(1.1 * csda_range_mm(self.medium, e_max) + 10.0))


def _pencil(medium: str, energy: float, sigma: float, wet: int = 0, **kw) -> Case:
    name = kw.pop("name", None) or f"{medium}_{energy:g}_s{sigma:g}" + (f"_wet{wet}" if wet else "")
    return Case(name, medium, sigma, ((0.0, 0.0, energy, 1.0),), wet, **kw)


def _field() -> Case:
    grid = np.arange(-15.0, 15.1, 5.0)
    spots = tuple(
        (float(x), float(y), energy, 1.0 + i)
        for i, energy in enumerate((130.0, 135.0, 140.0, 145.0, 150.0))
        for y in grid
        for x in grid
    )
    return Case("water_field_5x7x7", "water", 4.0, spots, lateral=120, scale=4.0)


CASES = (
    # fast: small grids, each aimed at a place a transport bug would show first.
    _pencil("water", 100.0, 1.0, lateral=40, fast=True, name="water_100_s1_narrow"),  # scatter-dominated width
    _pencil("copper", 70.0, 3.0, lateral=30, fast=True),  # high Z, 6 mm range, ends below the ICRU 7 MeV cut
    _pencil("aluminum", 120.0, 3.0, wet=40, lateral=40, fast=True),  # water -> aluminum interface
    _pencil("water", 180.0, 3.0, lateral=40, fast=True),  # nuclear build-up and secondary protons
    Case(  # off-axis spots, mixed energies and weights: placement, mirroring, weighting
        "pmma_offaxis_3spot", "pmma", 3.0,
        ((12.0, -7.0, 100.0, 1.0), (-8.0, 9.0, 120.0, 2.0), (0.0, 0.0, 110.0, 0.5)), lateral=60, fast=True,
    ),
    # full only
    *(_pencil("water", e, s) for e in (70.0, 100.0, 150.0, 200.0, 230.0) for s in (3.0, 6.0)),
    _pencil("pmma", 150.0, 4.0),
    _pencil("polystyrene", 150.0, 4.0),
    _pencil("aluminum", 100.0, 4.0),
    _pencil("copper", 100.0, 4.0),
    _pencil("water", 150.0, 4.0, wet=30),
    _field(),
)


def _tables() -> dict:
    from scan_kit.views.dose_mc import load_tables

    return load_tables()


def medium_rho(medium: str) -> float:
    t = _tables()
    return float(t["m_props"][list(t["media"]).index(medium), 0])


def csda_range_mm(medium: str, energy: float) -> float:
    """CSDA range from MCsquare's stopping powers (0.5 MeV bins from 0)."""
    t = _tables()
    m = list(t["media"]).index(medium)
    stop = t["m_stop"][m] * float(t["m_props"][m, 0]) / 1e6  # MeV/cm
    e = np.arange(stop.size) * 0.5
    fine = np.linspace(0.5, energy, 2000)
    return float(np.trapezoid(1.0 / np.interp(fine, e, stop), fine) * 10.0)


# ---- metrics on (nz, ny, nx) Gy-per-proton volumes, z index 0 deepest, 1 mm voxels

def depth_dose(vol: np.ndarray) -> np.ndarray:
    """Laterally integrated dose, surface first."""
    return np.asarray(vol, dtype=float)[::-1].sum(axis=(1, 2))


def distal_depth(idd: np.ndarray, frac: float) -> float:
    """Depth (mm) where *idd* falls through *frac* of its maximum beyond the peak."""
    peak = int(np.argmax(idd))
    level = frac * float(idd[peak])
    for i in range(peak, idd.size - 1):
        if idd[i + 1] < level <= idd[i]:
            return i + 0.5 + (idd[i] - level) / (idd[i] - idd[i + 1])
    return float("nan")


def fwhm_sigma(profile: np.ndarray) -> float:
    """Gaussian sigma (voxels) from the full width at half maximum."""
    p = np.asarray(profile, dtype=float)
    peak = int(np.argmax(p))
    half = 0.5 * float(p[peak])
    lo = peak
    while lo > 0 and p[lo - 1] >= half:
        lo -= 1
    hi = peak
    while hi < p.size - 1 and p[hi + 1] >= half:
        hi += 1
    if lo == 0 or hi == p.size - 1:
        return float("nan")
    left = lo - (p[lo] - half) / (p[lo] - p[lo - 1])
    right = hi + (p[hi] - half) / (p[hi] - p[hi + 1])
    return (right - left) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def lateral_sigmas(vol: np.ndarray, depths) -> np.ndarray:
    surface_first = np.asarray(vol, dtype=float)[::-1]
    return np.array([fwhm_sigma(surface_first[int(d)].sum(axis=0)) for d in depths])


def centroid_mm(vol: np.ndarray) -> np.ndarray:
    """Dose-weighted (x, y) in plan mm."""
    v = np.asarray(vol, dtype=float)
    ny, nx = v.shape[1:]
    x = (v.sum(axis=(0, 1)) * (np.arange(nx) + 0.5 - nx / 2)).sum()
    y = (v.sum(axis=(0, 2)) * (np.arange(ny) + 0.5 - ny / 2)).sum()
    return np.array([x, y]) / v.sum()


def energy_mev(vol: np.ndarray, medium: str) -> float:
    """Energy deposited in the grid per proton (1 mm voxels)."""
    return float(np.asarray(vol, dtype=float).sum()) * medium_rho(medium) * 1e-6 / MEV_TO_J


def summary(vol: np.ndarray, case: Case) -> dict:
    idd = depth_dose(vol)
    r80 = distal_depth(idd, 0.8)
    depths = np.array([f * r80 for f in SIGMA_DEPTHS])
    return {
        "idd": idd,
        "r80": r80,
        "r90": distal_depth(idd, 0.9),
        "sigma_depths": depths,
        "sigmas": lateral_sigmas(vol, depths) if case.pencil else np.full(depths.size, np.nan),
        "energy": energy_mev(vol, case.medium),
        "centroid": centroid_mm(vol),
    }


def compare(ref: dict, gpu: dict) -> dict:
    """Relative / absolute differences of *gpu* from *ref* summaries."""
    upto = int(ref["r90"])
    n = min(upto, gpu["idd"].size, ref["idd"].size)
    idd = float(np.max(np.abs(gpu["idd"][:n] / ref["idd"][:n] - 1.0)))
    sig = gpu["sigmas"] / ref["sigmas"] - 1.0
    return {
        "idd": idd,
        "r80": float(gpu["r80"] - ref["r80"]),
        "sigma": float(np.max(np.abs(sig))) if np.all(np.isfinite(sig)) else float("nan"),
        "energy": float(gpu["energy"] / ref["energy"] - 1.0),
        "centroid": float(np.max(np.abs(gpu["centroid"] - ref["centroid"]))),
    }


# ---- MCsquare

def mcsquare_exe() -> Path:
    raw = os.environ.get("MCSQUARE_DIR", "")
    if not raw:
        raise SystemExit("set MCSQUARE_DIR to an MCsquare build folder or executable")
    path = Path(raw)
    if path.is_file():
        return path
    names = {"win32": "MCsquare_win.exe", "darwin": "MCsquare_mac"}.get(sys.platform, "MCsquare_linux")
    for name in (names, "MCsquare.exe", "MCsquare"):
        if (path / name).is_file():
            return path / name
    raise SystemExit(f"no MCsquare executable in {path}")


def _env() -> dict:
    return {**os.environ, "MCsquare_Materials_Dir": str(MATERIALS)}


def mcsquare_version(exe: Path) -> str:
    out = subprocess.run([str(exe), "-v"], capture_output=True, text=True, env=_env(), check=False).stdout
    return " | ".join(line.strip() for line in out.splitlines() if line.strip())


def _material_label(name: str) -> int:
    for line in (MATERIALS / "list.dat").read_text().splitlines():
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and parts[1] == name:
            return int(parts[0])
    raise KeyError(name)


def write_inputs(case: Case, work: Path, primaries: int) -> None:
    t = _tables()
    names = dict(zip((str(m) for m in t["media"]), (str(n) for n in t["mcsquare_names"])))
    lat, depth, wet = case.lateral, case.depth, case.wet
    total = depth + wet
    # MHD axes: 0 = plan X, 1 = beam (enters at the top face, travels -1), 2 = plan Y.
    hu = np.zeros((lat, total, lat), dtype=np.float32)
    hu[:, depth:, :] = -1000.0  # water in front of the phantom
    hu.tofile(work / "CT.raw")
    (work / "CT.mhd").write_text(
        "ObjectType = Image\nNDims = 3\n"
        f"DimSize = {lat} {total} {lat}\nElementSpacing = 1 1 1\nOffset = 0 0 0\n"
        "ElementType = MET_FLOAT\nElementByteOrderMSB = False\nElementDataFile = CT.raw\n"
    )
    rho_water, rho = medium_rho("water"), medium_rho(case.medium)
    (work / "HU_Density.txt").write_text(f"-1000 {rho_water}\n0 {rho}\n")
    (work / "HU_Material.txt").write_text(
        f"-1000 {_material_label(names['water'])}\n-1 {_material_label(names[case.medium])}\n"
    )
    # MCsquare's 2x2 eigen-solve divides 0/0 at zero divergence or correlation;
    # 1e-5 rad at 0.1 moves sigma by < 0.01 % over 40 cm.
    s, d, r = case.sigma, 1e-5, 0.1
    beam = f"{s} {d} {r} {s} {d} {r}"
    row = f"{SPREAD_PCT} 1.0 1.0 {beam} 0.0 {beam}"
    (work / "BDL.txt").write_text(
        "--UPenn beam model (double gaussian)--\n# scan-kit validation: parallel mono-Gaussian beam\n\n"
        "Nozzle exit to Isocenter distance\n0.01\n\n"
        "SMX to Isocenter distance\n10000000.0\n\nSMY to Isocenter distance\n10000000.0\n\n"
        "Beam parameters\n2 energies\n\n"
        "NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x "
        "SpotSize1y Divergence1y Correlation1y Weight2 SpotSize2x Divergence2x Correlation2x "
        "SpotSize2y Divergence2y Correlation2y\n"
        f"1.0 1.0 {row}\n400.0 400.0 {row}\n"
    )
    layers: dict[float, list] = {}
    for x, y, e, w in case.spots:
        layers.setdefault(float(e), []).append((x, y, w))
    weight = sum(s[3] for s in case.spots)
    lines = [
        "#TREATMENT-PLAN-DESCRIPTION", "#PlanName", case.name, "#NumberOfFractions", "1",
        "##FractionID", "1", "##NumberOfFields", "1", "###FieldsID", "1",
        "#TotalMetersetWeightOfAllFields", f"{weight}", "", "#FIELD-DESCRIPTION", "###FieldID", "1",
        "###FinalCumulativeMeterSetWeight", f"{weight}", "###GantryAngle", "0", "###PatientSupportAngle", "0",
        # Isocenter on the entry face; the 0.01 mm nozzle gap keeps air loss negligible.
        "###IsocenterPosition", f"{lat / 2} {total} {lat / 2}",
        "###NumberOfControlPoints", f"{len(layers)}", "", "#SPOTS-DESCRIPTION",
    ]
    cum = 0.0
    for k, (e, spots) in enumerate(sorted(layers.items()), start=1):
        cum += sum(w for _, _, w in spots)
        lines += ["####ControlPointIndex", f"{k}", "####SpotTunnedID", "1", "####CumulativeMetersetWeight",
                  f"{cum}", "####Energy (MeV)", f"{e}", "####NbOfScannedSpots", f"{len(spots)}", "####X Y Weight"]
        lines += [f"{x} {y} {w}" for x, y, w in spots]
    (work / "Plan.txt").write_text("\n".join(lines) + "\n")
    (work / "config.txt").write_text(
        f"Num_Threads 0\nRNG_Seed 0\nNum_Primaries {primaries}\n"
        "CT_File CT.mhd\nHU_Density_Conversion_File HU_Density.txt\n"
        "HU_Material_Conversion_File HU_Material.txt\n"
        "BDL_Machine_Parameter_File BDL.txt\nBDL_Plan_File Plan.txt\n"
        "Output_Directory Outputs\nDose_MHD_Output True\nCompute_stat_uncertainty True\n"
    )
    (work / "Outputs").mkdir(exist_ok=True)


def read_mhd(path: Path) -> np.ndarray:
    head = dict(
        (k.strip(), v.strip()) for k, _, v in (line.partition("=") for line in path.read_text().splitlines()) if v
    )
    dims = [int(v) for v in head["DimSize"].split()]
    dtype = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8"}[head["ElementType"]]
    data = np.fromfile(path.parent / head["ElementDataFile"], dtype=dtype)
    return data.reshape(dims[::-1])


def run_mcsquare(case: Case, exe: Path, primaries: int, work: Path) -> np.ndarray:
    """MCsquare dose on our grid: (nz, ny, nx) Gy per proton, z index 0 deepest."""
    write_inputs(case, work, primaries)
    done = subprocess.run([str(exe), "config.txt"], cwd=work, env=_env(), capture_output=True, text=True)
    if done.returncode != 0 or not (work / "Outputs" / "Dose.mhd").is_file():
        raise RuntimeError(f"MCsquare failed on {case.name}:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    dose = read_mhd(work / "Outputs" / "Dose.mhd") * EV_PER_G_TO_GY  # [plan Y, beam, -plan X]
    # MCsquare's transport x runs against the CT's (get_CT_Offset), so plan X is mirrored.
    vol = np.ascontiguousarray(dose.transpose(1, 0, 2)[: case.depth, :, ::-1], dtype=np.float32)
    # The beam must land where the plan puts it; each spot deposits roughly protons x energy.
    expect = np.average([s[:2] for s in case.spots], axis=0, weights=[s[3] * s[2] for s in case.spots])
    got = centroid_mm(vol)
    if not np.all(np.abs(got - expect) <= 1.0):
        raise RuntimeError(f"{case.name}: MCsquare beam centroid {got} mm, expected {expect}")
    return vol


# ---- goldens: MCsquare summaries plus the dose around the beam

def golden_path(case: Case) -> Path:
    return GOLDEN_DIR / f"{case.name}.npz"


def _params(case: Case) -> dict:
    return dict(
        medium=case.medium, sigma=case.sigma, spots=np.array(case.spots, dtype=float), wet=case.wet,
        lateral=case.lateral, depth=case.depth, spread_pct=SPREAD_PCT,
    )


def write_golden(case: Case, vol: np.ndarray, primaries: int, version: str) -> Path:
    idx = np.argwhere(vol >= CROP_FRAC * vol.max())
    lo = np.maximum(idx.min(axis=0) - CROP_MARGIN, 0)
    hi = np.minimum(idx.max(axis=0) + CROP_MARGIN + 1, vol.shape)
    core = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    scale = float(core.max())
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(case)
    np.savez_compressed(
        path, **_params(case), primaries=primaries, version=version,
        crop=np.stack([lo, hi]), dose_scale=scale, dose=(core / scale).astype(np.float16),
        **{k: np.asarray(v) for k, v in summary(vol, case).items()},
    )
    return path


def load_golden(case: Case) -> dict:
    """The cached MCsquare result; raises if it was made for a different case setup."""
    with np.load(golden_path(case)) as g:
        gold = {k: g[k] for k in g.files}
    for k, v in _params(case).items():
        if not np.array_equal(np.asarray(gold[k]), np.asarray(v)):
            raise ValueError(f"{case.name}: golden {k} is {gold[k]}, case has {v}; rerun --write-goldens")
    gold["dose"] = gold["dose"].astype(np.float32) * np.float32(gold["dose_scale"])
    return gold


# ---- GPU

def gpu_grid(case: Case):
    from scan_kit.views.dose_volume_fill import DoseGrid

    lat, depth = case.lateral, case.depth
    return DoseGrid(np.array([-lat / 2, -lat / 2, -float(depth)]), (lat, lat, depth), False, 1.0)


def run_gpu(canvas, case: Case, histories: int, seed: int = SEED):
    """GPU dose (nz, ny, nx) Gy per proton and the McResult."""
    from scan_kit.views.dose_mc import mc_fill_texture
    from scan_kit.views.dose_volume_raycast import _alloc_texture, read_texture

    grid = gpu_grid(case)
    shape = tuple(grid.shape[::-1])
    tex = _alloc_texture(shape)
    x, y, e, w = (np.array(c, dtype=float) for c in zip(*case.spots))
    res = mc_fill_texture(
        canvas, tex, x, y, np.full_like(x, case.sigma), np.full_like(x, case.sigma), e, w / w.sum(), case.medium,
        grid, depth=float(case.depth), wet=float(case.wet), spread_pct=SPREAD_PCT, histories=histories, seed=seed,
    )
    return read_texture(canvas, tex, shape), res


def gamma_rate(canvas, ref: np.ndarray, evl: np.ndarray, dose_pct: float) -> float:
    """Global gamma pass rate, dose_pct / 1 mm, 10 % cutoff, of *evl* against *ref*."""
    from scan_kit.views.dose_volume_fill import DoseGrid
    from scan_kit.views.dose_volume_physics import GammaCriteria
    from scan_kit.views.dose_volume_raycast import _alloc_texture, gamma_texture

    ref_tex, evl_tex, out_tex = (_alloc_texture(ref.shape) for _ in range(3))
    ref_tex.set_data(np.ascontiguousarray(ref, dtype=np.float32))
    evl_tex.set_data(np.ascontiguousarray(evl, dtype=np.float32))
    grid = DoseGrid(np.zeros(3), ref.shape[::-1], False, 1.0)
    passed, evaluated = gamma_texture(canvas, ref_tex, evl_tex, out_tex, grid, GammaCriteria(dose_pct, 1.0, 10.0))
    return passed / evaluated if evaluated else float("nan")


def make_canvas():
    from PySide6.QtWidgets import QApplication

    from scan_kit.views.vispy_plot import make_scene_canvas

    app = QApplication.instance() or QApplication(sys.argv)
    canvas = make_scene_canvas(size=(64, 64), show=False, gl="gl+")
    canvas.set_current()
    return app, canvas


def passes(diff: dict, gamma: float, tol: dict) -> bool:
    ok = all(abs(diff[k]) <= tol[k] for k in ("idd", "r80", "energy", "centroid"))
    ok &= not (diff["sigma"] > tol["sigma"])  # nan (fields) is not a failure
    return bool(ok and gamma >= tol["gamma"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite", choices=SUITES)
    ap.add_argument("--case", action="append", default=[], help="only these cases (repeatable)")
    ap.add_argument("--histories", type=float, help="GPU histories per case (default: the suite's)")
    ap.add_argument("--write-goldens", action="store_true", help="rerun MCsquare and replace the cached goldens")
    ap.add_argument("--golden-primaries", type=float, default=GOLDEN_PRIMARIES)
    args = ap.parse_args(argv)
    tol = SUITES[args.suite]
    cases = [c for c in CASES if c.fast or args.suite == "full"]
    unknown = set(args.case) - {c.name for c in cases}
    if unknown:
        ap.error(f"unknown {args.suite} cases {sorted(unknown)}; choose from {[c.name for c in cases]}")
    cases = [c for c in cases if not args.case or c.name in args.case]
    histories = int(args.histories or tol["histories"])

    if args.write_goldens:
        exe = mcsquare_exe()
        version = mcsquare_version(exe)
        print(f"Writing goldens with MCsquare: {version}")
        for case in cases:
            n = int(args.golden_primaries * case.scale)
            t0 = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="mcsq_") as tmp:
                write_golden(case, run_mcsquare(case, exe, n, Path(tmp)), n, version)
            print(f"  {case.name:24} {n:.0e} primaries  {time.perf_counter() - t0:5.0f} s", flush=True)
        print()

    _app, canvas = make_canvas()
    print(f"{args.suite} suite: GPU {histories:.0e} histories per case against cached MCsquare\n")
    print(f"{'case':24} {'IDD':>7} {'dR80':>7} {'sigma':>7} {'energy':>7} {'xy':>7} {'gamma':>7}  GPU s")
    failed = []
    for case in cases:
        if not golden_path(case).is_file():
            print(f"{case.name:24} no golden; run with --write-goldens")
            failed.append(case.name)
            continue
        gold = load_golden(case)
        t0 = time.perf_counter()
        vol, _res = run_gpu(canvas, case, int(histories * case.scale))
        seconds = time.perf_counter() - t0
        diff = compare(gold, summary(vol, case))
        lo, hi = gold["crop"]
        gamma = gamma_rate(canvas, gold["dose"], vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], tol["gamma_pct"])
        ok = passes(diff, gamma, tol)
        if not ok:
            failed.append(case.name)
        print(
            f"{case.name:24} {100 * diff['idd']:6.2f}% {diff['r80']:+6.2f}mm {100 * diff['sigma']:6.2f}% "
            f"{100 * diff['energy']:+6.2f}% {diff['centroid']:5.2f}mm {100 * gamma:6.2f}%  {seconds:4.0f}  "
            f"{'ok' if ok else 'FAIL'}",
            flush=True,
        )
    print(
        f"\nTolerances: IDD {100 * tol['idd']:g}% to R90, |dR80| {tol['r80']} mm, sigma {100 * tol['sigma']:g}% "
        f"at {SIGMA_DEPTHS} R80, energy {100 * tol['energy']:g}%, centroid {tol['centroid']} mm, "
        f"gamma {tol['gamma_pct']:g}%/1mm (10% cutoff) >= {100 * tol['gamma']:g}%"
    )
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All cases pass.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
