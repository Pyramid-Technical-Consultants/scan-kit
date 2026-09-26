"""Validate the GPU Monte Carlo against MCsquare, the reference engine.

Each case is run through an MCsquare binary (uniform phantom CT, parallel
mono-Gaussian beam model, one field at gantry 0 with the nozzle on the CT
surface) and through ``mc_fill_texture`` on the same 1 mm grid, then compared on
depth dose, R80, lateral sigma, deposited energy and 3D gamma. Run from the
repo root with ``MCSQUARE_DIR`` pointing at an MCsquare build (the folder or
the executable)::

    python scripts/mcsquare_validate.py                  # all cases, 1e7 primaries
    python scripts/mcsquare_validate.py water_150_s3     # just some cases
    python scripts/mcsquare_validate.py --write-goldens  # refresh tests/data/mcsquare

Exits nonzero if any case misses a tolerance.
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
GOLDEN_DIR = ROOT / "tests" / "data" / "mcsquare"
EV_PER_G_TO_GY = 1.602176e-16
MEV_TO_J = 1.602176634e-13
SPREAD_PCT = 0.7
SEED = 1
LATERAL_MM = 100
SIGMA_DEPTHS = (0.25, 0.5, 0.9)  # fractions of R80
TOL = {"idd": 0.01, "r80": 0.2, "sigma": 0.02, "energy": 0.005, "gamma": 0.99}


@dataclass(frozen=True)
class Case:
    name: str
    medium: str
    sigma: float  # mm, both axes, at the nozzle
    spots: tuple  # ((x mm, y mm, energy MeV, protons), ...)
    wet: int = 0  # mm of water in front of the phantom
    lateral: int = LATERAL_MM
    scale: float = 1.0  # x --primaries; a field spreads them over far more voxels than a pencil

    @property
    def pencil(self) -> bool:
        return len(self.spots) == 1

    @property
    def depth(self) -> int:
        e_max = max(s[2] for s in self.spots)
        return int(math.ceil(1.1 * csda_range_mm(self.medium, e_max) + 10.0))


def _pencil(medium: str, energy: float, sigma: float, wet: int = 0) -> Case:
    name = f"{medium}_{energy:g}_s{sigma:g}" + (f"_wet{wet}" if wet else "")
    return Case(name, medium, sigma, ((0.0, 0.0, energy, 1.0),), wet)


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
    # The beam axis must land where the plan puts it: centred laterally.
    c = np.arange(case.lateral) + 0.5 - case.lateral / 2
    for axis, name in ((2, "x"), (1, "y")):
        prof = vol.sum(axis=tuple(a for a in range(3) if a != axis))
        off = float((prof * c).sum() / prof.sum()) if prof.sum() > 0 else float("nan")
        expect = float(np.average([s[0 if name == "x" else 1] for s in case.spots], weights=[s[3] for s in case.spots]))
        if not abs(off - expect) <= 1.0:
            raise RuntimeError(f"{case.name}: MCsquare beam centroid {name}={off:.2f} mm, expected {expect:.2f}")
    return vol


# ---- GPU

def gpu_grid(case: Case):
    from scan_kit.views.dose_volume_fill import DoseGrid

    lat, depth = case.lateral, case.depth
    return DoseGrid(np.array([-lat / 2, -lat / 2, -float(depth)]), (lat, lat, depth), False, 1.0)


def run_gpu(canvas, case: Case, histories: int, seed: int = SEED):
    """GPU dose (nz, ny, nx) Gy per proton, the texture it lives in, and the McResult."""
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
    return read_texture(canvas, tex, shape), tex, res


def gamma_rate(canvas, case: Case, ref: np.ndarray, gpu_tex) -> float:
    from scan_kit.views.dose_volume_physics import GammaCriteria
    from scan_kit.views.dose_volume_raycast import _alloc_texture, gamma_texture

    grid = gpu_grid(case)
    ref_tex = _alloc_texture(ref.shape)
    ref_tex.set_data(ref)
    out_tex = _alloc_texture(ref.shape)
    passed, evaluated = gamma_texture(canvas, ref_tex, gpu_tex, out_tex, grid, GammaCriteria(1.0, 1.0, 10.0))
    return passed / evaluated if evaluated else float("nan")


def make_canvas():
    from PySide6.QtWidgets import QApplication

    from scan_kit.views.vispy_plot import make_scene_canvas

    app = QApplication.instance() or QApplication(sys.argv)
    canvas = make_scene_canvas(size=(64, 64), show=False, gl="gl+")
    canvas.set_current()
    return app, canvas


def write_golden(case: Case, ref: dict, primaries: int, version: str) -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{case.name}.npz"
    np.savez_compressed(
        path, medium=case.medium, sigma=case.sigma, spots=np.array(case.spots, dtype=float), wet=case.wet,
        lateral=case.lateral, depth=case.depth, spread_pct=SPREAD_PCT, primaries=primaries, version=version,
        **{k: np.asarray(v) for k, v in ref.items()},
    )
    return path


def passes(diff: dict, gamma: float) -> bool:
    ok = diff["idd"] <= TOL["idd"] and abs(diff["r80"]) <= TOL["r80"] and abs(diff["energy"]) <= TOL["energy"]
    ok &= not (diff["sigma"] > TOL["sigma"])  # nan (fields) is not a failure
    return bool(ok and gamma >= TOL["gamma"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--primaries", type=float, default=1e7)
    ap.add_argument("--write-goldens", action="store_true")
    args = ap.parse_args(argv)
    names = {c.name for c in CASES}
    unknown = set(args.cases) - names
    if unknown:
        ap.error(f"unknown cases {sorted(unknown)}; choose from {sorted(names)}")
    cases = [c for c in CASES if not args.cases or c.name in args.cases]
    primaries = int(args.primaries)
    exe = mcsquare_exe()
    version = mcsquare_version(exe)
    print(f"MCsquare: {version}\n{primaries:.0e} primaries per case\n")
    _app, canvas = make_canvas()
    print(f"{'case':24} {'IDD':>7} {'dR80':>7} {'sigma':>7} {'energy':>7} {'gamma':>7}  MCsq/GPU s")
    failed = []
    for case in cases:
        n = int(primaries * case.scale)
        with tempfile.TemporaryDirectory(prefix="mcsq_") as tmp:
            t0 = time.perf_counter()
            ref_vol = run_mcsquare(case, exe, n, Path(tmp))
            t1 = time.perf_counter()
        gpu_vol, gpu_tex, _res = run_gpu(canvas, case, n)
        t2 = time.perf_counter()
        ref, gpu = summary(ref_vol, case), summary(gpu_vol, case)
        diff = compare(ref, gpu)
        gamma = gamma_rate(canvas, case, ref_vol, gpu_tex)
        ok = passes(diff, gamma)
        if not ok:
            failed.append(case.name)
        print(
            f"{case.name:24} {100 * diff['idd']:6.2f}% {diff['r80']:+6.2f}mm {100 * diff['sigma']:6.2f}% "
            f"{100 * diff['energy']:+6.2f}% {100 * gamma:6.2f}%  {t1 - t0:5.0f}/{t2 - t1:<4.0f} "
            f"{'ok' if ok else 'FAIL'}",
            flush=True,
        )
        if args.write_goldens:
            write_golden(case, ref, n, version)
    tol = TOL
    print(
        f"\nTolerances: IDD {100 * tol['idd']:g}% to R90, |dR80| {tol['r80']} mm, sigma {100 * tol['sigma']:g}% "
        f"at {SIGMA_DEPTHS} R80, energy {100 * tol['energy']:g}%, gamma 1%/1mm (10% cutoff) >= {100 * tol['gamma']:g}%"
    )
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All cases pass.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
