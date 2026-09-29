"""Pack MCsquare material data into ``scan_kit/assets/mc_materials.npz`` and copy its
scanner calibrations and beam models to ``scan_kit/assets/mcsquare``.

Reproduces MCsquare's own preprocessing (``data_materials.c``, ``data_nuclear.c``,
``data_Stop_Pow.c``) for the media the GPU Monte Carlo supports, so the shader
reads exactly the numbers MCsquare would. The media are the phantom slabs plus every
material an MCsquare scanner calibration or beam-model range shifter refers to.
Run from the repo root after ``git submodule update --init``::

    python scripts/build_mc_tables.py
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MCSQUARE = ROOT / "third_party" / "MCsquare"
MATERIALS = MCSQUARE / "Materials"
OUT = ROOT / "scan_kit" / "assets" / "mc_materials.npz"

# scan-kit slab medium key -> MCsquare material name; these keep the first rows.
SLAB_MEDIA = {
    "water": "Water",
    "pmma": "PMMA",
    "polystyrene": "Polystyrene",
    "aluminum": "aluminium",
    "copper": "copper",
}
N_AVO = 6.0221415e23
INTERP_BINS = 250  # ceil(250 / INTERP_BIN), INTERP_BIN = 1 MeV
PSTAR_BIN = 0.5
ICRU_ANGLES = (0, 10, 20, 30, 40, 50, 60, 70, 90, 110, 130, 150, 180)
SPECIES = ("proton", "deuteron", "alpha")
TYPE_MIXTURE, TYPE_ICRU, TYPE_PP = 0, 1, 2


def solid_angle(theta_deg: float) -> float:
    return 2 * math.pi * (1 - math.cos(math.radians(theta_deg)))


def binary_search(value: float, values) -> int:
    """MCsquare ``Binary_Search``: largest i with values[i] < value, or -1."""
    if len(values) <= 1:
        return -1
    i, j = -1, len(values)
    while j - i > 1:
        k = (i + j) // 2
        if values[k] < value:
            i = k
        else:
            j = k
    return i


def lerp(x, x1, x2, y1, y2):
    return (y2 - y1) / (x2 - x1) * (x - x1) + y1


def tokens(path: Path):
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("#"):
            continue
        parts = line.split("#")[0].split()
        if parts:
            yield parts


def material_ids() -> dict[str, int]:
    """MCsquare material name -> label, from ``Materials/list.dat``."""
    return {t[1]: int(t[0]) for t in tokens(MATERIALS / "list.dat") if len(t) >= 2 and t[0].isdigit()}


def media() -> dict[str, str]:
    """scan-kit key -> MCsquare name: the slabs, then scanner and range-shifter materials by label."""
    by_id = {v: k for k, v in material_ids().items()}
    used = {int(t[1]) for f in sorted(MCSQUARE.glob("Scanners/*/HU_Material_Conversion.txt")) for t in tokens(f)}
    for f in sorted(MCSQUARE.glob("BDL/*.txt")):
        for line in f.read_text(encoding="latin-1").splitlines():
            key, _, value = line.split("#")[0].partition("=")
            if key.strip() == "RS_material":
                used.add(int(value))
    out = dict(SLAB_MEDIA)
    for label in sorted(used):
        if by_id[label] not in out.values():
            out[by_id[label]] = by_id[label]
    return out


def read_properties(name: str) -> dict:
    props = {"name": name, "A": 0.0, "components": []}
    for t in tokens(MATERIALS / name / "Material_Properties.dat"):
        key = t[0]
        if key in ("Atomic_Weight", "Molecular_Weight"):
            props["A"] = float(t[1])
        elif key == "Density":
            props["density"] = float(t[1])
        elif key == "Electron_Density":
            props["electron_density"] = float(t[1])
        elif key == "Radiation_Length":
            props["X0"] = float(t[1])
        elif key == "Nuclear_Data":
            props["type"] = {"Mixture": TYPE_MIXTURE, "ICRU": TYPE_ICRU, "proton-proton": TYPE_PP}[t[1]]
        elif key == "Mixture_Component":
            props["components"].append((t[2], float(t[3]) / 100.0))
    return props


def read_stop_pow(name: str) -> np.ndarray:
    data = np.loadtxt(MATERIALS / name / "G4_Stop_Pow.dat")
    # MCsquare indexes this table as uniform 0.5 MeV bins from 0.
    expected = PSTAR_BIN * np.arange(len(data))
    if not np.allclose(data[:, 0], expected, atol=1e-6):
        raise ValueError(f"{name}: G4_Stop_Pow.dat is not on a uniform {PSTAR_BIN} MeV grid from 0")
    return data[:, 1] * 1e6  # eV cm^2 / g


def read_elastic(name: str, A: float):
    energies, sigma, cdfs = [], [], []
    lines = list(tokens(MATERIALS / name / "ICRU_Nuclear_elastic.dat"))
    i = 0
    while i < len(lines):
        t = lines[i]
        if t[0] == "Energy":
            energies.append(float(t[1]))
        elif t[0] == "Cross_section":
            sigma.append(N_AVO * float(t[1]) * 1e-24 / A)
        elif t[0] == "Differential_cross_section":
            cdf = np.zeros(36)
            for k in range(36):
                i += 1
                angle, value = float(lines[i][0]), float(lines[i][1])
                if angle != (k + 1) * 5.0:
                    raise ValueError(f"{name}: elastic angle {angle} at row {k}")
                if k == 0:
                    w = solid_angle(7.5) - solid_angle(5)
                elif k < 35:
                    w = solid_angle((k + 1) * 5 + 2.5) - solid_angle((k + 1) * 5 - 2.5)
                else:
                    w = 4 * math.pi - solid_angle(177.5)
                cdf[k] = (cdf[k - 1] if k else 0.0) + value * w
            cdfs.append(cdf / cdf[35])
        i += 1
    return np.array(energies), np.array(sigma), np.array(cdfs)


def _dd_weights() -> np.ndarray:
    a = ICRU_ANGLES
    w = np.zeros(13)
    w[0] = solid_angle(5)
    w[12] = 4 * math.pi - solid_angle(165)
    for j in range(1, 12):
        w[j] = solid_angle((a[j] + a[j + 1]) / 2) - solid_angle((a[j - 1] + a[j]) / 2)
    return w


def read_inelastic(name: str, A: float):
    """Per incident energy: E, sigma, mult[3], recoil, and per species rows
    (secondary energy, cumulative D, cumulative-over-angle DD). Sampling
    interpolates then accumulates; both are linear so pre-accumulating is exact."""
    weights = _dd_weights()
    records = []
    lines = list(tokens(MATERIALS / name / "ICRU_Nuclear_inelastic.dat"))
    i = 0
    while i < len(lines):
        t = lines[i]
        if t[0] == "Energy":
            records.append({"E": float(t[1]), "sigma": 0.0, "mult": [0.0, 0.0, 0.0], "recoil": 0.0,
                            "rows": [None, None, None]})
        elif t[0] == "Cross_section":
            records[-1]["sigma"] = N_AVO * float(t[1]) * 1e-24 / A
        elif t[0] == "Multiplicity":
            records[-1]["mult"][SPECIES.index(t[1])] = float(t[2])
        elif t[0] == "Energy_fraction_recoils":
            records[-1]["recoil"] = float(t[1])
        elif t[0] == "Differential_cross_section":
            s = SPECIES.index(t[1])
            n = int(t[2])
            rows = np.array([[float(v) for v in lines[i + 1 + r][:15]] for r in range(n)])
            i += n
            dd = rows[:, 2:15] * weights
            records[-1]["rows"][s] = (rows[:, 0], np.cumsum(rows[:, 1]), np.cumsum(dd, axis=1))
        i += 1
    return records


def interp_total(props: dict, elements: dict) -> np.ndarray:
    """MCsquare ``Interp_Nuclear_Cross_section`` on 1 MeV bins 0..249."""
    out = np.zeros(INTERP_BINS)

    def pp(e):
        return (0.315 * e ** -1.126 + 3.78e-6 * e) / 0.1119

    def icru_clamped(el, e):
        def clamped(E, S):
            k = min(max(binary_search(e, E), 0), len(E) - 2)
            return lerp(e, E[k], E[k + 1], S[k], S[k + 1])

        return max(clamped(el["el_E"], el["el_sigma"]), 0.0) + clamped(el["in_E"], el["in_sigma"])

    def icru_mixture(el, e):
        total = 0.0
        for E, S in ((el["el_E"], el["el_sigma"]), (el["in_E"], el["in_sigma"])):
            k = binary_search(e, E)
            total += lerp(e, E[k], E[k + 1], S[k], S[k + 1])
        return total

    for j in range(INTERP_BINS):
        e = float(j)
        if props["type"] == TYPE_PP:
            out[j] = pp(e) if e >= 10.0 else 0.0
        elif props["type"] == TYPE_ICRU:
            out[j] = icru_clamped(elements[props["name"]], e)
        else:
            for comp, frac in props["components"]:
                el = elements[comp]
                if el["type"] == TYPE_PP:
                    if e >= 10.0:
                        out[j] += frac * pp(e)
                elif e > 7:
                    out[j] += frac * icru_mixture(el, e)
    return out


def build() -> dict:
    MEDIA = media()
    ids = material_ids()
    media_props = {key: read_properties(name) for key, name in MEDIA.items()}
    element_names: list[str] = []
    for props in media_props.values():
        names = [c[0] for c in props["components"]] if props["type"] == TYPE_MIXTURE else [props["name"]]
        for n in names:
            if n not in element_names:
                element_names.append(n)

    elements = {}
    for n in element_names:
        p = read_properties(n)
        el = {"type": p["type"], "A": p["A"]}
        if p["type"] == TYPE_ICRU:
            el["el_E"], el["el_sigma"], el["el_cdf"] = read_elastic(n, p["A"])
            el["inelastic"] = read_inelastic(n, p["A"])
            el["in_E"] = np.array([r["E"] for r in el["inelastic"]])
            el["in_sigma"] = np.array([r["sigma"] for r in el["inelastic"]])
        elements[n] = el

    out: dict[str, np.ndarray] = {
        "media": np.array(list(MEDIA)),
        "mcsquare_names": np.array(list(MEDIA.values())),
        "mcsquare_ids": np.array([ids[n] for n in MEDIA.values()], dtype=np.int32),
        "elements": np.array(element_names),
    }
    nm = len(MEDIA)
    out["m_props"] = np.array([[p["density"], p["electron_density"] / p["density"], p["X0"]]
                               for p in media_props.values()])
    out["m_type"] = np.array([p["type"] for p in media_props.values()], dtype=np.int32)
    width = max(len(p["components"]) if p["type"] == TYPE_MIXTURE else 1 for p in media_props.values())
    out["m_comp"] = np.full((nm, width), -1, dtype=np.int32)
    out["m_frac"] = np.zeros((nm, width))
    for m, p in enumerate(media_props.values()):
        comps = p["components"] if p["type"] == TYPE_MIXTURE else [(p["name"], 1.0)]
        for k, (n, f) in enumerate(comps):
            out["m_comp"][m, k] = element_names.index(n)
            out["m_frac"][m, k] = f
    stops = [read_stop_pow(name) for name in MEDIA.values()]
    out["m_stop"] = np.array(stops)
    out["m_nuc"] = np.array([interp_total(p, elements) for p in media_props.values()])

    out["e_type"] = np.array([elements[n]["type"] for n in element_names], dtype=np.int32)
    out["e_A"] = np.array([elements[n]["A"] for n in element_names])

    el_start, el_E, el_sigma, el_cdf = [0], [], [], []
    in_start, in_E, in_sigma, in_mult, in_recoil = [0], [], [], [], []
    rows_start, rows_n, sec_E, sec_cD, sec_cDD = [], [], [], [], []
    n_rows = 0
    for n in element_names:
        el = elements[n]
        if el["type"] == TYPE_ICRU:
            el_E += list(el["el_E"])
            el_sigma += list(el["el_sigma"])
            el_cdf += list(el["el_cdf"])
            for r in el["inelastic"]:
                in_E.append(r["E"])
                in_sigma.append(r["sigma"])
                in_mult.append(r["mult"])
                in_recoil.append(r["recoil"])
                starts, counts = [], []
                for rows in r["rows"]:
                    if rows is None:
                        starts.append(n_rows)
                        counts.append(0)
                        continue
                    e, cd, cdd = rows
                    starts.append(n_rows)
                    counts.append(len(e))
                    sec_E += list(e)
                    sec_cD += list(cd)
                    sec_cDD += list(cdd)
                    n_rows += len(e)
                rows_start.append(starts)
                rows_n.append(counts)
        el_start.append(len(el_E))
        in_start.append(len(in_E))

    # MCsquare samples rows of incident energy i and i+1 with row count of i.
    for s in range(3):
        for e in range(len(element_names)):
            for k in range(in_start[e], in_start[e + 1] - 1):
                a, b = rows_n[k][s], rows_n[k + 1][s]
                if a and b and a != b:
                    raise ValueError(f"{element_names[e]}: {SPECIES[s]} row count changes at {in_E[k]} MeV")

    out.update(
        el_start=np.array(el_start, dtype=np.int32), el_E=np.array(el_E), el_sigma=np.array(el_sigma),
        el_cdf=np.array(el_cdf).reshape(-1, 36),
        in_start=np.array(in_start, dtype=np.int32), in_E=np.array(in_E), in_sigma=np.array(in_sigma),
        in_mult=np.array(in_mult).reshape(-1, 3), in_recoil=np.array(in_recoil),
        in_rows_start=np.array(rows_start, dtype=np.int32).reshape(-1, 3),
        in_rows_n=np.array(rows_n, dtype=np.int32).reshape(-1, 3),
        sec_E=np.array(sec_E), sec_cD=np.array(sec_cD), sec_cDD=np.array(sec_cDD).reshape(-1, 13),
    )
    return out


def main() -> int:
    if not MATERIALS.is_dir():
        print(f"missing {MATERIALS}; run: git submodule update --init", file=sys.stderr)
        return 1
    tables = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **tables)
    print(f"wrote {OUT.relative_to(ROOT)}: media={list(tables['media'])} elements={list(tables['elements'])}")
    for data in ("Scanners", "BDL"):
        shutil.copytree(MCSQUARE / data, OUT.parent / "mcsquare" / data, dirs_exist_ok=True)
    print("copied the MCsquare scanner calibrations and beam models to scan_kit/assets/mcsquare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
