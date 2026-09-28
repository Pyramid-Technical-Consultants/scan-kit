"""MCsquare beam data library (BDL): the UPenn double-Gaussian pencil-beam model.

:meth:`BeamModel.spot_records` turns spots (nominal energy, position at the isocenter
plane, MU) into the kernel's phase-space records, interpolating the BDL in nominal energy
exactly as MCsquare's ``Sample_particle`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MCSQUARE_BDL = Path(__file__).resolve().parents[1] / "assets" / "mcsquare" / "BDL"
COLUMNS = (
    "NominalEnergy", "MeanEnergy", "EnergySpread", "ProtonsMU",
    "Weight1", "SpotSize1x", "Divergence1x", "Correlation1x", "SpotSize1y", "Divergence1y", "Correlation1y",
    "Weight2", "SpotSize2x", "Divergence2x", "Correlation2x", "SpotSize2y", "Divergence2y", "Correlation2y",
)
BDL_STRIDE = 40
RS_DISTANCE_MM = 400.0  # MCsquare's IsocenterToRangeShifterDistance when a plan leaves it out


@dataclass(frozen=True)
class RangeShifter:
    id: str
    material: int  # MCsquare material label
    density: float  # g/cm3
    wet: float  # mm, the BDL's nominal value; plans give their own per layer
    kind: str = "binary"


@dataclass(frozen=True)
class BeamModel:
    name: str
    nozzle_to_iso: float  # mm
    smx_to_iso: float
    smy_to_iso: float
    table: dict = field(repr=False)  # COLUMNS -> per-energy arrays, ascending NominalEnergy
    range_shifters: tuple[RangeShifter, ...] = ()

    @classmethod
    def read(cls, path: str | Path) -> BeamModel:
        path = Path(path) if Path(path).is_file() else MCSQUARE_BDL / f"{path}.txt"
        lines = path.read_text(encoding="latin-1").splitlines()
        if not lines or "UPenn" not in lines[0]:
            raise ValueError(f"{path}: only the UPenn double-Gaussian BDL is supported")
        values, shifters, rows = {}, [], None
        i = 0
        while i < len(lines):
            head = lines[i].strip()
            if head in ("Nozzle exit to Isocenter distance", "SMX to Isocenter distance", "SMY to Isocenter distance"):
                values[head] = float(lines[i + 1].split()[0])
                i += 2
                continue
            if head == "Range Shifter parameters":
                rs = {}
                i += 1
                while i < len(lines) and "=" in lines[i]:
                    key, _, value = lines[i].split("#")[0].partition("=")
                    rs[key.strip()] = value.strip()
                    i += 1
                shifters.append(RangeShifter(rs["RS_ID"], int(rs["RS_material"]), float(rs["RS_density"]),
                                             float(rs.get("RS_WET", 0.0)), rs.get("RS_type", "binary")))
                continue
            if head == "Beam parameters":
                n = int(lines[i + 1].split()[0])
                numeric = [ln.split() for ln in lines[i + 2:] if ln.split() and _is_number(ln.split()[0])]
                rows = np.array([[float(v) for v in parts] for parts in numeric[:n]])
                break
            i += 1
        if rows is None or rows.shape[1] != len(COLUMNS) or len(rows) < 2:
            raise ValueError(f"{path}: expected at least two energies of {len(COLUMNS)} beam parameters")
        if np.any(np.diff(rows[:, 0]) <= 0.0):
            raise ValueError(f"{path}: nominal energies must strictly increase")
        return cls(path.stem, values["Nozzle exit to Isocenter distance"], values["SMX to Isocenter distance"],
                   values["SMY to Isocenter distance"], {c: rows[:, k] for k, c in enumerate(COLUMNS)},
                   tuple(shifters))

    def at(self, energy) -> dict:
        """Every column at nominal *energy* (MeV): MCsquare's Sequential_Search bracket, linear in between."""
        e = np.asarray(energy, dtype=float)
        nominal = self.table["NominalEnergy"]
        i = np.clip(np.searchsorted(nominal, e, side="left") - 1, 0, len(nominal) - 2)
        t = (e - nominal[i]) / (nominal[i + 1] - nominal[i])
        return {c: v[i] + t * (v[i + 1] - v[i]) for c, v in self.table.items()}

    def protons_per_mu(self, energy) -> np.ndarray:
        return self.at(energy)["ProtonsMU"]

    def range_shifter(self, rs_id: str) -> RangeShifter:
        for rs in self.range_shifters:
            if rs.id == rs_id:
                return rs
        # ponytail: TPS and BDL IDs often differ; a single-shifter BDL is taken to be that shifter.
        if len(self.range_shifters) == 1:
            return self.range_shifters[0]
        raise ValueError(f"beam model {self.name!r} has no range shifter {rs_id!r}")

    def spot_records(self, energy, x, y, *, beam=0, rs_id="", rs_wet=0.0, rs_distance=np.nan) -> np.ndarray:
        """``(n, 40)`` float32 kernel records, CDF left for :meth:`McRun.patient` to fill.

        Per-spot arrays or scalars: nominal *energy* MeV, *x*, *y* mm at the isocenter plane,
        *beam* index into the run's beam records, and the range shifter: *rs_id*, *rs_wet* mm
        (0 when OUT, NaN when IN at the BDL's nominal WET) and *rs_distance* mm from the
        isocenter to its downstream face.
        """
        from ..dicom.calibration import relative_stop
        from ..views.dose_mc import load_tables

        energy, x, y, beam, rs_wet, rs_distance = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (energy, x, y, beam, rs_wet, rs_distance)))
        n = energy.size
        rec = np.zeros((n, BDL_STRIDE), dtype=np.float64)
        p = self.at(energy.reshape(-1))
        rec[:, 0] = x.reshape(-1)
        rec[:, 1] = y.reshape(-1)
        rec[:, 2] = p["MeanEnergy"]
        rec[:, 3] = p["EnergySpread"] * energy.reshape(-1) / 100.0
        rec[:, 4] = p["Weight1"] / (p["Weight1"] + p["Weight2"])
        rec[:, 6] = beam.reshape(-1)
        for at, g in ((8, "1"), (20, "2")):
            for off, axis in ((0, "x"), (6, "y")):
                T, s = _phase_space(p[f"SpotSize{g}{axis}"], p[f"Divergence{g}{axis}"], p[f"Correlation{g}{axis}"])
                rec[:, at + off:at + off + 4] = T.reshape(n, 4)
                rec[:, at + off + 4:at + off + 6] = s
        wet = rs_wet.reshape(-1)
        on = ~(wet <= 0.0)
        if on.any():
            rs = self.range_shifter(rs_id)
            wet = np.where(np.isnan(wet), rs.wet, wet)
            if np.any(wet[on] <= 0.0):
                raise ValueError(f"range shifter {rs_id!r} has no WET in the plan or in beam model {self.name!r}")
            ids = [int(v) for v in load_tables()["mcsquare_ids"]]
            if rs.material not in ids:
                raise ValueError(f"range shifter material {rs.material} is not in the Monte Carlo tables")
            medium = ids.index(rs.material)
            spr = rs.density * float(relative_stop(np.array([medium]))[0])
            dist = np.where(np.isfinite(rs_distance.reshape(-1)), rs_distance.reshape(-1), RS_DISTANCE_MM)
            rec[on, 32] = wet[on] / spr / 10.0
            rec[on, 33] = dist[on] / 10.0
            rec[on, 34] = medium
            rec[on, 35] = rs.density
        return rec.astype(np.float32)


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _phase_space(size, divergence, correlation) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of each (x, θ) covariance in mm and rad: ``(T (n,2,2), sigmas (n,2))``."""
    s, d, c = (np.asarray(v, dtype=float).reshape(-1) for v in (size, divergence, correlation))
    A = np.empty((s.size, 2, 2))
    A[:, 0, 0] = s * s
    A[:, 1, 1] = d * d
    A[:, 0, 1] = A[:, 1, 0] = c * s * d
    vals, vecs = np.linalg.eigh(A)
    # MCsquare's diagonalize puts the larger eigenvalue first; the sign of each vector is free.
    return vecs[:, :, ::-1], np.sqrt(np.maximum(vals[:, ::-1], 0.0))
