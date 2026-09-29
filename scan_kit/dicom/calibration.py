"""CT calibration: HU to mass density and to Monte Carlo material, or a direct density / SPR map.

The HU tables use MCsquare's ``Scanners/<name>`` format and lookup rules, so a voxel gets
the same density and material MCsquare would give it. The calibration's :attr:`digest`
goes into every result that depends on it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ct import DicomError

MCSQUARE_SCANNERS = Path(__file__).resolve().parents[1] / "assets" / "mcsquare" / "Scanners"
SPR_ENERGY_MEV = 100.0  # MCsquare's range-shifter SPR energy; also used for SPR maps
_HU_SCAN = np.arange(-1100.0, 4001.0, 1.0)


def _rows(path: Path) -> list[list[float]]:
    out = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("#"):
            continue
        parts = line.split("#")[0].split()
        if len(parts) >= 2:
            out.append([float(parts[0]), float(parts[1])])
    return out


def _below(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """MCsquare ``Binary_Search``: index of the last value strictly below *x*, or -1."""
    return np.searchsorted(values, x, side="left") - 1


@dataclass(frozen=True)
class CtCalibration:
    name: str
    hu_points: np.ndarray  # ascending HU of the density curve
    densities: np.ndarray  # g/cm3 at hu_points
    hu_thresholds: np.ndarray  # ascending HU where each material starts
    labels: np.ndarray  # MCsquare material label per threshold

    @classmethod
    def mcsquare(cls, scanner: str | Path = "default") -> CtCalibration:
        """Read ``HU_Density_Conversion.txt`` and ``HU_Material_Conversion.txt`` from a scanner folder."""
        folder = Path(scanner) if Path(scanner).is_dir() else MCSQUARE_SCANNERS / str(scanner)
        dens = np.array(_rows(folder / "HU_Density_Conversion.txt"))
        mats = np.array(_rows(folder / "HU_Material_Conversion.txt"))
        if len(dens) < 2 or len(mats) < 1:
            raise DicomError(f"{folder}: calibration needs at least two density points and one material")
        for col, what in ((dens[:, 0], "density"), (mats[:, 0], "material")):
            if np.any(np.diff(col) <= 0.0):
                raise DicomError(f"{folder}: HU of the {what} table must strictly increase")
        return cls(folder.name, dens[:, 0], dens[:, 1], mats[:, 0], mats[:, 1].astype(np.int32))

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.hu_points, self.densities, self.hu_thresholds, self.labels.astype(np.float64)):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]

    def density(self, hu) -> np.ndarray:
        """g/cm3, linear in HU between points and extrapolated past the ends, floored at 1e-6."""
        hu = np.asarray(hu, dtype=np.float64)
        i = np.clip(_below(self.hu_points, hu), 0, len(self.hu_points) - 2)
        x0, x1 = self.hu_points[i], self.hu_points[i + 1]
        y0, y1 = self.densities[i], self.densities[i + 1]
        d = (y1 - y0) / (x1 - x0) * (hu - x0) + y0
        return np.where(d <= 0.0, 1e-6, d).astype(np.float32)

    def material(self, hu) -> np.ndarray:
        """Monte Carlo medium index (into the material tables) per HU, as uint8."""
        i = np.clip(_below(self.hu_thresholds, np.asarray(hu, dtype=np.float64)), 0, len(self.labels) - 1)
        return self._media()[i]

    def _media(self) -> np.ndarray:
        from ..views.dose_mc import load_tables

        ids = [int(v) for v in load_tables()["mcsquare_ids"]]
        missing = sorted({int(label) for label in self.labels} - set(ids))
        if missing:
            raise DicomError(f"calibration {self.name!r} uses MCsquare materials {missing} the tables lack; "
                             "add them and rerun scripts/build_mc_tables.py")
        return np.array([ids.index(int(label)) for label in self.labels], dtype=np.uint8)

    def spr(self, hu) -> np.ndarray:
        """Stopping-power ratio to water at :data:`SPR_ENERGY_MEV`."""
        return self.density(hu) * relative_stop(self.material(hu))

    def voxels(self, hu) -> tuple[np.ndarray, np.ndarray]:
        """(material uint8, density float32) for the transport, per HU voxel."""
        hu = np.asarray(hu)
        lo, hi = (float(hu.min()), float(hu.max())) if hu.size else (0.0, 0.0)
        if hi - lo < 1 << 16 and np.array_equal(hu, np.rint(hu)):
            # Integer HU (every rescaled CT): the curve once per HU value, then a lookup.
            keys = np.arange(lo, hi + 1.0)
            idx = (hu - lo).astype(np.intp)
            return self.material(keys)[idx], self.density(keys)[idx]
        return self.material(hu), self.density(hu)

    def from_density(self, density) -> tuple[np.ndarray, np.ndarray]:
        """A mass-density map: materials from the HU the curve gives that density."""
        dens = self.density(_HU_SCAN).astype(np.float64)
        hu = np.interp(np.asarray(density, dtype=np.float64), np.maximum.accumulate(dens), _HU_SCAN)
        return self.material(hu), np.asarray(density, dtype=np.float32)

    def from_spr(self, spr) -> tuple[np.ndarray, np.ndarray]:
        """An SPR map (dual-energy or photon-counting CT): materials from the matching HU,
        densities so each voxel's stopping power matches its SPR at the reference energy."""
        curve = np.maximum.accumulate(self.spr(_HU_SCAN).astype(np.float64))
        spr = np.asarray(spr, dtype=np.float64)
        mat = self.material(np.interp(spr, curve, _HU_SCAN))
        return mat, (spr / relative_stop(mat)).astype(np.float32)


def relative_stop(material: np.ndarray) -> np.ndarray:
    """Mass stopping power of each medium over water's, at the reference energy."""
    from ..views.dose_mc import load_tables

    t = load_tables()
    b = int(round(SPR_ENERGY_MEV / 0.5))
    water = list(t["media"]).index("water")
    return (t["m_stop"][:, b] / t["m_stop"][water, b])[np.asarray(material, dtype=np.intp)]
