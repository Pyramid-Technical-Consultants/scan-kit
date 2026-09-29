"""Proton RBE: the clinical constant 1.1, and variable models of LETd, dose per fraction and (α/β)x.

The linear-quadratic models give RBEmax (the RBE as the dose goes to 0) and RBEmin (as it
grows); at a proton dose *d* per fraction

    RBE = (sqrt((α/β)² + 4 d (α/β) RBEmax + 4 d² RBEmin²) − α/β) / (2 d)

LETd is in water at unit density, from primary and secondary protons: what the models were
fitted to and what the EPTN consensus (Phys Imaging Radiat Oncol 2024) asks centres to report.
"""

from __future__ import annotations

import numpy as np

from .analysis import RBE as RBE_CONSTANT

CONSTANT, MCNAMARA, WEDENBERG, CARABE, UNKELBACH = "constant", "mcnamara", "wedenberg", "carabe", "unkelbach"
MODELS = (
    (CONSTANT, "Constant 1.1"),
    (MCNAMARA, "McNamara 2015"),
    (WEDENBERG, "Wedenberg 2013"),
    (CARABE, "Carabe 2012"),
    (UNKELBACH, "Unkelbach 2016 (1 + 0.04·LETd)"),
)
DEFAULT_ALPHA_BETA = 2.0  # Gy: the Dutch centres' McNamara reporting value (Radiother Oncol 202, 110653)
UNKELBACH_C = 0.04  # µm/keV


def uses_alpha_beta(model: str) -> bool:
    return model in (MCNAMARA, WEDENBERG, CARABE)


def rbe_limits(model: str, let, alpha_beta: float) -> tuple[np.ndarray, np.ndarray]:
    """(RBEmax, RBEmin) at LETd *let* (keV/µm) for (α/β)x in Gy."""
    let = np.asarray(let, dtype=np.float64)
    ab = float(alpha_beta)
    if model == MCNAMARA:
        # Fit values as tabulated with their standard errors; some tools carry p0 = 0.999064 instead.
        return 0.99064 + 0.35605 / ab * let, 1.1012 - 0.0038703 * np.sqrt(ab) * let
    if model == WEDENBERG:
        return 1.0 + 0.434 / ab * let, np.ones_like(let)
    if model == CARABE:
        r = 2.686 / ab * let
        return 0.843 + 0.154 * r, 1.09 + 0.006 * r
    raise ValueError(f"{model!r} is not a linear-quadratic RBE model")


def rbe(model: str, dose, let, alpha_beta: float = DEFAULT_ALPHA_BETA) -> np.ndarray:
    """RBE per voxel: *dose* physical Gy per fraction, *let* LETd keV/µm in water."""
    dose = np.asarray(dose, dtype=np.float64)
    if model == CONSTANT:
        return np.full(dose.shape, RBE_CONSTANT)
    let = np.maximum(np.asarray(let, dtype=np.float64), 0.0)
    if model == UNKELBACH:
        return 1.0 + UNKELBACH_C * let
    hi, lo = rbe_limits(model, let, alpha_beta)
    ab = float(alpha_beta)
    d = np.maximum(dose, 1e-6)
    r = (np.sqrt(ab * ab + 4.0 * d * ab * hi + 4.0 * d * d * lo * lo) - ab) / (2.0 * d)
    return np.where(dose > 1e-6, r, hi)
