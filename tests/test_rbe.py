"""Variable proton RBE models against the linear-quadratic iso-effect they solve."""

from __future__ import annotations

import numpy as np
import pytest

from scan_kit.qa.rbe import CARABE, CONSTANT, MCNAMARA, UNKELBACH, WEDENBERG, rbe, rbe_limits


@pytest.mark.parametrize("model", [MCNAMARA, WEDENBERG, CARABE])
def test_rbe_gives_the_photon_dose_of_equal_effect(model) -> None:
    beta_x = 0.05
    d = np.array([0.5, 1.8, 2.0, 5.0, 8.0])  # Gy per fraction
    let = np.array([0.5, 2.0, 4.0, 8.0, 15.0])  # keV/µm
    for ab in (2.0, 3.0, 10.0):
        hi, lo = rbe_limits(model, let, ab)
        alpha_x = ab * beta_x
        proton = hi * alpha_x * d + lo * lo * beta_x * d * d
        dx = rbe(model, d, let, ab) * d
        assert np.allclose(alpha_x * dx + beta_x * dx * dx, proton, rtol=1e-9)


def test_rbe_limits_and_trends() -> None:
    let = np.linspace(1.0, 12.0, 12)
    for model in (MCNAMARA, WEDENBERG, CARABE):
        hi, _lo = rbe_limits(model, let, 2.0)
        assert np.allclose(rbe(model, np.zeros_like(let), let, 2.0), hi)  # dose -> 0 gives RBEmax
        r = rbe(model, 2.0, let, 2.0)
        assert np.all(np.diff(r) > 0)  # more LET, more effect
        assert np.all(rbe(model, 2.0, let, 10.0) < r)  # late-responding tissue gains more
    assert np.allclose(rbe(CONSTANT, np.ones(3), let[:3]), 1.1)
    assert rbe(UNKELBACH, 2.0, 2.5) == pytest.approx(1.1)
    # McNamara in a typical target (2 Gy, LETd 2 keV/µm, (α/β)x 10 Gy): close to the clinical 1.1.
    assert float(rbe(MCNAMARA, 2.0, 2.0, 10.0)) == pytest.approx(1.0663, abs=1e-3)
