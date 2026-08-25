"""Compatibility tests for sigma channel loading (unified catalog)."""

from __future__ import annotations

import numpy as np


def test_load_g2_session_timeline(g2_timeline_catalog) -> None:
    data = g2_timeline_catalog
    assert data is not None
    assert data["n_samples"] > 0
    assert len(data["sigma_ic1_x"]) == data["n_samples"]
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.nanmedian(data["sigma_ic1_x"]) > 0.5


def test_load_g3_session_timeline(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    assert data is not None
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.all(data["sigma_ic1_x"][np.isfinite(data["sigma_ic1_x"])] > 0)
