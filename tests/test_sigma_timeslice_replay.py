"""Compatibility tests for sigma channel loading (unified catalog)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.views.timeslice_replay_channels import load_session_timeline_catalog

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G2_SESSION = "590658542"
G3_SESSION = "1091134775"


def test_load_g2_session_timeline() -> None:
    data = load_session_timeline_catalog(G2_SESSION, str(TEST_DATA))
    assert data is not None
    assert data["n_samples"] > 0
    assert len(data["sigma_ic1_x"]) == data["n_samples"]
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.nanmedian(data["sigma_ic1_x"]) > 0.5


def test_load_g3_session_timeline() -> None:
    data = load_session_timeline_catalog(G3_SESSION, str(TEST_DATA))
    assert data is not None
    assert np.isfinite(data["sigma_ic1_x"]).any()
    assert np.all(data["sigma_ic1_x"][np.isfinite(data["sigma_ic1_x"])] > 0)
