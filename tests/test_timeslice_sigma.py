"""Tests for beam-on timeslice sigma loading and distribution view."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from scan_kit.common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)
from scan_kit.common.sigma_distribution import render_sigma_distribution
from scan_kit.common.timeslice_sigma import (
    TIMESLICE_SIGMA_COLS,
    load_session_beam_on_sigmas,
    resolve_timeslice_sigma_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G2_SESSION = "590658542"
G3_SESSION = "1091134775"


def test_g2_session_loads_beam_on_sigmas() -> None:
    sigmas = load_session_beam_on_sigmas(G2_SESSION, str(TEST_DATA))
    assert sigmas is not None
    assert sigmas.ic1_x.size > 0
    assert np.isfinite(sigmas.ic1_x).any()
    assert np.nanmedian(sigmas.ic1_x) > 0.5
    assert np.nanmedian(sigmas.ic1_x) < 10.0


def test_g3_session_loads_beam_on_sigmas() -> None:
    sigmas = load_session_beam_on_sigmas(G3_SESSION, str(TEST_DATA))
    assert sigmas is not None
    assert np.isfinite(sigmas.ic1_x).any()
    assert np.all(sigmas.ic1_x[np.isfinite(sigmas.ic1_x)] > 0)


def test_g3_resolves_sigma_source_with_quality_gating() -> None:
    src = resolve_session_source(G3_SESSION, str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(src, usecols=TIMESLICE_SIGMA_COLS)
    source = resolve_timeslice_sigma_source(frames[0].columns)
    assert source is not None
    assert source.mode == "g3"


def test_render_sigma_distribution_smoke() -> None:
    sigmas = load_session_beam_on_sigmas(G3_SESSION, str(TEST_DATA))
    assert sigmas is not None
    render_sigma_distribution(
        {G3_SESSION: sigmas},
        [G3_SESSION],
        title="Sigma Distribution (test)",
        base_dir=str(TEST_DATA),
    )
