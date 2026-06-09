"""Tests for beam position error summary view."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.timeslice_position_error import load_session_beam_on_position_errors

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_session_loads_beam_on_errors() -> None:
    errors = load_session_beam_on_position_errors("1091134775", str(TEST_DATA))
    assert errors is not None
    assert errors.ic1_x.size > 0
    assert np.isfinite(errors.ic1_x).any()
    assert np.isfinite(errors.ic2_y).any()


def test_g2_session_loads_beam_on_errors() -> None:
    errors = load_session_beam_on_position_errors("590658542", str(TEST_DATA))
    assert errors is not None
    assert errors.ic1_x.size > 0
    assert np.isfinite(errors.ic1_x).any()
    assert np.isfinite(errors.ic2_y).any()
    ic1_cov = np.isfinite(errors.ic1_x).mean()
    assert ic1_cov > 0.9
    assert np.nanmedian(np.abs(errors.ic1_x)) < 0.1
    assert np.nanpercentile(np.abs(errors.ic2_x[np.isfinite(errors.ic2_x)]), 95) < 1.0
