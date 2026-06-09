"""Tests for G2 timeslice position error loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)
from scan_kit.common.timeslice_position_error import (
    load_session_beam_on_position_errors,
    resolve_session_timeslice_error_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G2_SESSION = "590658542"


def test_g2_resolves_filtered_ic1_error_columns() -> None:
    src = resolve_session_source(G2_SESSION, str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(src)
    assert frames

    source = resolve_session_timeslice_error_source(src, frames)
    assert source is not None
    assert source.mode == "direct"
    assert source.columns["ic1_x_err"] == "x_err_filtered"
    assert source.columns["ic1_y_err"] == "y_err_filtered"


def test_g2_ic2_errors_exclude_spot_transition_ramps() -> None:
    errors = load_session_beam_on_position_errors(G2_SESSION, str(TEST_DATA))
    assert errors is not None

    ic2_x = errors.ic2_x[np.isfinite(errors.ic2_x)]
    ic2_y = errors.ic2_y[np.isfinite(errors.ic2_y)]
    assert ic2_x.size > 0
    assert ic2_y.size > 0
    assert np.max(np.abs(ic2_x)) < 10.0
    assert np.max(np.abs(ic2_y)) < 10.0
    assert np.nanpercentile(np.abs(ic2_x), 95) < 0.2
    assert np.nanpercentile(np.abs(ic2_y), 95) < 0.3
