"""Tests for beam-on timeslice confidence correlation loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.timeslice_confidence import (
    load_session_beam_on_confidence_correlations,
    resolve_timeslice_confidence_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_session_loads_confidence_correlations() -> None:
    samples = load_session_beam_on_confidence_correlations("1091134775", str(TEST_DATA))
    assert samples is not None
    assert samples.ic1_x_conf.size > 0
    assert np.isfinite(samples.ic1_x_conf).any()
    assert np.isfinite(samples.ic2_y_conf).any()
    assert samples.has_peak
    assert np.isfinite(samples.ic1_x_peak).any()
    assert np.isfinite(samples.ic1_primary).any()


def test_g2_session_loads_confidence_without_peak() -> None:
    samples = load_session_beam_on_confidence_correlations("590658542", str(TEST_DATA))
    assert samples is not None
    assert samples.ic1_x_conf.size > 0
    assert np.isfinite(samples.ic1_x_conf).any()
    assert not samples.has_peak
    assert np.isfinite(samples.ic1_primary).any()


def test_resolve_source_requires_confidence_columns() -> None:
    assert resolve_timeslice_confidence_source(["layer_id", "ic1_primary_channel"]) is None
