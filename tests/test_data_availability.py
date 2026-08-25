"""Tests for unified session availability probing."""

from __future__ import annotations

from scan_kit.data import option_key, REGISTRY, probe_session, probe_sessions
from scan_kit.data.sources.dose_rate import SOURCE_DOSE_RATE
from scan_kit.data.sources.position_error import SOURCE_POSITION_ERROR
from scan_kit.views.unified_catalog import DATA_SOURCE_SPOT, DATA_SOURCE_TIMESLICE
from tests.conftest import G3_SESSION, TEST_DATA


def test_registry_lists_all_core_sources() -> None:
    expected = {
        "ic12_pos_diff",
        "position",
        "position_error",
        "sigma",
        "sigma_error",
        "confidence",
        "gaussian_fit_filter",
        "dose_rate",
        "current_ratio",
        "ic_current",
    }
    assert expected <= set(REGISTRY.keys())


def test_probe_session_returns_option_keys() -> None:
    avail = probe_session(G3_SESSION, str(TEST_DATA))
    assert option_key(DATA_SOURCE_SPOT, SOURCE_POSITION_ERROR) in avail
    assert option_key(DATA_SOURCE_TIMESLICE, SOURCE_POSITION_ERROR) in avail


def test_probe_sessions_matches_single_session() -> None:
    single = probe_session(G3_SESSION, str(TEST_DATA))
    merged = probe_sessions([G3_SESSION], str(TEST_DATA))
    assert merged == single


def test_dose_rate_availability_key() -> None:
    avail = probe_session(G3_SESSION, str(TEST_DATA))
    key = option_key(DATA_SOURCE_SPOT, SOURCE_DOSE_RATE)
    assert key in avail
    assert avail[key]
