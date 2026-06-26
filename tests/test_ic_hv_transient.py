"""Tests for the IC HV transient test view: capacitance reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.views import ic_hv_transient
from scan_kit.views.ic_hv_transient import _load_session_hv, _parse_result, run

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
SESSION = "447945951"


def test_load_session_hv_devices() -> None:
    data = _load_session_hv(SESSION, str(TEST_DATA))
    assert data is not None
    assert set(data) == {"FX4", "IX256_1", "IX256_2"}


def test_delta_v_read_from_config() -> None:
    data = _load_session_hv(SESSION, str(TEST_DATA))
    assert data is not None
    # config: starting 1900, ending 1980 -> 80 V step, for every device.
    for dev in data.values():
        assert dev.delta_v == 80.0


def test_quad_capacitance_in_pf_range() -> None:
    data = _load_session_hv(SESSION, str(TEST_DATA))
    assert data is not None
    fx4 = data["FX4"]
    assert not fx4.has_strips
    assert len(fx4.hcc_caps) == 4
    # All four HCC channels should land in a sane few-pF range.
    caps = np.array(list(fx4.hcc_caps.values()))
    assert np.all((caps > 0.5) & (caps < 50.0))
    assert fx4.overall == "pass"


def test_strip_capacitance_and_cross_check() -> None:
    data = _load_session_hv(SESSION, str(TEST_DATA))
    assert data is not None
    ix = data["IX256_2"]
    assert ix.has_strips
    assert ix.strip_caps.size == 256
    assert ix.strip_fail is not None and ix.strip_fail.size == 256
    # Primary HCC capacitance should be a few hundred pF, in 1..1000 pF range.
    assert 1.0 < ix.primary_cap < 1000.0
    # Strip-sum should track the primary channel within a factor of ~2
    # (sparse strip sampling undersamples the fast decay).
    strip_sum = float(np.nansum(ix.strip_caps))
    assert 0.4 * ix.primary_cap < strip_sum < 1.6 * ix.primary_cap
    assert ix.overall == "fail"


def test_parse_result_handles_scalar_and_array() -> None:
    overall, channels = _parse_result('{"result": {"x/value": "pass"}}')
    assert overall == "pass"
    assert channels == []

    overall, channels = _parse_result(
        '{"result": {"a/test_results/value": ["pass", "fail", "pass"], '
        '"a/summary/value": "fail"}}'
    )
    assert overall == "fail"
    assert channels == ["pass", "fail", "pass"]


def test_missing_hv_data_returns_none() -> None:
    assert _load_session_hv("590658542", str(TEST_DATA)) is None


def test_run_smoke(monkeypatch) -> None:
    shown = {"n": 0}

    def _fake_finish(fig, *args, **kwargs):
        shown["n"] += 1

    monkeypatch.setattr(ic_hv_transient, "finish_view", _fake_finish)
    run([SESSION], str(TEST_DATA))
    assert shown["n"] == 1

    run(["590658542"], str(TEST_DATA))  # no HV data: must not raise
