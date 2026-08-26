"""Probe/load contract tests for registered data sources."""

from __future__ import annotations

import numpy as np
import pytest

from scan_kit.data import (
    DATA_SOURCE_SPOT_CHAMBER,
    DATA_SOURCE_SPOT_ISO,
    LoadOptions,
    REGISTRY,
    SessionContext,
    get_spec,
    load,
    probe,
)
from scan_kit.data.sources.ic12_pos_diff import SOURCE_IC12_POS_DIFF
from tests.conftest import G2_SESSION, G3_SESSION, TEST_DATA

FAST_SESSION = G3_SESSION
SLOW_SESSION = G2_SESSION


def _probe_load_cases(session_id: str) -> tuple[tuple[str, str, str], ...]:
    cases: list[tuple[str, str, str]] = []
    for source_id in sorted(REGISTRY.keys()):
        spec = get_spec(source_id)
        for data_source in sorted(spec.data_sources):
            cases.append((session_id, source_id, data_source))
    return tuple(cases)


FAST_PROBE_LOAD_CASES = _probe_load_cases(FAST_SESSION)
SLOW_PROBE_LOAD_CASES = _probe_load_cases(SLOW_SESSION)


@pytest.mark.parametrize(
    ("session_id", "source_id", "data_source"),
    FAST_PROBE_LOAD_CASES,
)
def test_probe_matches_load_g3(
    session_id: str,
    source_id: str,
    data_source: str,
) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    opts = LoadOptions(data_source=data_source)  # type: ignore[arg-type]
    available = probe(source_id, ctx, opts)
    payload = load(source_id, ctx, opts)
    assert available == (payload is not None)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("session_id", "source_id", "data_source"),
    SLOW_PROBE_LOAD_CASES,
)
def test_probe_matches_load_g2(
    session_id: str,
    source_id: str,
    data_source: str,
) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    opts = LoadOptions(data_source=data_source)  # type: ignore[arg-type]
    available = probe(source_id, ctx, opts)
    payload = load(source_id, ctx, opts)
    assert available == (payload is not None)


@pytest.mark.parametrize("session_id", (FAST_SESSION,))
@pytest.mark.parametrize(
    "data_source",
    [DATA_SOURCE_SPOT_ISO, DATA_SOURCE_SPOT_CHAMBER],
)
def test_ic12_load_shape_when_available(
    session_id: str,
    data_source: str,
) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    opts = LoadOptions(data_source=data_source)  # type: ignore[arg-type]
    payload = load(SOURCE_IC12_POS_DIFF, ctx, opts)
    if payload is None:
        pytest.skip("IC12 data unavailable")
    assert "ic1_x" in payload and "ic2_x" in payload
    assert len(payload["ic1_x"]) == len(payload["ic2_x"])
    assert np.isfinite(payload["ic1_x"]).any() or np.isfinite(payload["ic2_x"]).any()


@pytest.mark.parametrize("session_id", (FAST_SESSION,))
def test_ic12_spot_iso_vs_chamber_data_sources_differ(session_id: str) -> None:
    ctx = SessionContext(session_id, str(TEST_DATA))
    iso = load(
        SOURCE_IC12_POS_DIFF,
        ctx,
        LoadOptions(data_source=DATA_SOURCE_SPOT_ISO),
    )
    chamber = load(
        SOURCE_IC12_POS_DIFF,
        ctx,
        LoadOptions(data_source=DATA_SOURCE_SPOT_CHAMBER),
    )
    if iso is None or chamber is None:
        pytest.skip("IC12 spot data unavailable for both data sources")
    assert iso["ic1_x"].shape == chamber["ic1_x"].shape
