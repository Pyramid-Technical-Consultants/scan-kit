"""Tests for shared timeline channel catalog."""

from __future__ import annotations

import numpy as np
import pytest

from scan_kit.data.timeline_channels import (
    FFT_CHANNEL_SPECS,
    REPLAY_CHANNEL_SPECS,
    TIMELINE_CHANNEL_BY_KEY,
    TIMELINE_CHANNEL_SPECS,
    channel_available,
)
from scan_kit.views.fft_catalog import CHANNEL_BY_ID, FFT_METRICS
from scan_kit.views.timeslice_replay_catalog import REPLAY_METRICS
from scan_kit.views.timeslice_replay_channels import CATALOG_FAMILY_KEYS, CHANNEL_DEFS
from tests.conftest import (
    AMP_G3_SESSION,
    G2_SESSION,
    G3_SESSION,
    HAS_TEST_DATA,
    TEST_DATA,
)


def test_replay_channels_match_shared_specs() -> None:
    assert tuple(c.key for c in CHANNEL_DEFS) == tuple(
        spec.key for spec in REPLAY_CHANNEL_SPECS
    )


def test_fft_channels_cover_shared_fft_specs() -> None:
    fft_keys = {channel.id for channel in CHANNEL_BY_ID.values()}
    assert fft_keys == {spec.key for spec in FFT_CHANNEL_SPECS}


def test_catalog_families_cover_all_timeline_channels() -> None:
    """A listed channel that isn't in a catalog family silently fails to load."""
    covered = set().union(*CATALOG_FAMILY_KEYS)
    listed = {spec.key for spec in TIMELINE_CHANNEL_SPECS}
    missing = listed - covered
    orphan = covered - listed
    assert not missing, f"timeline channels have no catalog family: {sorted(missing)}"
    assert not orphan, f"catalog family keys with no timeline spec: {sorted(orphan)}"


def test_ic_usecols_omit_sigma_and_position() -> None:
    from scan_kit.views.timeslice_replay_channels import timeslice_usecols_for_channel_keys

    cols = timeslice_usecols_for_channel_keys(frozenset({"ic1", "ic2", "ic3"}))
    assert cols is not None
    joined = " ".join(cols)
    assert "ic1_current" in cols
    assert "sigma" not in joined
    assert "peak_amplitude" not in joined
    assert "rci_in_trigger" in cols


def test_fft_header_probe_maps_every_fft_channel() -> None:
    """Unmapped FFT channels stay hidden even when the file has the columns."""
    from scan_kit.views.fft_data import _channel_available_from_flags

    class _AllTrue(dict):
        def get(self, key, default=None):
            return True

    for spec in FFT_CHANNEL_SPECS:
        assert _channel_available_from_flags(spec.key, _AllTrue()), spec.key


def test_fft_metrics_group_by_family() -> None:
    for metric in FFT_METRICS:
        families = {TIMELINE_CHANNEL_BY_KEY[ch.id].family for ch in metric.channels}
        assert len(families) == 1


def test_fft_gaussian_fit_metrics_are_adjacent() -> None:
    ids = [metric.id for metric in FFT_METRICS]
    gauss = ("position", "sigma", "peak_amplitude")
    idxs = [ids.index(metric_id) for metric_id in gauss]
    assert idxs == list(range(min(idxs), max(idxs) + 1))


def test_g3_timeline_channel_availability(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    for spec in REPLAY_CHANNEL_SPECS:
        if spec.key in {"ic1", "ic2", "bx", "by"}:
            assert channel_available(data, spec)


def _opened_one_frame(session_id: str):
    from scan_kit.common.timeslice_table import load_session_timeslice_frames

    opened = load_session_timeslice_frames(
        session_id, str(TEST_DATA), max_frames=1,
    )
    if opened is None:
        pytest.skip(f"no timeslice frames for {session_id}")
    return opened


@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ is not available")
@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION, AMP_G3_SESSION])
def test_advertised_fft_metrics_load_from_catalog(session_id: str) -> None:
    from scan_kit.views.fft_data import (
        channel_keys_for_metric,
        probe_fft_metric_availability_headers,
    )
    from scan_kit.views.timeslice_replay_channels import load_session_timeline_catalog

    avail = probe_fft_metric_availability_headers([session_id], str(TEST_DATA))
    opened = _opened_one_frame(session_id)
    advertised = [metric for metric in FFT_METRICS if avail.get(metric.id, False)]
    assert advertised, f"no FFT metrics advertised for {session_id}"
    for metric in advertised:
        keys = channel_keys_for_metric(metric.id)
        data = load_session_timeline_catalog(
            session_id, str(TEST_DATA), opened=opened, channel_keys=keys,
        )
        assert data is not None, metric.id
        present = [
            key for key in keys
            if key in data and np.asarray(data[key]).size > 0
        ]
        assert present, (
            f"{session_id} {metric.id} advertised but catalog returned {sorted(data)}"
        )


@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ is not available")
def test_ic_catalog_skips_g3_iso_context(monkeypatch) -> None:
    from scan_kit.views.timeslice_replay_channels import load_session_timeline_catalog

    def boom(*_args, **_kwargs):
        raise AssertionError("iso error context should not run for IC-only catalog")

    monkeypatch.setattr(
        "scan_kit.common.timeslice_position_error.resolve_session_timeslice_error_source",
        boom,
    )
    opened = _opened_one_frame(G3_SESSION)
    data = load_session_timeline_catalog(
        G3_SESSION,
        str(TEST_DATA),
        opened=opened,
        channel_keys=frozenset({"ic1", "ic2", "ic3"}),
    )
    assert data is not None
    assert "ic1" in data


@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ is not available")
def test_ic_usecols_load_keeps_currents() -> None:
    from scan_kit.common.timeslice_table import load_session_timeslice_frames
    from scan_kit.views.timeslice_replay_channels import (
        load_session_timeline_catalog,
        timeslice_usecols_for_channel_keys,
    )

    keys = frozenset({"ic1", "ic2", "ic3"})
    opened = load_session_timeslice_frames(
        G3_SESSION,
        str(TEST_DATA),
        max_frames=1,
        usecols=timeslice_usecols_for_channel_keys(keys),
    )
    if opened is None:
        pytest.skip(f"no timeslice frames for {G3_SESSION}")
    data = load_session_timeline_catalog(
        G3_SESSION, str(TEST_DATA), opened=opened, channel_keys=keys,
    )
    assert data is not None
    assert "ic1" in data
    assert np.asarray(data["ic1"]).size > 0
    if data.get("has_ic3"):
        assert "ic3" in data


@pytest.mark.skipif(not HAS_TEST_DATA, reason="test_data/ is not available")
@pytest.mark.parametrize("session_id", [G3_SESSION, G2_SESSION])
def test_replay_metrics_load_from_catalog(session_id: str) -> None:
    from scan_kit.views.timeslice_replay_channels import load_session_timeline_catalog

    opened = _opened_one_frame(session_id)
    loaded_any = False
    for metric in REPLAY_METRICS:
        keys = frozenset(metric.channel_keys)
        data = load_session_timeline_catalog(
            session_id, str(TEST_DATA), opened=opened, channel_keys=keys,
        )
        if data is None:
            continue
        present = [
            key for key in keys
            if key in data and np.asarray(data[key]).size > 0
        ]
        if present:
            loaded_any = True
    assert loaded_any, f"no replay metrics loaded for {session_id}"
