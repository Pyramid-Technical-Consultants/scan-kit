"""Tests for shared timeline channel catalog."""

from __future__ import annotations

from scan_kit.data.timeline_channels import (
    FFT_CHANNEL_SPECS,
    REPLAY_CHANNEL_SPECS,
    TIMELINE_CHANNEL_BY_KEY,
    channel_available,
)
from scan_kit.views.fft_catalog import CHANNEL_BY_ID, FFT_METRICS
from scan_kit.views.timeslice_replay_channels import CHANNEL_DEFS


def test_replay_channels_match_shared_specs() -> None:
    assert tuple(c.key for c in CHANNEL_DEFS) == tuple(
        spec.key for spec in REPLAY_CHANNEL_SPECS
    )


def test_fft_channels_cover_shared_fft_specs() -> None:
    fft_keys = {channel.id for channel in CHANNEL_BY_ID.values()}
    assert fft_keys == {spec.key for spec in FFT_CHANNEL_SPECS}


def test_fft_metrics_group_by_family() -> None:
    for metric in FFT_METRICS:
        families = {TIMELINE_CHANNEL_BY_KEY[ch.id].family for ch in metric.channels}
        assert len(families) == 1


def test_g3_timeline_channel_availability(g3_timeline_catalog) -> None:
    data = g3_timeline_catalog
    for spec in REPLAY_CHANNEL_SPECS:
        if spec.key in {"ic1", "ic2", "bx", "by"}:
            assert channel_available(data, spec)

