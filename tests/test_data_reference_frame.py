"""Unit tests for reference-frame policy helpers."""

from __future__ import annotations

from scan_kit.data.reference_frame import (
    TIMESLICE_POSITION_KEYS,
    spot_positions_raw,
    spot_sigma_prefer_raw,
    timeslice_position_loader,
    timeslice_position_table_hooks,
)
from scan_kit.data.types import REFERENCE_CHAMBER, REFERENCE_ISO
from scan_kit.common.timeslice_position_error import (
    load_session_beam_on_chamber_positions,
    load_session_beam_on_iso_positions,
)


def test_spot_positions_raw_maps_chamber_to_raw() -> None:
    assert spot_positions_raw(REFERENCE_CHAMBER) is True
    assert spot_positions_raw(REFERENCE_ISO) is False


def test_spot_sigma_prefer_raw_maps_frames() -> None:
    assert spot_sigma_prefer_raw(REFERENCE_CHAMBER) is True
    assert spot_sigma_prefer_raw(REFERENCE_ISO) is False


def test_timeslice_position_loader_dispatches() -> None:
    assert timeslice_position_loader(REFERENCE_ISO) is load_session_beam_on_iso_positions
    assert (
        timeslice_position_loader(REFERENCE_CHAMBER)
        is load_session_beam_on_chamber_positions
    )


def test_timeslice_position_table_hooks_keys() -> None:
    for frame in (REFERENCE_ISO, REFERENCE_CHAMBER):
        _prepare, _extract, keys = timeslice_position_table_hooks(frame)
        assert keys == TIMESLICE_POSITION_KEYS
