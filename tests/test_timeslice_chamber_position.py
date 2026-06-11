"""Tests for timeslice chamber-plane IC position loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.session_source import (
    load_session_timeslice_device_units,
    resolve_session_source,
)
from scan_kit.common.timeslice_position_error import (
    TIMESLICE_POSITION_ERROR_COLS,
    frame_timeslice_chamber_position_arrays,
    resolve_session_timeslice_chamber_position_source,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def test_g3_chamber_positions_have_finite_samples() -> None:
    src = resolve_session_source("1943968267", str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    source = resolve_session_timeslice_chamber_position_source(src, frames)
    assert source is not None
    assert source.mode == "g3_chamber"
    arrays = frame_timeslice_chamber_position_arrays(frames[0], source)
    assert arrays is not None
    assert np.isfinite(arrays[0]).any()


def test_g2_chamber_positions_have_finite_samples() -> None:
    src = resolve_session_source("590658542", str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    source = resolve_session_timeslice_chamber_position_source(src, frames)
    assert source is not None
    assert source.mode == "g2_chamber"
    arrays = frame_timeslice_chamber_position_arrays(frames[0], source)
    assert arrays is not None
    assert np.isfinite(arrays[0]).any()


def test_g2_ic2_strip_direction_is_consistent_across_frames() -> None:
    """IC2 strip sense is decided once per session, so |IC1-IC2| stays small.

    Per-frame direction flips used to produce a bimodal IC2 cloud (the IC2
    median drifting far from IC1), faking a large beam tilt.
    """
    from scan_kit.common import detect_beam_on_mask

    src = resolve_session_source("590658542", str(TEST_DATA))
    assert src is not None
    frames = load_session_timeslice_device_units(
        src, usecols=TIMESLICE_POSITION_ERROR_COLS
    )
    source = resolve_session_timeslice_chamber_position_source(src, frames)
    assert source is not None

    ic1_parts: list[np.ndarray] = []
    ic2_parts: list[np.ndarray] = []
    for df in frames:
        beam = detect_beam_on_mask(df)
        if beam is None:
            continue
        arrays = frame_timeslice_chamber_position_arrays(df, source)
        if arrays is None:
            continue
        ic1_x, _, ic2_x, _ = arrays
        ic1_parts.append(ic1_x[beam])
        ic2_parts.append(ic2_x[beam])

    ic1 = np.concatenate(ic1_parts)
    ic2 = np.concatenate(ic2_parts)
    # IC2 should track IC1 to within a few mm; a flipped subset would push the
    # IC2 median tens of mm away from IC1.
    assert abs(np.nanmedian(ic2) - np.nanmedian(ic1)) < 10.0
    assert np.nanmedian(np.abs(ic1 - ic2)) < 10.0
