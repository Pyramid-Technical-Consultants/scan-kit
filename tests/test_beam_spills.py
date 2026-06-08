"""Tests for beam spill segmentation."""

import numpy as np
import pandas as pd

from scan_kit.common.beam_spills import (
    MIN_SPILL_GAP_MS,
    detect_beam_on_mask,
    detect_spill_segments,
)


def test_single_spill_no_gaps():
    beam_on = np.ones(1000, dtype=bool)
    assert detect_spill_segments(beam_on) == [(0, 1000)]


def test_two_spills_separated_by_exact_gap():
    gap = int(MIN_SPILL_GAP_MS)
    beam_on = np.zeros(2000, dtype=bool)
    beam_on[:200] = True
    beam_on[200 + gap : 200 + gap + 300] = True
    segments = detect_spill_segments(beam_on, gap_ms=MIN_SPILL_GAP_MS)
    assert segments == [(0, 200), (200 + gap, 200 + gap + 300)]


def test_brief_off_flicker_does_not_split():
    gap = int(MIN_SPILL_GAP_MS)
    beam_on = np.ones(1000, dtype=bool)
    beam_on[400:400 + gap - 1] = False
    segments = detect_spill_segments(beam_on, gap_ms=MIN_SPILL_GAP_MS)
    assert segments == [(0, 1000)]


def test_min_on_slices_filters_noise():
    beam_on = np.array([True, False, True, True, True, False], dtype=bool)
    assert detect_spill_segments(beam_on, gap_ms=1, min_on_slices=2) == [(2, 5)]


def test_detect_beam_on_mask_g3():
    df = pd.DataFrame({"rci_in_trigger": [1, 1, 0, 0, 1]})
    mask = detect_beam_on_mask(df)
    assert mask is not None
    np.testing.assert_array_equal(mask, [True, True, False, False, True])


def test_detect_beam_on_mask_g2():
    df = pd.DataFrame({"r_beamOk": [1, 0, 1]})
    mask = detect_beam_on_mask(df)
    assert mask is not None
    np.testing.assert_array_equal(mask, [True, False, True])


def test_detect_beam_on_mask_missing_columns():
    df = pd.DataFrame({"ic1_current": [1.0, 2.0]})
    assert detect_beam_on_mask(df) is None
