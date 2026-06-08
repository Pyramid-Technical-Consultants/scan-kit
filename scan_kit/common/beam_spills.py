"""Beam spill segmentation from hardware gate timeslice signals."""

from __future__ import annotations

import numpy as np

from .processing import _detect_beam_off_mask

MIN_SPILL_GAP_MS = 500
MS_PER_SLICE = 1.0


def detect_beam_on_mask(df, *, strict: bool = False) -> np.ndarray | None:
    """Return a boolean mask that is True when the hardware gate confirms beam on.

    Uses spill-level gating by default (G3 ``rci_in_trigger``, G2 ``r_beamOk``).
    Returns ``None`` when no gate columns are present.
    """
    off = _detect_beam_off_mask(df, strict=strict)
    if off is None:
        return None
    return ~off


def detect_spill_segments(
    beam_on: np.ndarray,
    *,
    gap_ms: float = MIN_SPILL_GAP_MS,
    min_on_slices: int = 2,
) -> list[tuple[int, int]]:
    """Segment a 1 ms boolean timeline into spill ranges.

    Brief off flickers shorter than *gap_ms* do not split a spill.  A spill
    ends after *gap_ms* consecutive off samples.  Segments with fewer than
    *min_on_slices* beam-on samples are dropped.

    Returns half-open slice ranges ``(start, end)``.
    """
    if beam_on.size == 0:
        return []

    gap_slices = max(1, int(round(gap_ms / MS_PER_SLICE)))
    on = np.asarray(beam_on, dtype=bool)
    segments: list[tuple[int, int]] = []

    in_spill = False
    start = 0
    off_count = 0

    for i, is_on in enumerate(on):
        if is_on:
            if not in_spill:
                start = i
                in_spill = True
            off_count = 0
        elif in_spill:
            off_count += 1
            if off_count >= gap_slices:
                end = i - off_count + 1
                if end - start >= min_on_slices:
                    segments.append((start, end))
                in_spill = False
                off_count = 0

    if in_spill:
        end = len(on)
        if end - start >= min_on_slices:
            segments.append((start, end))

    return segments
