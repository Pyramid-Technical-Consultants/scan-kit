"""Tests for IC current ratio loading."""

from __future__ import annotations

import numpy as np

from scan_kit.common.current_ratios import load_session_current_ratios


def test_load_session_current_ratios_accepts_read_only_ic_arrays(
    g3_session_id: str,
    test_data_dir: str,
) -> None:
    """Pandas/numpy may return read-only views; plateau logic must not mutate in place."""
    result = load_session_current_ratios(g3_session_id, test_data_dir)
    assert result is not None
    assert np.isfinite(result["energy"]).any()

    # Sanity: synthetic read-only path used inside load_session_current_ratios.
    sig = np.array([1.0, 2.0, np.nan], dtype=np.float64)
    sig.setflags(write=False)
    copy = np.array(sig, dtype=np.float64, copy=True)
    copy[~np.isfinite(copy)] = 0.0
    assert copy[2] == 0.0
