"""Tests for layer-aggregated IC current ratio loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scan_kit.common.current_ratios import load_session_current_ratios, sym_pct
from scan_kit.views.binned_summary_catalog import PRESET_CURRENT_RATIO_ENERGY, Y_CURRENT_RATIO
from scan_kit.views.binned_summary_data import load_sessions_current_ratios

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"


def test_sym_pct_basic() -> None:
    a = np.array([100.0, 200.0])
    b = np.array([110.0, 180.0])
    out = sym_pct(b, a)
    assert out[0] > 0
    assert out[1] < 0


def test_load_current_ratios_skips_empty_timeslice_frames(monkeypatch) -> None:
    from scan_kit.common.session_source import SessionSource

    src = SessionSource(kind="directory", path=Path("/fake"), session_id="sess")
    input_map = pd.DataFrame({"energy": [200.0], "layer_id": [1]})
    empty = pd.DataFrame({"_layer_idx": pd.Series([], dtype=int)})
    nonempty = pd.DataFrame(
        {
            "ic1_current": [50.0] * 20,
            "ic2_current": [50.0] * 20,
        }
    )
    nonempty["_layer_idx"] = 0

    monkeypatch.setattr(
        "scan_kit.common.current_ratios.resolve_session_source",
        lambda sid, base: src,
    )
    monkeypatch.setattr(
        "scan_kit.common.current_ratios.load_session_csv",
        lambda s, name: input_map if name == "input_map.csv" else None,
    )
    monkeypatch.setattr(
        "scan_kit.common.current_ratios.load_session_timeslice_device_units",
        lambda s: [empty, nonempty],
    )

    result = load_session_current_ratios("sess", "/fake")
    assert result is not None
    assert len(result["energy"]) == 1
    assert "ic21_ratio" in result


def test_current_ratio_loader_g3() -> None:
    data = load_sessions_current_ratios([G3_SESSION], str(TEST_DATA))
    if not data:
        pytest.skip("current ratio data unavailable in fixture")
    assert PRESET_CURRENT_RATIO_ENERGY
    assert Y_CURRENT_RATIO in data[G3_SESSION] or "ic21_ratio" in data[G3_SESSION]
