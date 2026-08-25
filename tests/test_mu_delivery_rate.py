"""Tests for layer-aggregated MU delivery rate loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.mu_delivery_rate import load_session_mu_delivery_rates
from scan_kit.views.binned_summary_catalog import PRESET_BY_ID, PRESET_DOSE_RATE_ENERGY, Y_DOSE_RATE
from scan_kit.views.binned_summary_data import (
    available_y_groups,
    load_sessions_dose_rate,
    probe_view_option_availability,
)
from scan_kit.views.unified_catalog import DATA_SOURCE_SPOT, option_key

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"


def test_load_session_mu_delivery_rates_g3() -> None:
    data = load_session_mu_delivery_rates(G3_SESSION, str(TEST_DATA))
    if data is None:
        return
    assert data["energy"].size == data["mu_rate"].size
    assert data["energy"].size >= 1
    assert np.all(np.isfinite(data["energy"]))
    assert np.all(np.isfinite(data["mu_rate"]))
    assert np.all(data["mu_rate"] > 0)
    assert float(data["session_avg_rate"]) > 0


def test_dose_rate_available_in_binned_summary_probe() -> None:
    dose_rate = load_sessions_dose_rate([G3_SESSION], str(TEST_DATA))
    if not dose_rate:
        return
    assert Y_DOSE_RATE in available_y_groups(dose_rate)
    availability = probe_view_option_availability(
        [G3_SESSION],
        str(TEST_DATA),
        dose_rate_data=dose_rate,
    )
    assert availability.get(option_key(DATA_SOURCE_SPOT, Y_DOSE_RATE))


def test_dose_rate_preset_exists() -> None:
    preset = PRESET_BY_ID[PRESET_DOSE_RATE_ENERGY]
    assert preset.y_group == Y_DOSE_RATE
    assert preset.glyph == "mean"
