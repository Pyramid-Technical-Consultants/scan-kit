"""Tests for layer-aggregated MU delivery rate loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan_kit.common.mu_delivery_rate import (
    load_session_mu_delivery_rates,
    probe_session_mu_delivery_rates,
)
from scan_kit.views.binned_summary_catalog import PRESET_BY_ID, PRESET_DOSE_RATE_ENERGY, Y_DOSE_RATE
from scan_kit.views.binned_summary_data import (
    available_y_groups,
    load_sessions_dose_rate,
    probe_view_option_availability,
)
from scan_kit.data import DATA_SOURCE_SPOT_ISO, option_key
from tests.conftest import G3_SESSION, TEST_DATA


def test_load_session_mu_delivery_rates_g3() -> None:
    data = load_session_mu_delivery_rates(G3_SESSION, str(TEST_DATA))
    if data is None:
        return
    assert probe_session_mu_delivery_rates(G3_SESSION, str(TEST_DATA))
    assert data["energy"].size == data["mu_rate"].size
    assert data["energy"].size >= 1
    assert np.all(np.isfinite(data["energy"]))
    assert np.all(np.isfinite(data["mu_rate"]))
    assert np.all(data["mu_rate"] > 0)
    assert float(data["session_avg_rate"]) > 0


def test_dose_rate_available_in_binned_summary_probe(
    g3_dose_rate,
    g3_spot_summary,
    g3_source_availability,
) -> None:
    if not g3_dose_rate:
        return
    assert Y_DOSE_RATE in available_y_groups(g3_dose_rate)
    availability = probe_view_option_availability(
        [G3_SESSION],
        str(TEST_DATA),
        spot_data=g3_spot_summary,
        registry_availability=g3_source_availability,
    )
    assert availability.get(option_key(DATA_SOURCE_SPOT_ISO, Y_DOSE_RATE))


def test_dose_rate_preset_exists() -> None:
    preset = PRESET_BY_ID[PRESET_DOSE_RATE_ENERGY]
    assert preset.y_group == Y_DOSE_RATE
    assert preset.glyph == "mean"
