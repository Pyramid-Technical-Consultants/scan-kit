"""Tests for spot sigma-error availability via devices.xml targets."""

from __future__ import annotations

import numpy as np

from scan_kit.data import DATA_SOURCE_SPOT_ISO, LoadOptions, SessionContext, load
from scan_kit.data.adapters.binned import sigma_error_to_binned_columns
from scan_kit.data.availability import probe_source_option
from scan_kit.data.sources.sigma_error import SOURCE_SIGMA_ERROR
from tests.conftest import G3_SESSION, TEST_DATA


def test_spot_sigma_error_available_when_devices_xml_present() -> None:
    assert probe_source_option(
        G3_SESSION,
        str(TEST_DATA),
        SOURCE_SIGMA_ERROR,
        DATA_SOURCE_SPOT_ISO,
    )


def test_spot_sigma_error_loads_from_devices_xml_targets() -> None:
    payload = load(
        SOURCE_SIGMA_ERROR,
        SessionContext(G3_SESSION, str(TEST_DATA)),
        LoadOptions(data_source=DATA_SOURCE_SPOT_ISO),
    )
    assert payload is not None
    assert "energy" in payload
    assert np.any(np.isfinite(payload["ic1_x_err"]))
    binned = sigma_error_to_binned_columns(payload)
    assert binned is not None
    assert "ic1_sig_x_err" in binned
