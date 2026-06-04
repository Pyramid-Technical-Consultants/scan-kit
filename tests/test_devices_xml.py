"""Tests for devices.xml parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from scan_kit.common.devices_xml import (
    IC_DEVICE_TO_SIG_KEY,
    parse_devices_xml,
    load_session_devices_config,
)

_SAMPLE = """\
<?xml version="1.0" encoding="utf-8"?>
<MapToMap type="device" version="1.0">
 <devices>
  <ion_chamber>
   <device name="IC_1_X"/>
   <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="249.5" max_energy="250.5" K0="2.588" K1="0.0" K2="0.0" K3="0.0"/>
   <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="69.5" max_energy="70.5" K0="5.983" K1="0.0" K2="0.0" K3="0.0"/>
  </ion_chamber>
  <ion_chamber>
   <device name="IC_1_Y"/>
   <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="249.5" max_energy="250.5" K0="3.100" K1="0.0" K2="0.0" K3="0.0"/>
  </ion_chamber>
  <ion_chamber>
   <device name="IC_1_HCC"/>
   <beam_sigma_conversions in_units="MEV" out_units="mm" min_energy="69.99" max_energy="250.01" K0="2.55665" K1="-0.0029056" K2="393.568" K3="2474.07"/>
  </ion_chamber>
 </devices>
</MapToMap>
"""


def test_parse_beam_sigma_band_lookup() -> None:
    config = parse_devices_xml(_SAMPLE)
    assert config.expected_sigma_mm("IC_1_X", 250.0) == 2.588
    assert config.expected_sigma_mm("IC_1_X", 70.0) == 5.983
    assert config.expected_sigma_mm("IC_1_X", 100.0) is None


def test_expected_sigmas_by_key_maps_devices() -> None:
    config = parse_devices_xml(_SAMPLE)
    by_key = config.expected_sigmas_by_key([250.0, 70.0], keys=("ic1_sig_x", "ic1_sig_y"))
    assert by_key["ic1_sig_x"][250.0] == 2.588
    assert by_key["ic1_sig_x"][70.0] == 5.983
    assert by_key["ic1_sig_y"][250.0] == 3.1
    assert "ic1_sig_y" not in by_key or 70.0 not in by_key.get("ic1_sig_y", {})


def test_polynomial_sigma_conversion() -> None:
    config = parse_devices_xml(_SAMPLE)
    conv = config.beam_sigmas["IC_1_HCC"][0]
    energy = 200.0
    expected = conv.k0 + conv.k1 * energy + conv.k2 * energy**2 + conv.k3 * energy**3
    assert conv.sigma_mm(energy) == pytest.approx(expected)


def test_ic_device_mapping_covers_sigma_view_keys() -> None:
    assert set(IC_DEVICE_TO_SIG_KEY.values()) == {
        "ic1_sig_x",
        "ic1_sig_y",
        "ic2_sig_x",
        "ic2_sig_y",
    }


def test_load_fixture_session_devices_xml() -> None:
    root = Path(__file__).resolve().parent.parent
    config = load_session_devices_config("1943968267", root / "test_data")
    assert config is not None
    sigma = config.expected_sigma_mm("IC_1_X", 250.0)
    assert sigma is not None
    assert 2.0 < sigma < 3.0
