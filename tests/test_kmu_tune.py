"""Formula and XML-write tests for kMU (dose calibration) auto-tuning."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scan_kit.workflows.config_tuning.auto_tuning.kmu_tune import (
    MeasuredDoseSpots,
    apply_kmu_to_tree,
    ic_family_from_device,
    is_raw_dose_column,
    merge_measured_dose_spots,
    resolve_spot_dose_mu_column,
)
from scan_kit.workflows.config_tuning.auto_tuning.registry import (
    AUTO_TUNE_REGISTRY,
    get_auto_tune_workflow,
)
from scan_kit.workflows.config_tuning.auto_tuning.workflows.kmu_tuning import (
    KmuTuningWorkflow,
)

_PRI_KMU = 2.0e-08
_PRI_HCC_KMU = 2.2e-08
_SEC_KMU = 3.0e-08
_SEC_HCC_KMU = 3.3e-08


def _devices_xml() -> str:
    return f"""\
<MapToMap>
 <devices>
  <ion_chamber>
   <device name="IC_1_X"/>
   <gain_conversions>
    <gain_conversion in_units="gP" units="coulombs" K0="1"/>
    <gain_conversion in_units="MU" units="coulombs" K_MU="{_PRI_KMU}"/>
   </gain_conversions>
  </ion_chamber>
  <ion_chamber>
   <device name="IC_1_HCC"/>
   <gain_conversions>
    <gain_conversion in_units="MU" units="coulombs" K_MU="{_PRI_HCC_KMU}"/>
   </gain_conversions>
  </ion_chamber>
  <ion_chamber>
   <device name="IC_2_X"/>
   <gain_conversions>
    <gain_conversion in_units="MU" units="coulombs" K_MU="{_SEC_KMU}"/>
   </gain_conversions>
  </ion_chamber>
  <ion_chamber>
   <device name="IC_2_HCC"/>
   <gain_conversions>
    <gain_conversion in_units="MU" units="coulombs" K_MU="{_SEC_HCC_KMU}"/>
   </gain_conversions>
  </ion_chamber>
  <scan_magnet>
   <device name="SC_X"/>
   <gain_conversions>
    <gain_conversion in_units="mm" units="volts" K1="1" K_MU="9.9e-08"/>
   </gain_conversions>
  </scan_magnet>
 </devices>
</MapToMap>
"""


def _spots(
    pri: np.ndarray,
    sec: np.ndarray,
    plan: np.ndarray | None = None,
) -> MeasuredDoseSpots:
    n = pri.size
    energy = np.full(n, 200.0)
    charge_req = plan if plan is not None else pri.copy()
    return MeasuredDoseSpots(
        mu_by_ic={
            "ic1": np.asarray(pri, dtype=float),
            "ic2": np.asarray(sec, dtype=float),
            "ic3": np.full(n, np.nan),
        },
        energy=energy,
        charge_req=np.asarray(charge_req, dtype=float),
    )


def _kmu(root: ET.Element, device: str) -> float:
    for chamber in root.iter("ion_chamber"):
        name_el = chamber.find("device")
        if name_el is None or name_el.get("name") != device:
            continue
        for conv in chamber.iter("gain_conversion"):
            if (conv.get("in_units") or "").upper() != "MU":
                continue
            return float(conv.get("K_MU") or "nan")
    raise AssertionError(f"no K_MU for {device}")


def test_resolver_prefers_processed_mu_over_raw() -> None:
    columns = ["ic1_total_dose_spot", "ic1_total_dose_spot_raw", "other"]
    assert resolve_spot_dose_mu_column(columns, "ic1") == "ic1_total_dose_spot"
    assert not is_raw_dose_column("ic1_total_dose_spot")
    assert is_raw_dose_column("ic1_total_dose_spot_raw")


def test_resolver_uses_canonical_processed_name_when_raw_still_present() -> None:
    columns = ["ic1_total_dose", "ic1_total_dose_spot_raw"]
    assert resolve_spot_dose_mu_column(columns, "ic1") == "ic1_total_dose"


def test_resolver_never_returns_raw_nC_column() -> None:
    assert resolve_spot_dose_mu_column(["ic1_total_dose_spot_raw"], "ic1") is None
    assert resolve_spot_dose_mu_column(["ic1_dose_spot_raw"], "ic1") is None


def test_ic_family_from_device() -> None:
    assert ic_family_from_device("IC_1_HCC") == "ic1"
    assert ic_family_from_device("IC_2_Y") == "ic2"
    assert ic_family_from_device("IC_3") == "ic3"
    assert ic_family_from_device("SC_X") is None


def test_unchanged_primary_scales_secondaries_to_primary() -> None:
    pri = np.full(10, 1.0)
    sec = np.full(10, 1.02)
    root = ET.fromstring(_devices_xml())
    result = apply_kmu_to_tree(root, _spots(pri, sec), primary="ic1", mode="unchanged")
    assert result.ok
    assert _kmu(root, "IC_1_X") == pytest.approx(_PRI_KMU)
    assert _kmu(root, "IC_1_HCC") == pytest.approx(_PRI_HCC_KMU)
    scale = float(sec.sum() / pri.sum())
    assert _kmu(root, "IC_2_X") == pytest.approx(_SEC_KMU * scale)
    assert _kmu(root, "IC_2_HCC") == pytest.approx(_SEC_HCC_KMU * scale)
    written = {row.device: row for row in result.rows if row.will_write}
    assert "IC_1_X" not in written
    assert written["IC_2_X"].scale == pytest.approx(scale)


def test_percent_decreases_primary_kmu_and_secondaries_follow_adjusted() -> None:
    pri = np.full(10, 1.0)
    sec = np.full(10, 1.05)
    root = ET.fromstring(_devices_xml())
    result = apply_kmu_to_tree(
        root, _spots(pri, sec), primary="ic1", mode="percent", percent=2.0
    )
    assert result.ok
    assert _kmu(root, "IC_1_X") == pytest.approx(_PRI_KMU / 1.02)
    assert _kmu(root, "IC_1_HCC") == pytest.approx(_PRI_HCC_KMU / 1.02)
    mu_target = float(pri.sum()) * 1.02
    sec_scale = float(sec.sum()) / mu_target
    assert _kmu(root, "IC_2_X") == pytest.approx(_SEC_KMU * sec_scale)
    by_device = {row.device: row for row in result.rows}
    assert by_device["IC_2_X"].vs_primary_pct_after == pytest.approx(0.0, abs=1e-9)


def test_known_mu_increases_primary_kmu() -> None:
    pri = np.full(10, 1.0)
    sec = np.full(10, 1.0)
    root = ET.fromstring(_devices_xml())
    result = apply_kmu_to_tree(
        root, _spots(pri, sec), primary="ic1", mode="known_mu", known_mu=9.8
    )
    assert result.ok
    scale = 10.0 / 9.8
    assert _kmu(root, "IC_1_X") == pytest.approx(_PRI_KMU * scale)
    assert _kmu(root, "IC_2_X") == pytest.approx(_SEC_KMU * scale)


def test_pooled_sessions_use_sums_not_mean_of_ratios() -> None:
    s1 = _spots(np.full(10, 1.0), np.full(10, 1.2))
    s2 = _spots(np.full(10, 9.0), np.full(10, 9.0))
    merged = merge_measured_dose_spots([s1, s2])
    assert merged is not None
    root = ET.fromstring(_devices_xml())
    result = apply_kmu_to_tree(root, merged, primary="ic1", mode="unchanged")
    pooled = 102.0 / 100.0
    mean_of_ratios = (1.2 + 1.0) / 2.0
    assert result.ok
    assert _kmu(root, "IC_2_X") == pytest.approx(_SEC_KMU * pooled)
    assert _kmu(root, "IC_2_X") != pytest.approx(_SEC_KMU * mean_of_ratios)


def test_large_scale_warns_but_still_writes() -> None:
    pri = np.full(10, 1.0)
    sec = np.full(10, 1.4)
    root = ET.fromstring(_devices_xml())
    result = apply_kmu_to_tree(root, _spots(pri, sec), primary="ic1", mode="unchanged")
    assert result.ok
    assert _kmu(root, "IC_2_X") == pytest.approx(_SEC_KMU * 1.4)
    assert any("1.4000" in warning for warning in result.warnings)


def test_scan_magnet_kmu_is_not_written() -> None:
    pri = np.full(10, 1.0)
    sec = np.full(10, 1.02)
    root = ET.fromstring(_devices_xml())
    apply_kmu_to_tree(root, _spots(pri, sec), primary="ic1", mode="unchanged")
    magnet = root.find(".//scan_magnet/gain_conversions/gain_conversion")
    assert magnet is not None
    assert magnet.get("K_MU") == "9.9e-08"


def test_kmu_workflow_is_registered() -> None:
    ids = {workflow.id for workflow in AUTO_TUNE_REGISTRY}
    assert "kmu_tuning" in ids
    assert get_auto_tune_workflow("kmu_tuning") is not None
    workflow = KmuTuningWorkflow()
    assert workflow.uses_session_browser()
    assert workflow.validate({"session_ids": [], "data_dir": ""})
