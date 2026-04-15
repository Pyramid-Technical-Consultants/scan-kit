"""End-to-end test of constrained calibration mode."""
from scan_kit.common.processing import compute_calibration_factors, apply_calibration_factors
from scan_kit.common.settings import ViewSettings
import numpy as np

# Test compute_calibration_factors with a known session
sids = ["1565104513"]
factors = compute_calibration_factors(sids, "test_data")
print("Single session factors:", {k: f"{v:.6f}" for k, v in factors.items()})
assert len(factors) > 0, "Should have at least one factor"

# Test with multiple sessions
sids2 = ["1565104513", "585383148"]
factors2 = compute_calibration_factors(sids2, "test_data")
print("Multi-session factors:", {k: f"{v:.6f}" for k, v in factors2.items()})

# Test apply_calibration_factors
data = {"ic1_total_dose": np.array([100.0, 200.0, 300.0])}
result = apply_calibration_factors(data, ["ic1_total_dose"], {"ic1_total_dose": 1.05})
assert np.allclose(result["ic1_total_dose"], [105.0, 210.0, 315.0])
print("apply_calibration_factors: OK")

# Test ViewSettings round-trip with cal_factors
s = ViewSettings(calibration_mode="constrained", cal_factors=factors)
j = s.to_json()
s2 = ViewSettings.from_json(j)
assert s2.cal_factors == factors
assert s2.calibration_mode == "constrained"
assert s2.auto_calibrate is True
print("Settings round-trip: OK")

# Test per_session has no cal_factors
s3 = ViewSettings(calibration_mode="per_session")
assert s3.auto_calibrate is True
assert s3.cal_factors is None
print("Per-session mode: OK")

print("\nAll constrained calibration tests passed!")
