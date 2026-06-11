import numpy as np
from scan_kit.views import amplifier_correlation as ac

for sid in ["1759842921", "870222918", "1943968267"]:
    s = ac._load_session_samples(sid, "test_data")
    if s is None:
        continue
    B = ac._energy_corrected_field_kg(s.field_x, s.momentum)
    sin_scaled = np.sin(s.angle_x_mrad / 1000.0) * 1000.0
    cub = ac._cubic_fit(B, sin_scaled)
    print(f"{sid} X: c0={cub.c0:.4f} c1={cub.c1:.4f} c2={cub.c2:.5f} c3={cub.c3:.6f}")
