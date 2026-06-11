import numpy as np
from scan_kit.views import amplifier_correlation as ac


def med(r):
    r = r[np.isfinite(r)]
    return float(np.median(np.abs(r)))


for sid in ["1759842921", "870222918", "1943968267"]:
    s = ac._load_session_samples(sid, "test_data")
    if s is None:
        continue
    B = ac._energy_corrected_field_kg(s.field_x, s.momentum)
    arc = ac._arc_fit(B, s.angle_x_mrad)
    print(f"{sid} X: med|res|={med(arc.residual):.4f}  gain={arc.c1:.4f} mrad/kG "
          f"cubic={arc.c3*1000:.4f} urad/kG3 offset={arc.c0:.4f} mrad")
