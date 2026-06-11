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
    th = s.angle_x_mrad
    cub = ac._cubic_fit(B, th)
    arc = ac._arc_fit(B, th)
    print(f"{sid} X max|angle|={np.nanmax(np.abs(th)):.0f} mrad")
    print(f"   cubic med|res|={med(cub.residual):.4f}  gain={cub.c1:.4f} mrad/kG cubic={cub.c3*1000:.4f} urad/kG3")
    print(f"   arc   med|res|={med(arc.residual):.4f}  gain={arc.c1:.4f} mrad/kG cubic={arc.c3*1000:.4f} urad/kG3")
