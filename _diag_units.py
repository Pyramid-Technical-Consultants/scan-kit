import numpy as np
from scan_kit.views import amplifier_correlation as ac

for sid in ["1359159551", "1759842921", "1943968267"]:
    s = ac._load_session_samples(sid, "test_data")
    if s is None:
        continue
    ec = ac._energy_corrected_field(s.field_x, s.momentum)  # Gauss
    fit = ac._cubic_fit(ec, s.angle_x_mrad)
    if fit is None:
        continue
    fmax = np.nanmax(np.abs(ec[np.isfinite(ec)]))
    print(f"=== {sid} X  (|field|max = {fmax:.0f} G = {fmax/1000:.2f} kG) ===")
    print(f"  c1: {fit.c1:.6g} mrad/G   = {fit.c1*1e3:.4g} mrad/kG")
    print(f"  c3: {fit.c3:.6g} mrad/G^3 = {fit.c3*1e9:.4g} mrad/kG^3")
    # contribution of cubic term at max field
    print(f"  cubic term @ max field: {fit.c3*fmax**3:.3f} mrad   linear term: {fit.c1*fmax:.3f} mrad")
