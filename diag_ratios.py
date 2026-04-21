"""Inspect filter edge behavior vs raw plateau values."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scan_kit.views.current_ratios import _load_current_ratios

d = _load_current_ratios("1653426060", "test_data")

e = np.array(d["energy"], dtype=float)
sort = np.argsort(e)
e_s = e[sort]

print(f"{'E':>7s} {'IC1 raw':>10s} {'IC1 filt':>10s} {'IC3 raw':>10s} {'IC3 filt':>10s}")
print("=" * 55)

ic1_raw = np.array(d["ic1_disp"], dtype=float)[sort]
ic1_filt = np.array(d["ic1_disp_filt"], dtype=float)[sort]
ic3_raw = np.array(d["ic3_disp"], dtype=float)[sort]
ic3_filt = np.array(d["ic3_disp_filt"], dtype=float)[sort]

# Show first 10, middle 5, last 10
idxs = list(range(10)) + list(range(len(e_s)//2 - 2, len(e_s)//2 + 3)) + list(range(len(e_s) - 10, len(e_s)))
prev = -1
for i in idxs:
    if i - prev > 1:
        print("  ...")
    print(f"  {e_s[i]:5.1f}  {ic1_raw[i]:10.2f}  {ic1_filt[i]:10.2f}  {ic3_raw[i]:10.2f}  {ic3_filt[i]:10.2f}")
    prev = i
