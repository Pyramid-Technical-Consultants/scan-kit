import matplotlib
matplotlib.use("Agg")
from scan_kit.views.dose_error_vs_target import run
import traceback

print("Test 1: single G3 session...", end=" ")
try:
    run(["1565104513"], "test_data")
    print("OK")
except Exception:
    traceback.print_exc()

print("Test 2: mixed G2/G3...", end=" ")
try:
    run(["1565104513", "1325745315"], "test_data")
    print("OK")
except Exception:
    traceback.print_exc()

print("Test 3: G2-only...", end=" ")
try:
    run(["1325745315"], "test_data")
    print("OK")
except Exception:
    traceback.print_exc()
