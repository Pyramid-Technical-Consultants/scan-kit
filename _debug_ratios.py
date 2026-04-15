import logging
logging.basicConfig(level=logging.WARNING)

from scan_kit.views.dose_ratios import run as run_ratios
import matplotlib
matplotlib.use("Agg")

print("Test 1: single G3 session...", end=" ")
try:
    run_ratios(["1565104513"], "test_data")
    print("OK")
except Exception as e:
    import traceback
    traceback.print_exc()

print("Test 2: mixed G2/G3...", end=" ")
try:
    run_ratios(["1565104513", "1325745315"], "test_data")
    print("OK")
except Exception as e:
    import traceback
    traceback.print_exc()

print("Test 3: G2-only sessions...", end=" ")
try:
    run_ratios(["1325745315"], "test_data")
    print("OK")
except Exception as e:
    import traceback
    traceback.print_exc()
