"""Try running the replay view to reproduce errors."""
import sys
sys.path.insert(0, ".")
from scan_kit.views.ic_timeslice_replay import run
run(["1022244633"], "test_data")
