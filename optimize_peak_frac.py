"""Sweep _PEAK_FRAC to find the value that minimises IC current ratios.

Usage:
    python optimize_peak_frac.py <session_id> [session_id ...]

Example:
    python optimize_peak_frac.py 888363373 510431764
"""

import math
import sys

import numpy as np

from scan_kit.common import (
    C_IC1_CURRENT, C_IC2_CURRENT,
    C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D,
    C_ENERGY, resolve_concept_column,
)
from scan_kit.common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

IC_COLS = {
    "ic1": [C_IC1_CURRENT],
    "ic2": [C_IC2_CURRENT],
    "ic3": [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D],
}
MS_S = 1e-3
NOISE_FLOOR = 5.0


def load_session_frames(session_id: str, base_dir: str = "test_data"):
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None
    frames = load_session_timeslice_device_units(src)
    if not frames:
        return None
    result = []
    for df in frames:
        ic_sums = {}
        for ic in ["ic1", "ic2", "ic3"]:
            cols = [c for c in IC_COLS[ic] if c in df.columns]
            if not cols:
                continue
            s = df[cols].sum(axis=1).to_numpy(dtype=np.float64, na_value=0.0)
            s[~np.isfinite(s)] = 0.0
            ic_sums[ic] = s
        if "ic1" in ic_sums:
            result.append(ic_sums)
    return result


def evaluate(all_sessions, peak_frac):
    """Return dict of ratio_key -> (mean, std, min_samples) across all sessions."""
    ratios = {"ic21": [], "ic31": [], "ic32": []}
    min_n = 999_999

    for frames in all_sessions:
        for ic_sums in frames:
            charges, nsamp = {}, {}
            for ic in ["ic1", "ic2", "ic3"]:
                if ic not in ic_sums:
                    continue
                s = ic_sums[ic]
                pk = float(np.nanpercentile(s, 99))
                thresh = max(NOISE_FLOOR, peak_frac * pk)
                v = s[s > thresh]
                charges[ic] = math.fsum(v) * MS_S
                nsamp[ic] = len(v)

            if "ic1" not in charges or charges["ic1"] <= 0:
                continue
            min_n = min(min_n, *nsamp.values())

            if "ic2" in charges and charges["ic2"] > 0:
                ratios["ic21"].append((charges["ic2"] / charges["ic1"] - 1) * 100)
            if "ic3" in charges and charges["ic3"] > 0:
                ratios["ic31"].append((charges["ic3"] / charges["ic1"] - 1) * 100)
                ratios["ic32"].append((charges["ic3"] / charges["ic2"] - 1) * 100)

    out = {}
    for k, vals in ratios.items():
        if vals:
            a = np.array(vals)
            out[k] = (float(np.nanmean(a)), float(np.nanstd(a)))
    return out, min_n


def score(stats, min_n):
    """Lower is better.  Penalises mean, std, and too-few samples."""
    total = 0.0
    for key, (mean, std) in stats.items():
        total += abs(mean) + std
    if min_n < 50:
        total += 100.0
    return total


def main():
    session_ids = sys.argv[1:] if len(sys.argv) > 1 else ["888363373", "510431764"]

    print(f"Loading sessions: {session_ids}")
    all_sessions = []
    for sid in session_ids:
        frames = load_session_frames(sid)
        if frames:
            all_sessions.append(frames)
            print(f"  {sid}: {len(frames)} layers")
        else:
            print(f"  {sid}: SKIPPED (no data)")
    if not all_sessions:
        print("No data loaded.")
        return

    # Coarse sweep
    print(f"\n{'PF':>6} | {'IC2/IC1':>18} | {'IC3/IC1':>18} | {'IC3/IC2':>18} | {'minN':>6} | {'score':>7}")
    print("-" * 95)

    best_score, best_pf = 1e9, 0.3
    pf_values = [round(x * 0.01, 2) for x in range(5, 81)]

    for pf in pf_values:
        stats, mn = evaluate(all_sessions, pf)
        sc = score(stats, mn)
        if sc < best_score:
            best_score = sc
            best_pf = pf

    # Fine sweep around best
    fine_range = [round(best_pf + d * 0.001, 3) for d in range(-20, 21)]
    fine_range = [p for p in fine_range if 0.01 <= p <= 0.95]

    for pf in fine_range:
        stats, mn = evaluate(all_sessions, pf)
        sc = score(stats, mn)
        if sc < best_score:
            best_score = sc
            best_pf = pf

    # Print results around optimum
    display_range = [round(best_pf + d * 0.01, 3) for d in range(-5, 6)]
    display_range = [p for p in display_range if 0.01 <= p <= 0.95]

    for pf in display_range:
        stats, mn = evaluate(all_sessions, pf)
        sc = score(stats, mn)
        parts = []
        for k in ["ic21", "ic31", "ic32"]:
            if k in stats:
                m, s = stats[k]
                parts.append(f"{m:+6.2f}% ± {s:5.2f}%")
            else:
                parts.append("       n/a       ")
        marker = "  <-- BEST" if abs(pf - best_pf) < 0.001 else ""
        print(f" {pf:>5.3f} | {' | '.join(parts)} | {mn:>6} | {sc:>7.2f}{marker}")

    print(f"\n==> Optimal _PEAK_FRAC = {best_pf:.3f}  (score={best_score:.2f})")

    # Show final stats at optimum
    stats, mn = evaluate(all_sessions, best_pf)
    print(f"    Min samples per layer: {mn}")
    for k in ["ic21", "ic31", "ic32"]:
        if k in stats:
            label = {"ic21": "IC2/IC1", "ic31": "IC3/IC1", "ic32": "IC3/IC2"}[k]
            m, s = stats[k]
            print(f"    {label}: {m:+.3f}% ± {s:.3f}%")


if __name__ == "__main__":
    main()
