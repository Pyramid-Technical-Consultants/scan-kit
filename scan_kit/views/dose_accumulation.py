"""Dose accumulation: expected vs measured cumulative dose per IC."""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..common import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_LAYER_ID,
    C_SPOT_NO,
    ViewSettings,
    apply_auto_calibration,
    apply_calibration_factors,
    resolve_concept_column,
    subtract_background_frames,
    DEFAULT_SESSION_COLORS,
    set_view_header,
    apply_tight_layout,
    GRID_KW,
)
from ..common.session_source import (
    load_session_csv,
    load_session_timeslice_device_units,
    resolve_session_source,
)

_log = logging.getLogger(__name__)

_IC_CURRENT_COLS = {
    "ic1": [C_IC1_CURRENT],
    "ic2": [C_IC2_CURRENT],
    "ic3": [C_IC3_CURRENT_A, C_IC3_CURRENT_B, C_IC3_CURRENT_C, C_IC3_CURRENT_D],
}


def _load_timeslice_current_sums(
    session_id: str, base_dir: str, ic_keys: list[str], n_spots: int,
    *, bg_subtract: bool = False,
) -> dict[str, np.ndarray]:
    """Sum raw IC current per spot from timeslice data.

    Returns ``{"ic1_current_sum": array, ...}`` with one value per spot,
    ordered to match the spot_data / input_map row order (layer-by-layer,
    spot-by-spot within each layer).
    """
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return {}
    frames = load_session_timeslice_device_units(src)
    if not frames:
        return {}
    if bg_subtract:
        subtract_background_frames(frames)

    per_ic_lists: dict[str, list[np.ndarray]] = {ic: [] for ic in ic_keys}

    for layer_df in frames:
        if C_SPOT_NO not in layer_df.columns:
            continue
        spot_no = layer_df[C_SPOT_NO]
        for ic in ic_keys:
            cur_cols = [c for c in _IC_CURRENT_COLS.get(ic, []) if c in layer_df.columns]
            if not cur_cols:
                continue
            cur_total = layer_df[cur_cols].sum(axis=1)
            grouped = cur_total.groupby(spot_no, sort=False)
            spot_ids_ordered = spot_no.drop_duplicates()
            sums = grouped.sum().reindex(spot_ids_ordered).values.astype(float)
            per_ic_lists[ic].append(sums)

    result: dict[str, np.ndarray] = {}
    for ic in ic_keys:
        parts = per_ic_lists[ic]
        if not parts:
            continue
        arr = np.concatenate(parts)
        if len(arr) > n_spots:
            arr = arr[:n_spots]
        elif len(arr) < n_spots:
            arr = np.pad(arr, (0, n_spots - len(arr)), constant_values=np.nan)
        result[f"{ic}_current_sum"] = arr
    return result


def _load_dose_data(session_id: str, base_dir: str) -> dict | None:
    """Load per-spot expected and measured dose for each IC."""
    src = resolve_session_source(session_id, base_dir)
    if src is None:
        return None

    input_map = load_session_csv(src, "input_map.csv")
    spot_data = load_session_csv(src, "spot_data.csv")
    if input_map is None or spot_data is None:
        return None

    col_charge = resolve_concept_column(input_map.columns, C_CHARGE_REQ)
    col_energy = resolve_concept_column(input_map.columns, C_ENERGY)
    col_layer_im = resolve_concept_column(input_map.columns, C_LAYER_ID)
    if col_charge is None or col_energy is None:
        _log.debug("Session %s: missing CHARGE_REQ or ENERGY in input_map", session_id)
        return None

    col_ic1 = resolve_concept_column(spot_data.columns, C_IC1_TOTAL_DOSE)
    col_ic2 = resolve_concept_column(spot_data.columns, C_IC2_TOTAL_DOSE)
    col_ic3 = resolve_concept_column(spot_data.columns, C_IC3_TOTAL_DOSE)

    if col_ic1 is None and col_ic2 is None:
        _log.debug("Session %s: no IC dose columns in spot_data", session_id)
        return None

    n = min(len(input_map), len(spot_data))
    charge_req = input_map[col_charge].values[:n].astype(float)
    energy = input_map[col_energy].values[:n].astype(float)

    layer_id = None
    if col_layer_im is not None:
        layer_id = input_map[col_layer_im].values[:n]

    result: dict = {
        "charge_req": charge_req,
        "energy": energy,
        "n": n,
    }
    if layer_id is not None:
        result["layer_id"] = layer_id

    ic_keys = []
    if col_ic1 is not None:
        result["ic1_dose"] = spot_data[col_ic1].values[:n].astype(float)
        ic_keys.append("ic1")
    if col_ic2 is not None:
        result["ic2_dose"] = spot_data[col_ic2].values[:n].astype(float)
        ic_keys.append("ic2")
    if col_ic3 is not None:
        result["ic3_dose"] = spot_data[col_ic3].values[:n].astype(float)
        ic_keys.append("ic3")

    result["ic_keys"] = ic_keys
    return result


def run(session_ids: list[str], base_dir: str = "test_data", *, settings=None) -> None:
    """Plot expected vs measured cumulative dose for IC1, IC2, IC3."""
    if not session_ids:
        return

    session_data: dict[str, dict] = {}
    for sid in session_ids:
        data = _load_dose_data(sid, base_dir)
        if data is not None:
            if settings and settings.auto_calibrate:
                dose_cols = [f"{ic}_dose" for ic in data["ic_keys"]]
                if settings.cal_factors:
                    _canonical = {"ic1": C_IC1_TOTAL_DOSE, "ic2": C_IC2_TOTAL_DOSE, "ic3": C_IC3_TOTAL_DOSE}
                    mapped = {f"{ic}_dose": settings.cal_factors[_canonical[ic]]
                              for ic in data["ic_keys"] if _canonical.get(ic) in (settings.cal_factors or {})}
                    data = apply_calibration_factors(data, dose_cols, mapped)
                else:
                    data = apply_auto_calibration(data, "charge_req", dose_cols)
            bg = settings.bg_subtract if settings else False
            ts = _load_timeslice_current_sums(sid, base_dir, data["ic_keys"], data["n"],
                                              bg_subtract=bg)
            data.update(ts)
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid dose data found for any session")
        return

    all_ic_keys: set[str] = set()
    for d in session_data.values():
        all_ic_keys.update(d["ic_keys"])
    ic_keys = sorted(all_ic_keys)
    ic_labels = {"ic1": "IC1", "ic2": "IC2", "ic3": "IC3"}

    has_current = any(
        f"{ic}_current_sum" in d for d in session_data.values() for ic in ic_keys
    )
    n_rows = 3 if has_current else 2
    n_cols = len(ic_keys)
    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False,
    )

    for col_idx, ic in enumerate(ic_keys):
        ax_cum = axes[0, col_idx]
        ax_err = axes[1, col_idx]
        ax_raw = axes[2, col_idx] if has_current else None
        dose_key = f"{ic}_dose"
        cur_key = f"{ic}_current_sum"

        for si, (sid, data) in enumerate(session_data.items()):
            if dose_key not in data:
                continue

            charge_req = data["charge_req"]
            measured = data[dose_key]

            valid = np.isfinite(charge_req) & np.isfinite(measured)
            charge_req = charge_req[valid]
            measured = measured[valid]

            cum_expected = np.cumsum(charge_req)
            cum_measured = np.cumsum(measured)
            spot_idx = np.arange(1, len(cum_expected) + 1)

            color = colors[si]

            ax_cum.plot(
                spot_idx, cum_expected,
                color=color, linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"{sid} expected" if col_idx == 0 else None,
            )
            ax_cum.plot(
                spot_idx, cum_measured,
                color=color, linewidth=1.0, alpha=0.9,
                label=f"{sid} measured" if col_idx == 0 else None,
            )

            cum_error = cum_measured - cum_expected
            ax_err.plot(
                spot_idx, cum_error,
                color=color, linewidth=1.0, alpha=0.85,
                label=sid if col_idx == 0 else None,
            )

            if ax_raw is not None and cur_key in data:
                raw_cur = data[cur_key][valid]
                ok = np.isfinite(raw_cur)
                if ok.any():
                    cum_raw = np.cumsum(np.where(ok, raw_cur, 0.0))
                    total_raw = cum_raw[-1] if cum_raw[-1] != 0 else 1.0
                    total_dose = cum_measured[-1] if cum_measured[-1] != 0 else 1.0
                    scale = total_dose / total_raw
                    ax_raw.plot(
                        spot_idx, cum_measured,
                        color=color, linewidth=1.0, alpha=0.5,
                        label=f"{sid} spot dose" if col_idx == 0 else None,
                    )
                    ax_raw.plot(
                        spot_idx, cum_raw * scale,
                        color=color, linewidth=1.0, linestyle="--", alpha=0.9,
                        label=f"{sid} Σ current (scaled)" if col_idx == 0 else None,
                    )

            all_axes = [ax_cum, ax_err] + ([ax_raw] if ax_raw is not None else [])
            if "layer_id" in data:
                layer_ids = data["layer_id"][valid]
                energies = data["energy"][valid]
                changes = np.where(np.diff(layer_ids.astype(float)) != 0)[0] + 1
                for ci in changes:
                    for ax in all_axes:
                        ax.axvline(ci, color="#333333", linewidth=0.5, alpha=0.4)
                    if col_idx == 0:
                        ax_cum.text(
                            ci, 1.0, f" {energies[ci]:g} MeV",
                            transform=ax_cum.get_xaxis_transform(),
                            fontsize=6, va="top", ha="left",
                            color="#333333", alpha=0.6, rotation=90,
                        )

        label = ic_labels.get(ic, ic)
        ax_cum.set_title(label)
        if col_idx == 0:
            ax_cum.set_ylabel("Cumulative dose")
        ax_cum.grid(**GRID_KW)

        ax_err.set_title(f"{label} — Cumulative Dose Error")
        if not has_current:
            ax_err.set_xlabel("Spot index")
        if col_idx == 0:
            ax_err.set_ylabel("Cumulative dose error")
        ax_err.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax_err.grid(**GRID_KW)

        if ax_raw is not None:
            ax_raw.set_title(f"{label} — Spot Dose vs Raw Σ Current")
            ax_raw.set_xlabel("Spot index")
            if col_idx == 0:
                ax_raw.set_ylabel("Cumulative dose")
            ax_raw.grid(**GRID_KW)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    if has_current:
        axes[2, 0].legend(loc="upper left", fontsize=8)

    set_view_header(
        fig,
        "Dose Accumulation: Expected vs Measured",
        loaded_ids,
        colors,
        base_dir=base_dir,
    )

    apply_tight_layout()
    plt.show()
