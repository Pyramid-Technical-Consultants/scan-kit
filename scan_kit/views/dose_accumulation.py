"""Dose accumulation: expected vs measured cumulative dose per IC."""

import logging

import numpy as np
import matplotlib.pyplot as plt

from ..common import (
    C_CHARGE_REQ,
    C_ENERGY,
    C_IC1_TOTAL_DOSE,
    C_IC2_TOTAL_DOSE,
    C_IC3_TOTAL_DOSE,
    C_LAYER_ID,
    ViewSettings,
    apply_auto_calibration,
    apply_calibration_factors,
    resolve_concept_column,
    DEFAULT_SESSION_COLORS,
    SUPTITLE_KW,
    GRID_KW,
)
from ..common.session_source import (
    load_session_csv,
    resolve_session_source,
)

_log = logging.getLogger(__name__)


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
            session_data[sid] = data

    if not session_data:
        _log.debug("No valid dose data found for any session")
        return

    all_ic_keys: set[str] = set()
    for d in session_data.values():
        all_ic_keys.update(d["ic_keys"])
    ic_keys = sorted(all_ic_keys)
    ic_labels = {"ic1": "IC1", "ic2": "IC2", "ic3": "IC3"}

    n_cols = len(ic_keys)
    loaded_ids = list(session_data.keys())
    colors = DEFAULT_SESSION_COLORS[: len(loaded_ids)]

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), squeeze=False)
    fig.suptitle("Dose Accumulation: Expected vs Measured", **SUPTITLE_KW)

    for col_idx, ic in enumerate(ic_keys):
        ax_cum = axes[0, col_idx]
        ax_err = axes[1, col_idx]
        dose_key = f"{ic}_dose"

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

            # Row 0: cumulative dose
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

            # Row 1: cumulative dose error (drift)
            cum_error = cum_measured - cum_expected
            ax_err.plot(
                spot_idx, cum_error,
                color=color, linewidth=1.0, alpha=0.85,
                label=sid if col_idx == 0 else None,
            )

            # Layer boundaries on both rows
            if "layer_id" in data:
                layer_ids = data["layer_id"][valid]
                energies = data["energy"][valid]
                changes = np.where(np.diff(layer_ids.astype(float)) != 0)[0] + 1
                for ci in changes:
                    ax_cum.axvline(ci, color="#333333", linewidth=0.5, alpha=0.4)
                    ax_err.axvline(ci, color="#333333", linewidth=0.5, alpha=0.4)
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
        ax_err.set_xlabel("Spot index")
        if col_idx == 0:
            ax_err.set_ylabel("Cumulative dose error")
        ax_err.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax_err.grid(**GRID_KW)

    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[1, 0].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.25)
    plt.show()
