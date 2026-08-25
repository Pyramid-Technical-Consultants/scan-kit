"""Shared sample/spot filters for unified analysis views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .session_ic_xy import SessionIcXYData
from .timeslice_position_error import SessionPositionErrors
from .timeslice_sigma import SessionIcSigmas

# Iglewicz-Hoaglin cutoff used by the former Position Error Outliers (Spot) view.
MOD_Z_THRESHOLD = 3.5

FILTER_ALL = "all"
FILTER_BEAM_ON = "beam_on"
FILTER_BEAM_OFF = "beam_off"
FILTER_BEAM_BOTH = "beam_both"
FILTER_LOWER_95 = "lower_95"
FILTER_UPPER_95 = "upper_95"
FILTER_MAD_OUTLIERS = "mad_outliers"

BEAM_STATE_FILTER_IDS = frozenset({FILTER_BEAM_ON, FILTER_BEAM_OFF, FILTER_BEAM_BOTH})

DOMAIN_FILTERS: tuple[tuple[str, str], ...] = (
    (FILTER_ALL, "All Data"),
    (FILTER_LOWER_95, "Within Lower 95%"),
    (FILTER_UPPER_95, "Upper 5% Only"),
    (FILTER_MAD_OUTLIERS, "MAD Outliers"),
)

BEAM_STATE_FILTERS: tuple[tuple[str, str], ...] = (
    (FILTER_BEAM_ON, "Beam On"),
    (FILTER_BEAM_OFF, "Beam Off"),
    (FILTER_BEAM_BOTH, "Beam On + Off"),
)

@dataclass(frozen=True)
class DataFilterSelection:
    """Independent domain and beam-state filters applied together."""

    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH

    def is_identity(self) -> bool:
        return (
            self.domain_filter == FILTER_ALL
            and self.beam_state_filter == FILTER_BEAM_BOTH
        )


def is_beam_state_filter(filter_id: str) -> bool:
    return filter_id in BEAM_STATE_FILTER_IDS


def default_domain_filter() -> str:
    return FILTER_ALL


def default_beam_state_filter(*, has_beam_state: bool) -> str:
    """Default beam filter: beam-on for timeslice payloads, both for spot tables."""
    return FILTER_BEAM_ON if has_beam_state else FILTER_BEAM_BOTH


def default_data_filter_selection(*, has_beam_state: bool) -> DataFilterSelection:
    return DataFilterSelection(
        domain_filter=default_domain_filter(),
        beam_state_filter=default_beam_state_filter(has_beam_state=has_beam_state),
    )


def beam_state_mask(
    beam_on: np.ndarray | None,
    filter_id: str,
    n: int,
) -> np.ndarray:
    """Return a keep-mask for one beam-state filter."""
    if filter_id == FILTER_BEAM_BOTH:
        return np.ones(n, dtype=bool)
    if beam_on is None:
        return np.ones(n, dtype=bool)
    on = np.asarray(beam_on, dtype=bool)
    if len(on) != n:
        return np.ones(n, dtype=bool)
    if filter_id == FILTER_BEAM_ON:
        return on
    if filter_id == FILTER_BEAM_OFF:
        return ~on
    raise ValueError(f"Not a beam-state filter: {filter_id!r}")


def modified_z(values: np.ndarray) -> np.ndarray:
    """Iglewicz-Hoaglin modified z-score: 0.6745*(x - median)/MAD."""
    v = np.asarray(values, dtype=float)
    z = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    if not finite.any():
        return z
    med = np.median(v[finite])
    mad = np.median(np.abs(v[finite] - med))
    if mad > 0:
        z[finite] = 0.6745 * (v[finite] - med) / mad
        return z
    mean = np.mean(v[finite])
    mean_ad = np.mean(np.abs(v[finite] - mean))
    if mean_ad > 0:
        z[finite] = (v[finite] - mean) / (1.253314 * mean_ad)
    else:
        z[finite] = 0.0
    return z


def _valid_severity(severity: np.ndarray) -> np.ndarray:
    return np.isfinite(severity)


def filter_mask_from_severity(severity: np.ndarray, filter_id: str) -> np.ndarray:
    """Return a boolean keep-mask from a per-sample severity scalar."""
    if filter_id == FILTER_ALL:
        return _valid_severity(severity)

    valid = _valid_severity(severity)
    if not np.any(valid):
        return np.zeros_like(severity, dtype=bool)

    s = severity[valid]
    if filter_id == FILTER_LOWER_95:
        cutoff = float(np.percentile(s, 95))
        return valid & (severity <= cutoff)
    if filter_id == FILTER_UPPER_95:
        cutoff = float(np.percentile(s, 95))
        return valid & (severity > cutoff)
    if filter_id == FILTER_MAD_OUTLIERS:
        z = modified_z(severity)
        return valid & (np.abs(z) > MOD_Z_THRESHOLD)
    return valid


def _mad_outlier_mask_four_axis(*arrays: np.ndarray) -> np.ndarray:
    """Flag samples where any axis exceeds the modified-z outlier cutoff."""
    n = len(arrays[0])
    abs_z = np.zeros((len(arrays), n))
    for i, arr in enumerate(arrays):
        z = modified_z(np.asarray(arr, dtype=float))
        abs_z[i] = np.where(np.isfinite(z), np.abs(z), 0.0)
    worst_z = np.max(abs_z, axis=0)
    return worst_z > MOD_Z_THRESHOLD


def _domain_mask_from_columns(
    session: dict,
    column_keys: Sequence[str],
    domain_filter: str,
) -> np.ndarray:
    """Build a keep-mask from severity/domain rules only."""
    n = _session_length(session, column_keys)
    if domain_filter == FILTER_ALL:
        return np.ones(n, dtype=bool)

    arrays = [
        np.asarray(session[key], dtype=float)
        for key in column_keys
        if key in session
    ]
    if not arrays:
        return np.zeros(n, dtype=bool)

    if domain_filter == FILTER_MAD_OUTLIERS and len(arrays) >= 2:
        return (
            _mad_outlier_mask_four_axis(*arrays[:4])
            if len(arrays) >= 4
            else _mad_outlier_mask_four_axis(*arrays)
        )

    severity = np.nanmax(np.abs(np.stack(arrays, axis=0)), axis=0)
    return filter_mask_from_severity(severity, domain_filter)


def filter_mask_from_columns(
    session: dict,
    column_keys: Sequence[str],
    domain_filter: str,
    beam_state_filter: str = FILTER_BEAM_BOTH,
) -> np.ndarray:
    """Build a keep-mask from domain and beam-state filters combined."""
    n = _session_length(session, column_keys)
    beam_on = session.get("beam_on")
    beam_arr = np.asarray(beam_on, dtype=bool) if beam_on is not None else None
    domain_mask = _domain_mask_from_columns(session, column_keys, domain_filter)
    beam_mask = beam_state_mask(beam_arr, beam_state_filter, n)
    return domain_mask & beam_mask


def _coerce_filter_selection(
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> DataFilterSelection:
    if isinstance(selection, DataFilterSelection):
        return selection
    if is_beam_state_filter(selection):
        return DataFilterSelection(
            domain_filter=FILTER_ALL,
            beam_state_filter=selection,
        )
    return DataFilterSelection(
        domain_filter=selection,
        beam_state_filter=beam_state_filter or FILTER_BEAM_BOTH,
    )


def _session_length(session: dict, column_keys: Sequence[str]) -> int:
    for key in ("energy", *column_keys):
        if key in session:
            return len(np.asarray(session[key]))
    return 0


def _mask_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    if len(out) != len(mask):
        return out
    out[~mask] = np.nan
    return out


def filter_session_dict(session: dict, mask: np.ndarray) -> dict:
    """Apply one boolean mask to every array column in a session summary dict."""
    out = dict(session)
    for key, value in session.items():
        if key in ("session_id", "has_ic3"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 1 and len(arr) == len(mask):
            if key == "beam_on":
                out[key] = arr[mask].astype(bool)
            else:
                out[key] = _mask_array(arr, mask)
    return out


def _mask_optional_bool(arr: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=bool)[mask]


def _position_error_mask(
    data: SessionPositionErrors,
    domain_filter: str,
    beam_state_filter: str,
) -> np.ndarray:
    n = len(data.ic1_x)
    if domain_filter == FILTER_MAD_OUTLIERS:
        domain_mask = _mad_outlier_mask_four_axis(
            data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y,
        )
    elif domain_filter == FILTER_ALL:
        domain_mask = np.ones(n, dtype=bool)
    else:
        severity = np.nanmax(
            np.abs(np.stack([data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y], axis=0)),
            axis=0,
        )
        domain_mask = filter_mask_from_severity(severity, domain_filter)
    beam_mask = beam_state_mask(data.beam_on, beam_state_filter, n)
    return domain_mask & beam_mask


def filter_session_position_errors(
    data: SessionPositionErrors,
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> SessionPositionErrors:
    filters = _coerce_filter_selection(selection, beam_state_filter)
    mask = _position_error_mask(
        data, filters.domain_filter, filters.beam_state_filter,
    )
    return SessionPositionErrors(
        ic1_x=_mask_array(data.ic1_x, mask),
        ic1_y=_mask_array(data.ic1_y, mask),
        ic2_x=_mask_array(data.ic2_x, mask),
        ic2_y=_mask_array(data.ic2_y, mask),
        beam_on=_mask_optional_bool(data.beam_on, mask),
    )


def filter_session_ic_xy(
    data: SessionIcXYData,
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> SessionIcXYData:
    filters = _coerce_filter_selection(selection, beam_state_filter)
    n = len(data.ic1_x)
    if data.plan_x is not None and data.plan_y is not None:
        dev = np.nanmax(
            np.abs(
                np.stack(
                    [
                        data.ic1_x - data.plan_x,
                        data.ic1_y - data.plan_y,
                        data.ic2_x - data.plan_x,
                        data.ic2_y - data.plan_y,
                    ],
                    axis=0,
                )
            ),
            axis=0,
        )
        if filters.domain_filter == FILTER_MAD_OUTLIERS:
            domain_mask = _mad_outlier_mask_four_axis(
                data.ic1_x - data.plan_x,
                data.ic1_y - data.plan_y,
                data.ic2_x - data.plan_x,
                data.ic2_y - data.plan_y,
            )
        elif filters.domain_filter == FILTER_ALL:
            domain_mask = np.ones(n, dtype=bool)
        else:
            domain_mask = filter_mask_from_severity(dev, filters.domain_filter)
    else:
        if filters.domain_filter == FILTER_MAD_OUTLIERS:
            domain_mask = _mad_outlier_mask_four_axis(
                data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y,
            )
        elif filters.domain_filter == FILTER_ALL:
            domain_mask = np.ones(n, dtype=bool)
        else:
            severity = np.nanmax(
                np.abs(np.stack([data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y], axis=0)),
                axis=0,
            )
            domain_mask = filter_mask_from_severity(severity, filters.domain_filter)
    mask = domain_mask & beam_state_mask(
        data.beam_on, filters.beam_state_filter, n,
    )

    plan_x = _mask_array(data.plan_x, mask) if data.plan_x is not None else None
    plan_y = _mask_array(data.plan_y, mask) if data.plan_y is not None else None
    return SessionIcXYData(
        ic1_x=_mask_array(data.ic1_x, mask),
        ic1_y=_mask_array(data.ic1_y, mask),
        ic2_x=_mask_array(data.ic2_x, mask),
        ic2_y=_mask_array(data.ic2_y, mask),
        plan_x=plan_x,
        plan_y=plan_y,
        beam_on=_mask_optional_bool(data.beam_on, mask),
    )


def filter_session_ic_sigmas(
    data: SessionIcSigmas,
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> SessionIcSigmas:
    filters = _coerce_filter_selection(selection, beam_state_filter)
    n = len(data.ic1_x)
    if filters.domain_filter == FILTER_MAD_OUTLIERS:
        domain_mask = _mad_outlier_mask_four_axis(
            data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y,
        )
    elif filters.domain_filter == FILTER_ALL:
        domain_mask = np.ones(n, dtype=bool)
    else:
        severity = np.nanmax(
            np.stack([data.ic1_x, data.ic1_y, data.ic2_x, data.ic2_y], axis=0),
            axis=0,
        )
        domain_mask = filter_mask_from_severity(severity, filters.domain_filter)
    mask = domain_mask & beam_state_mask(
        data.beam_on, filters.beam_state_filter, n,
    )
    return SessionIcSigmas(
        ic1_x=_mask_array(data.ic1_x, mask),
        ic1_y=_mask_array(data.ic1_y, mask),
        ic2_x=_mask_array(data.ic2_x, mask),
        ic2_y=_mask_array(data.ic2_y, mask),
        beam_on=_mask_optional_bool(data.beam_on, mask),
    )


def filter_distribution_session_data(
    session_data: dict[str, Any],
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> dict[str, Any]:
    """Apply data filters to loaded distribution-explorer session payloads."""
    filters = _coerce_filter_selection(selection, beam_state_filter)
    if filters.is_identity():
        return session_data

    out: dict[str, Any] = {}
    for sid, payload in session_data.items():
        if isinstance(payload, SessionPositionErrors):
            out[sid] = filter_session_position_errors(payload, filters)
        elif isinstance(payload, SessionIcXYData):
            out[sid] = filter_session_ic_xy(payload, filters)
        elif isinstance(payload, SessionIcSigmas):
            out[sid] = filter_session_ic_sigmas(payload, filters)
        else:
            out[sid] = payload
    return out


def filter_binned_session_data(
    session_data: dict[str, dict],
    column_keys: Sequence[str],
    selection: DataFilterSelection | str,
    beam_state_filter: str | None = None,
) -> dict[str, dict]:
    """Apply data filters to binned-summary session tables."""
    filters = _coerce_filter_selection(selection, beam_state_filter)
    if filters.is_identity():
        return session_data

    out: dict[str, dict] = {}
    for sid, session in session_data.items():
        mask = filter_mask_from_columns(
            session,
            column_keys,
            filters.domain_filter,
            filters.beam_state_filter,
        )
        out[sid] = filter_session_dict(session, mask)
    return out
