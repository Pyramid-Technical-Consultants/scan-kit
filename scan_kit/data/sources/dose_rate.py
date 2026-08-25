"""Layer-level MU delivery rate vs energy (Binned Summary)."""

from __future__ import annotations

from ...common.mu_delivery_rate import (
    load_session_mu_delivery_rates,
    probe_session_mu_delivery_rates,
)
from ..context import LoadOptions, SessionContext
from ..registry import DataSourceSpec, register
from ..types import GRANULARITY_LAYER, REFERENCE_ISO

SOURCE_DOSE_RATE = "dose_rate"


def probe_dose_rate(ctx: SessionContext, opts: LoadOptions) -> bool:
    if opts.granularity != GRANULARITY_LAYER:
        return False
    return probe_session_mu_delivery_rates(ctx.session_id, ctx.base_dir)


def load_dose_rate(ctx: SessionContext, opts: LoadOptions) -> dict | None:
    if opts.granularity != GRANULARITY_LAYER:
        return None
    rates = load_session_mu_delivery_rates(ctx.session_id, ctx.base_dir)
    if rates is None:
        return None
    return {
        "session_id": ctx.session_id,
        "energy": rates["energy"],
        "mu_rate": rates["mu_rate"],
        "session_avg_rate": rates["session_avg_rate"],
    }


SPEC = register(
    DataSourceSpec(
        id=SOURCE_DOSE_RATE,
        label="Dose Rate",
        granularities=frozenset({GRANULARITY_LAYER}),
        reference_frames=frozenset({REFERENCE_ISO}),
        supports_bg_subtract=False,
        supports_beam_filter=False,
        probe=probe_dose_rate,
        load=load_dose_rate,
    )
)
