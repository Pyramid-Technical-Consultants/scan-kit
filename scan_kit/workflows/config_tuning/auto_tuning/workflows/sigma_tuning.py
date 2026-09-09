"""Sigma tuning: rewrite IC beam_sigma K0 from session measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..session_params import parse_session_ids, validate_session_selection
from ..sigma_tune import (
    normalize_sigma_lower_headroom_percent,
    normalize_sigma_tolerance_percent,
    tune_sigmas_from_sessions,
)


class SigmaTuningWorkflow(AutoTuneWorkflow):
    """Set constant per-band K0 from measured IC spot sigmas in one or more sessions."""

    @property
    def id(self) -> str:
        return "sigma_tuning"

    @property
    def name(self) -> str:
        return "Sigma Tuning"

    @property
    def description(self) -> str:
        return "IC1/IC2 σ K0 to fit ±tolerance band"

    def uses_session_browser(self) -> bool:
        return True

    def validate(self, params: dict[str, Any]) -> list[str]:
        return validate_session_selection(params)

    def apply_to_root(
        self,
        root: ET.Element,
        params: dict[str, Any],
    ) -> AutoTuneRunResult:
        session_ids = parse_session_ids(params)
        data_dir = Path(str(params["data_dir"]).strip()).expanduser().resolve()

        tolerance_percent = normalize_sigma_tolerance_percent(
            params.get("sigma_tolerance_percent")
        )
        lower_headroom_percent = normalize_sigma_lower_headroom_percent(
            params.get("sigma_lower_headroom_percent")
        )
        tune_result = tune_sigmas_from_sessions(
            root,
            session_ids,
            str(data_dir),
            tolerance_percent=tolerance_percent,
            lower_headroom_percent=lower_headroom_percent,
        )
        if not tune_result.ok:
            return AutoTuneRunResult(
                success=False,
                message="No sigma bands were updated.",
                sigma=tune_result,
                warnings=list(tune_result.warnings),
            )

        if len(session_ids) == 1:
            session_label = session_ids[0]
        else:
            session_label = f"{len(session_ids)} sessions"
        msg = (
            f"Updated {tune_result.bands_updated} beam_sigma band(s) from "
            f"{session_label}. Save the configuration to write devices.xml."
        )
        return AutoTuneRunResult(
            success=True,
            message=msg,
            sigma=tune_result,
            warnings=list(tune_result.warnings),
        )
