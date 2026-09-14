"""kMU tuning: scale ion-chamber K_MU so secondaries agree with the primary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..kmu_tune import (
    normalize_kmu_known_mu,
    normalize_kmu_percent,
    normalize_kmu_primary_ic,
    normalize_kmu_primary_mode,
    tune_kmu_from_sessions,
)
from ..session_params import parse_session_ids, validate_session_selection


class KmuTuningWorkflow(AutoTuneWorkflow):
    """Match secondary IC K_MU to the primary; optionally rescale the primary first."""

    @property
    def id(self) -> str:
        return "kmu_tuning"

    @property
    def name(self) -> str:
        return "Dose Calibration"

    @property
    def description(self) -> str:
        return "Secondary IC K_MU to match primary (optional primary rescale)"

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
        primary = normalize_kmu_primary_ic(params.get("primary_ic"))
        mode = normalize_kmu_primary_mode(params.get("primary_mode"))
        known_mu = normalize_kmu_known_mu(params.get("known_mu"))
        percent = normalize_kmu_percent(params.get("percent"))

        tune_result = tune_kmu_from_sessions(
            root,
            session_ids,
            str(data_dir),
            primary=primary,
            mode=mode,
            known_mu=known_mu,
            percent=percent,
        )
        if not tune_result.ok:
            return AutoTuneRunResult(
                success=False,
                message="No K_MU values were updated.",
                kmu=tune_result,
                warnings=list(tune_result.warnings),
            )

        session_label = (
            session_ids[0] if len(session_ids) == 1 else f"{len(session_ids)} sessions"
        )
        msg = (
            f"Updated {tune_result.devices_updated} K_MU value(s) from {session_label}. "
            f"Save the configuration to write devices.xml."
        )
        return AutoTuneRunResult(
            success=True,
            message=msg,
            kmu=tune_result,
            warnings=list(tune_result.warnings),
        )
