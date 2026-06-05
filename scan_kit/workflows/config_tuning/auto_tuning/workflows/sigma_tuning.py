"""Sigma tuning: rewrite IC beam_sigma K0 from session measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from scan_kit.common.session_source import resolve_session_source

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..sigma_tune import tune_sigmas_from_session


class SigmaTuningWorkflow(AutoTuneWorkflow):
    """Set constant per-band K0 from measured IC spot sigmas in a session."""

    @property
    def id(self) -> str:
        return "sigma_tuning"

    @property
    def name(self) -> str:
        return "Sigma tuning"

    @property
    def description(self) -> str:
        return "IC1/IC2 σ K0 from session"

    def uses_session_browser(self) -> bool:
        return True

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        sid = str(params.get("session_id", "")).strip()
        if not sid:
            errors.append("Enter a session ID.")
        data_dir = str(params.get("data_dir", "")).strip()
        if not data_dir:
            errors.append("Enter the folder containing session data.")
        elif not Path(data_dir).expanduser().is_dir():
            errors.append("Session data folder is not a directory.")
        return errors

    def apply_to_root(
        self,
        root: ET.Element,
        params: dict[str, Any],
    ) -> AutoTuneRunResult:
        session_id = str(params["session_id"]).strip()
        data_dir = Path(str(params["data_dir"]).strip()).expanduser().resolve()

        if resolve_session_source(session_id, data_dir) is None:
            return AutoTuneRunResult(
                success=False,
                message=f"Session {session_id!r} was not found under {data_dir}.",
            )

        tune_result = tune_sigmas_from_session(root, session_id, str(data_dir))
        if not tune_result.ok:
            return AutoTuneRunResult(
                success=False,
                message="No sigma bands were updated.",
                sigma=tune_result,
                warnings=list(tune_result.warnings),
            )

        msg = (
            f"Updated {tune_result.bands_updated} beam_sigma band(s) from session "
            f"{session_id}. Save the configuration to write devices.xml."
        )
        return AutoTuneRunResult(
            success=True,
            message=msg,
            sigma=tune_result,
            warnings=list(tune_result.warnings),
        )
