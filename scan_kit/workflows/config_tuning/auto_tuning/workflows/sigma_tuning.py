"""Sigma tuning: rewrite IC beam_sigma K0 from session measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from scan_kit.common.session_source import resolve_session_source

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..sigma_tune import normalize_sigma_optimize_mode, tune_sigmas_from_sessions


def parse_sigma_session_ids(params: dict[str, Any]) -> list[str]:
    """Return deduplicated session IDs from workflow params."""
    raw = params.get("session_ids")
    if isinstance(raw, list):
        ids = [str(item).strip() for item in raw if str(item).strip()]
    else:
        legacy = str(params.get("session_id", "")).strip()
        ids = [legacy] if legacy else []

    seen: set[str] = set()
    ordered: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    return ordered


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
        return "IC1/IC2 σ K0 from session(s)"

    def uses_session_browser(self) -> bool:
        return True

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        session_ids = parse_sigma_session_ids(params)
        if not session_ids:
            errors.append("Select at least one session.")
        data_dir = str(params.get("data_dir", "")).strip()
        if not data_dir:
            errors.append("Enter the folder containing session data.")
        elif not Path(data_dir).expanduser().is_dir():
            errors.append("Session data folder is not a directory.")
        else:
            base = Path(data_dir).expanduser().resolve()
            missing = [
                sid
                for sid in session_ids
                if resolve_session_source(sid, base) is None
            ]
            if missing and len(missing) == len(session_ids):
                errors.append(
                    f"No selected sessions were found under {base}."
                )
        return errors

    def apply_to_root(
        self,
        root: ET.Element,
        params: dict[str, Any],
    ) -> AutoTuneRunResult:
        session_ids = parse_sigma_session_ids(params)
        data_dir = Path(str(params["data_dir"]).strip()).expanduser().resolve()

        optimize_mode = normalize_sigma_optimize_mode(params.get("optimize_method"))
        tune_result = tune_sigmas_from_sessions(
            root,
            session_ids,
            str(data_dir),
            optimize_mode=optimize_mode,
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
