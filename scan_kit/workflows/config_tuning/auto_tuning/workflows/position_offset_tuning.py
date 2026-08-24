"""Position offset tuning: rewrite IC zero_offset_at_iso_mm from session measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from scan_kit.common.session_position import normalize_position_data_source
from scan_kit.common.session_source import resolve_session_source

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..position_offset_tune import tune_position_offsets_from_sessions
from ..sigma_tune import normalize_sigma_optimize_mode
from ..session_params import parse_session_ids


class PositionOffsetTuningWorkflow(AutoTuneWorkflow):
    """Set zero_offset_at_iso_mm from measured IC position errors in one or more sessions."""

    @property
    def id(self) -> str:
        return "position_offset_tuning"

    @property
    def name(self) -> str:
        return "Position Offset Tuning"

    @property
    def description(self) -> str:
        return "IC1/IC2 zero offset at iso mm from session(s)"

    def uses_session_browser(self) -> bool:
        return True

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        session_ids = parse_session_ids(params)
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
        session_ids = parse_session_ids(params)
        data_dir = Path(str(params["data_dir"]).strip()).expanduser().resolve()
        data_source = normalize_position_data_source(params.get("data_source"))
        optimize_mode = normalize_sigma_optimize_mode(params.get("optimize_method"))

        tune_result = tune_position_offsets_from_sessions(
            root,
            session_ids,
            str(data_dir),
            data_source=data_source,
            optimize_mode=optimize_mode,
        )
        if not tune_result.ok:
            return AutoTuneRunResult(
                success=False,
                message="No position offsets were updated.",
                position_offset=tune_result,
                warnings=list(tune_result.warnings),
            )

        if len(session_ids) == 1:
            session_label = session_ids[0]
        else:
            session_label = f"{len(session_ids)} sessions"
        msg = (
            f"Updated {tune_result.offsets_updated} zero_offset_at_iso_mm value(s) from "
            f"{session_label}. Save the configuration to write devices.xml."
        )
        return AutoTuneRunResult(
            success=True,
            message=msg,
            position_offset=tune_result,
            warnings=list(tune_result.warnings),
        )
