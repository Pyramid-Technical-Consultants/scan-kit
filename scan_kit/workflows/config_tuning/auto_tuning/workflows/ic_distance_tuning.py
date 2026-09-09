"""IC distance tuning: fit source_to_device_distance_mm and zero offset together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

from scan_kit.common.session_position import normalize_position_data_source

from ..base import AutoTuneRunResult, AutoTuneWorkflow
from ..ic_distance_tune import tune_ic_distances_from_sessions
from ..session_params import parse_session_ids, validate_session_selection


class IcDistanceTuningWorkflow(AutoTuneWorkflow):
    """Scale each IC to the plan it was given, assuming delivery at iso is correct."""

    @property
    def id(self) -> str:
        return "ic_distance_tuning"

    @property
    def name(self) -> str:
        return "IC Distance Tuning"

    @property
    def description(self) -> str:
        return "IC1/IC2 source to device distance + zero offset from session(s)"

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
        # Spot data is the only source that carries the plan position a slope needs, so
        # the panel offers no source choice here and this falls back to the default.
        data_source = normalize_position_data_source(params.get("data_source"))

        tune_result = tune_ic_distances_from_sessions(
            root,
            session_ids,
            str(data_dir),
            data_source=data_source,
        )
        if not tune_result.ok:
            return AutoTuneRunResult(
                success=False,
                message="No IC distances were updated.",
                ic_distance=tune_result,
                warnings=list(tune_result.warnings),
            )

        session_label = (
            session_ids[0] if len(session_ids) == 1 else f"{len(session_ids)} sessions"
        )
        worst = max(tune_result.rows, key=lambda row: abs(row.delta_sdd_percent))
        best = max(tune_result.rows, key=lambda row: row.systematic_removed_mm)
        msg = (
            f"Updated {tune_result.distances_updated} IC distance(s) and zero offset(s) "
            f"from {session_label}; largest distance change {worst.delta_sdd_mm:+.2f} mm "
            f"({worst.delta_sdd_percent:+.2f}%) on {worst.device}, removing up to "
            f"{best.systematic_removed_mm:.3f} mm of position error at the field edge "
            f"({best.device}). Save the configuration to write devices.xml."
        )
        return AutoTuneRunResult(
            success=True,
            message=msg,
            ic_distance=tune_result,
            warnings=list(tune_result.warnings),
        )
