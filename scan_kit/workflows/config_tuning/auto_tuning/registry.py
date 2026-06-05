"""Built-in auto-tuning workflow registry."""

from __future__ import annotations

from .base import AutoTuneWorkflow
from .workflows.sigma_tuning import SigmaTuningWorkflow

AUTO_TUNE_REGISTRY: list[AutoTuneWorkflow] = [
    SigmaTuningWorkflow(),
]

_WORKFLOWS_BY_ID: dict[str, AutoTuneWorkflow] = {w.id: w for w in AUTO_TUNE_REGISTRY}


def get_auto_tune_workflow(workflow_id: str) -> AutoTuneWorkflow | None:
    return _WORKFLOWS_BY_ID.get(workflow_id)
