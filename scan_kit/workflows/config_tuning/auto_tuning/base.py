"""Auto-tuning workflow base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import xml.etree.ElementTree as ET

from .params import AutoTuneParamSpec
from .sigma_tune import SigmaTuneResult


@dataclass
class AutoTuneRunResult:
    """Outcome of running an auto-tuning workflow."""

    success: bool
    message: str
    sigma: SigmaTuneResult | None = None
    warnings: list[str] = field(default_factory=list)


class AutoTuneWorkflow(ABC):
    """Built-in configuration auto-tuning workflow."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable workflow identifier."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name in the workflow list."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description shown beside the workflow name."""

    def uses_session_browser(self) -> bool:
        """When true, workflow parameters come from :class:`SessionBrowserWidget`."""
        return False

    def param_specs(self) -> list[AutoTuneParamSpec]:
        return []

    def default_params(self) -> dict[str, Any]:
        return {spec.key: spec.default for spec in self.param_specs()}

    def validate(self, params: dict[str, Any]) -> list[str]:
        return []

    @abstractmethod
    def apply_to_root(
        self,
        root: ET.Element,
        params: dict[str, Any],
    ) -> AutoTuneRunResult:
        """Mutate the devices.xml element tree in memory."""
