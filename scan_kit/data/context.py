"""Session load context and options for the data-source registry."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.settings import ViewSettings
from .types import (
    GRANULARITY_SPOT,
    GranularityKind,
    REFERENCE_ISO,
    ReferenceFrameKind,
)


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    base_dir: str
    settings: ViewSettings | None = None

    @property
    def bg_subtract(self) -> bool:
        return bool(self.settings.bg_subtract) if self.settings else False


@dataclass(frozen=True)
class LoadOptions:
    granularity: GranularityKind = GRANULARITY_SPOT
    reference_frame: ReferenceFrameKind = REFERENCE_ISO
    bg_subtract: bool | None = None

    def resolved_bg_subtract(self, ctx: SessionContext) -> bool:
        if self.bg_subtract is not None:
            return self.bg_subtract
        return ctx.bg_subtract
