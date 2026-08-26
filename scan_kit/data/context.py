"""Session load context and options for the data-source registry."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.settings import ViewSettings
from .types import (
    DATA_SOURCE_SPOT_ISO,
    DataSourceKind,
    GranularityKind,
    ReferenceFrameKind,
    data_source_granularity,
    data_source_reference_frame,
)


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    base_dir: str
    settings: ViewSettings | None = None

    @property
    def bg_subtract(self) -> bool:
        return bool(self.settings.bg_subtract) if self.settings else False

    def settings_cache_key(self) -> tuple:
        """Hashable fingerprint for load-cache keys (calibration affects spot loads)."""
        if self.settings is None:
            return (False, "off", None)
        cal = self.settings.cal_factors
        cal_key = tuple(sorted(cal.items())) if cal else None
        return (
            bool(self.settings.bg_subtract),
            self.settings.calibration_mode,
            cal_key,
        )


@dataclass(frozen=True)
class LoadOptions:
    data_source: DataSourceKind = DATA_SOURCE_SPOT_ISO
    bg_subtract: bool | None = None
    _granularity: GranularityKind | None = None

    @property
    def granularity(self) -> GranularityKind:
        if self._granularity is not None:
            return self._granularity
        return data_source_granularity(self.data_source)

    @property
    def reference_frame(self) -> ReferenceFrameKind:
        return data_source_reference_frame(self.data_source)

    def resolved_bg_subtract(self, ctx: SessionContext) -> bool:
        if self.bg_subtract is not None:
            return self.bg_subtract
        return ctx.bg_subtract

    def with_granularity(self, granularity: GranularityKind) -> LoadOptions:
        return LoadOptions(
            data_source=self.data_source,
            bg_subtract=self.bg_subtract,
            _granularity=granularity,
        )
