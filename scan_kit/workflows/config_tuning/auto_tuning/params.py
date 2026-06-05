"""Parameter specifications for auto-tuning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AutoTuneParamKind = Literal["text", "directory"]


@dataclass(frozen=True)
class AutoTuneParamSpec:
    key: str
    label: str
    kind: AutoTuneParamKind = "text"
    default: str = ""
    placeholder: str = ""
