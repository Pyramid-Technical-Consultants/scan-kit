"""IC current column resolution for timeslice frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import (
    C_IC1_CURRENT,
    C_IC2_CURRENT,
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
    resolve_concept_column,
)

_IC3_CONCEPTS = (
    C_IC3_CURRENT_A,
    C_IC3_CURRENT_B,
    C_IC3_CURRENT_C,
    C_IC3_CURRENT_D,
)


@dataclass(frozen=True)
class IcCurrentColumns:
    ic1: str
    ic2: str
    ic3_parts: tuple[str, ...]


def resolve_ic_current_columns(columns) -> IcCurrentColumns | None:
    """Resolve IC1/IC2/IC3 current column names present in *columns*."""
    ic1 = resolve_concept_column(columns, C_IC1_CURRENT)
    ic2 = resolve_concept_column(columns, C_IC2_CURRENT)
    if not ic1 or not ic2:
        return None
    ic3_parts = tuple(
        col
        for concept in _IC3_CONCEPTS
        if (col := resolve_concept_column(columns, concept)) is not None
    )
    if ic3_parts and len(ic3_parts) != len(_IC3_CONCEPTS):
        ic3_parts = ()
    return IcCurrentColumns(ic1, ic2, ic3_parts)


def sum_ic3_current(df: pd.DataFrame, ic3_parts: tuple[str, ...]) -> np.ndarray:
    return sum(df[c].to_numpy(dtype=float) for c in ic3_parts)  # type: ignore[misc]
