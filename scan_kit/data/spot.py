"""Shared spot-table column helpers."""

from __future__ import annotations

from ..common.schema import (
    C_IC1_X_POS,
    C_IC1_Y_POS,
    C_IC2_X_POS,
    C_IC2_Y_POS,
    POSITION_KEY_G2,
    POSITION_KEY_G3,
    resolve_concept_column,
)


def spot_ic_position_columns_available(
    spot_cols: list[str],
    position_key: str,
) -> bool:
    for concept in (C_IC1_X_POS, C_IC1_Y_POS, C_IC2_X_POS, C_IC2_Y_POS):
        if resolve_concept_column(spot_cols, concept, position_key=position_key) is None:
            return False
    return True


def spot_has_ic_positions(spot_cols: list[str]) -> bool:
    return any(
        spot_ic_position_columns_available(spot_cols, position_key)
        for position_key in (POSITION_KEY_G3, POSITION_KEY_G2)
    )
