"""Default column generator set shared by built-in templates."""

from __future__ import annotations

from .generators.charge import SpotWeightGenerator
from .generators.constant import ConstantColumnGenerator
from .generators.empty import EmptyTrailingColumnGenerator
from .generators.pass_through import FromRowColumnGenerator
from .generators.spot_no import SequentialSpotNoGenerator
from .input_map import DEFAULT_BEAM_SIZE


def standard_column_generators() -> list:
    """Column generators used by Zero Field and Rectangular Field templates."""
    return [
        FromRowColumnGenerator("ENERGY", "energy"),
        ConstantColumnGenerator("CURRENT", 0.0),
        ConstantColumnGenerator("BEAM_SIZE_X", DEFAULT_BEAM_SIZE),
        ConstantColumnGenerator("BEAM_SIZE_Y", DEFAULT_BEAM_SIZE),
        FromRowColumnGenerator("X_POSITION", "x_position"),
        FromRowColumnGenerator("Y_POSITION", "y_position"),
        SpotWeightGenerator(),
        ConstantColumnGenerator("VELOCITY", 0.0),
        SequentialSpotNoGenerator(),
        FromRowColumnGenerator("layer_id", "layer_id"),
        ConstantColumnGenerator("beam_off", 1.0),
        ConstantColumnGenerator("map_checksum", 0),
        EmptyTrailingColumnGenerator(),
    ]
