"""Default column generator set shared by built-in templates."""

from __future__ import annotations

from .generators.charge import SpotWeightGenerator
from .generators.constant import ConstantColumnGenerator
from .generators.pass_through import FromRowColumnGenerator
from .generators.spot_no import SequentialSpotNoGenerator
from .input_map import DEFAULT_BEAM_SIZE, DEFAULT_CURRENT_A


def standard_column_generators() -> list:
    """Column generators used by Zero Field and Rectangular Field templates."""
    return [
        FromRowColumnGenerator("ENERGY", "energy"),
        ConstantColumnGenerator("CURRENT", DEFAULT_CURRENT_A),
        ConstantColumnGenerator("BEAM_SIZE", DEFAULT_BEAM_SIZE),
        FromRowColumnGenerator("X_POSITION", "x_position"),
        FromRowColumnGenerator("Y_POSITION", "y_position"),
        SpotWeightGenerator(),
        ConstantColumnGenerator("VELOCITY", 0.0),
        SequentialSpotNoGenerator(),
        FromRowColumnGenerator("layer_id", "layer_id"),
    ]
