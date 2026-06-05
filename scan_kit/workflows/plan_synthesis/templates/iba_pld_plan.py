"""IBA PLD plan import template."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..base import PlanTemplate
from ..generators.base import ColumnGenerator, SpotLayoutGenerator
from ..iba_pld_plan import pld_to_input_map, validate_pld_plan_path
from ..input_map import DEFAULT_BEAM_SIZE
from ..layouts.rectangular_field import FAST_AXIS_CHOICES, FAST_AXIS_X
from ..params import ParamSpec, validate_positive_float
from ..spot_order import (
    SPOT_ORDER_CHOICES,
    SPOT_ORDER_MINIMIZE_TRAVEL,
    SPOT_ORDER_PLAN,
    validate_spot_order_params,
)
from .dicom_rt_plan import _UnusedColumnGenerator, _UnusedLayout


class IbaPldPlanTemplate(PlanTemplate):
    """Import spot positions, energies, and MU from an IBA PBS PLD plan."""

    def __init__(self) -> None:
        self._layout = _UnusedLayout()
        self._columns = [_UnusedColumnGenerator()]

    @property
    def id(self) -> str:
        return "iba_pld_plan"

    @property
    def name(self) -> str:
        return "IBA PLD Plan"

    @property
    def description(self) -> str:
        return "Convert an IBA PBS plan (.pld) into an input map CSV."

    @property
    def layout(self) -> SpotLayoutGenerator:
        return self._layout

    @property
    def column_generators(self) -> list[ColumnGenerator]:
        return self._columns

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                key="pld_path",
                label="PLD Plan File",
                kind="file_path",
                default="",
                field_set="source",
                file_filter="IBA PLD (*.pld);;All files (*)",
            ),
            ParamSpec(
                key="beam_size_mm",
                label="Beam Size",
                kind="float",
                default=DEFAULT_BEAM_SIZE,
                minimum=0.001,
                maximum=500.0,
                decimals=3,
                step=0.1,
                suffix="mm",
                field_set="geometry",
            ),
            ParamSpec(
                key="spot_order",
                label="Spot Order",
                kind="button_group",
                default=SPOT_ORDER_PLAN,
                choices=SPOT_ORDER_CHOICES,
                field_set="geometry",
            ),
            ParamSpec(
                key="fast_axis",
                label="Fast Axis",
                kind="button_group",
                default=FAST_AXIS_X,
                choices=FAST_AXIS_CHOICES,
                field_set="geometry",
                visible_when={"spot_order": (SPOT_ORDER_MINIMIZE_TRAVEL,)},
            ),
        ]

    def validate(self, params: dict[str, Any]) -> list[str]:
        errors = validate_pld_plan_path(params.get("pld_path"))
        errors.extend(
            validate_positive_float(params.get("beam_size_mm"), label="Beam Size (mm)")
        )
        errors.extend(
            validate_spot_order_params(
                params.get("spot_order"),
                fast_axis=params.get("fast_axis"),
            )
        )
        return errors

    def generate(
        self,
        params: dict[str, Any],
        *,
        progress: Callable[[int], None] | None = None,
    ) -> pd.DataFrame:
        def report(value: int) -> None:
            if progress is not None:
                progress(max(0, min(100, value)))

        report(0)
        df = pld_to_input_map(
            params["pld_path"],
            default_beam_size=float(params.get("beam_size_mm", DEFAULT_BEAM_SIZE)),
            spot_order=str(params.get("spot_order", SPOT_ORDER_PLAN)),
            fast_axis=str(params.get("fast_axis", FAST_AXIS_X)),
        )
        report(100)
        return df
