"""DICOM RT Ion plan import template."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..base import PlanTemplate
from ..dicom_rt_plan import dicom_to_input_map, validate_dicom_plan_path
from ..generators.base import ColumnGenerator, SpotLayoutGenerator
from ..input_map import DEFAULT_BEAM_SIZE
from ..layouts.rectangular_field import FAST_AXIS_CHOICES, FAST_AXIS_X
from ..params import ParamSpec, validate_positive_float
from ..spot_order import (
    SPOT_ORDER_CHOICES,
    SPOT_ORDER_MINIMIZE_TRAVEL,
    SPOT_ORDER_PLAN,
    validate_spot_order_params,
)


class _UnusedLayout(SpotLayoutGenerator):
    @property
    def id(self) -> str:
        return "unused"

    def param_specs(self) -> list[ParamSpec]:
        return []

    def validate(self, params: dict[str, Any]) -> list[str]:
        return []

    def generate_rows(self, params: dict[str, Any]) -> list:
        raise NotImplementedError


class _UnusedColumnGenerator(ColumnGenerator):
    @property
    def column(self) -> str:
        return ""

    def values(self, rows: list, params: dict[str, Any]) -> list[Any]:
        raise NotImplementedError


class DicomRtPlanTemplate(PlanTemplate):
    """Import spot positions, energies, MU, and beam size from an RT Ion DICOM plan."""

    def __init__(self) -> None:
        self._layout = _UnusedLayout()
        self._columns = [_UnusedColumnGenerator()]

    @property
    def id(self) -> str:
        return "dicom_rt_plan"

    @property
    def name(self) -> str:
        return "DICOM RT Plan"

    @property
    def description(self) -> str:
        return "Convert an RT Ion therapy plan (.dcm) into an input map CSV."

    @property
    def layout(self) -> SpotLayoutGenerator:
        return self._layout

    @property
    def column_generators(self) -> list[ColumnGenerator]:
        return self._columns

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                key="dicom_path",
                label="RT Plan File",
                kind="file_path",
                default="",
                field_set="source",
                file_filter="RT Ion DICOM (*.dcm);;All files (*)",
            ),
            ParamSpec(
                key="use_dicom_beam_size",
                label="Use DICOM Scanning Spot Size",
                kind="bool",
                default=True,
                field_set="geometry",
            ),
            ParamSpec(
                key="beam_size_override_mm",
                label="Beam Size",
                kind="float",
                default=DEFAULT_BEAM_SIZE,
                minimum=0.001,
                maximum=500.0,
                decimals=3,
                step=0.1,
                suffix="mm",
                field_set="geometry",
                visible_when={"use_dicom_beam_size": (False,)},
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
        errors = validate_dicom_plan_path(params.get("dicom_path"))
        if not bool(params.get("use_dicom_beam_size", True)):
            errors.extend(
                validate_positive_float(
                    params.get("beam_size_override_mm"),
                    label="Beam Size (mm)",
                )
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
        use_dicom_beam_size = bool(params.get("use_dicom_beam_size", True))
        fallback_beam_size = (
            float(params.get("beam_size_override_mm", DEFAULT_BEAM_SIZE))
            if not use_dicom_beam_size
            else DEFAULT_BEAM_SIZE
        )
        df = dicom_to_input_map(
            params["dicom_path"],
            use_dicom_beam_size=use_dicom_beam_size,
            default_beam_size=fallback_beam_size,
            spot_order=str(params.get("spot_order", SPOT_ORDER_PLAN)),
            fast_axis=str(params.get("fast_axis", FAST_AXIS_X)),
        )
        report(100)
        return df
