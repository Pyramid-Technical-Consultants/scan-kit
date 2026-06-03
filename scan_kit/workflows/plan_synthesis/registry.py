"""Built-in plan template registry."""

from __future__ import annotations

from .base import PlanTemplate
from .templates.rectangular_field import RectangularFieldTemplate
from .templates.zero_field import ZeroFieldTemplate

TEMPLATE_REGISTRY: list[PlanTemplate] = [
    ZeroFieldTemplate(),
    RectangularFieldTemplate(),
]

_TEMPLATES_BY_ID: dict[str, PlanTemplate] = {t.id: t for t in TEMPLATE_REGISTRY}


def get_template(template_id: str) -> PlanTemplate | None:
    return _TEMPLATES_BY_ID.get(template_id)
