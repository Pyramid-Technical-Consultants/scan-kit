"""Reference-frame policy shared by spot and timeslice loaders."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..common.timeslice_position_error import (
    frame_timeslice_chamber_position_arrays,
    frame_timeslice_iso_position_arrays,
    load_session_beam_on_chamber_positions,
    load_session_beam_on_iso_positions,
    resolve_session_timeslice_chamber_position_source,
    resolve_session_timeslice_iso_position_source,
)
from .types import REFERENCE_CHAMBER, REFERENCE_ISO, ReferenceFrameKind

TIMESLICE_POSITION_KEYS: tuple[str, str, str, str] = (
    "ic1_x", "ic1_y", "ic2_x", "ic2_y",
)


def spot_positions_raw(reference_frame: ReferenceFrameKind) -> bool:
    """Chamber frame uses raw register columns; isocenter uses processed."""
    return reference_frame == REFERENCE_CHAMBER


def spot_sigma_prefer_raw(reference_frame: ReferenceFrameKind) -> bool | None:
    """Spot sigma column preference for the selected reference frame."""
    if reference_frame == REFERENCE_CHAMBER:
        return True
    if reference_frame == REFERENCE_ISO:
        return False
    return None


def timeslice_position_loader(
    reference_frame: ReferenceFrameKind,
) -> Callable[..., Any]:
    """Return the beam-on position loader for *reference_frame*."""
    if reference_frame == REFERENCE_CHAMBER:
        return load_session_beam_on_chamber_positions
    return load_session_beam_on_iso_positions


def timeslice_position_table_hooks(
    reference_frame: ReferenceFrameKind,
) -> tuple[Callable, Callable, tuple[str, ...]]:
    """prepare/extract/keys for :func:`load_energy_tagged_table`."""
    if reference_frame == REFERENCE_CHAMBER:
        def prepare(src, frames):
            return resolve_session_timeslice_chamber_position_source(src, frames)

        def extract(df, source):
            return frame_timeslice_chamber_position_arrays(df, source)

        return prepare, extract, TIMESLICE_POSITION_KEYS

    def prepare(src, frames):
        return resolve_session_timeslice_iso_position_source(src, frames)

    def extract(df, source):
        return frame_timeslice_iso_position_arrays(df, source)

    return prepare, extract, TIMESLICE_POSITION_KEYS
