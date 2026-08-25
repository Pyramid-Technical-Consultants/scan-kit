"""Shared timeslice channel catalog for FFT Explorer and Timeslice Replay.

Channel keys and availability rules live here; each view adds presentation
metadata (colors, PSD units, presets) via thin adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SOURCE_IC_CURRENT = "ic_current"
SOURCE_SIGMA = "sigma"

FAMILY_IC = "IC Current"
FAMILY_DDOSE = "dDose/dt"
FAMILY_SIGMA = "Sigma"
FAMILY_FIELD = "Magnetic Field"
FAMILY_AMPLIFIER = "Amplifier"
FAMILY_BEAM = "Beam Current"
FAMILY_POSITION = "Chamber Position"
FAMILY_PEAK = "Peak Amplitude (G3)"

TimelineBundle = dict[str, Any]
ChannelExtract = Callable[[TimelineBundle], np.ndarray | None]


def _array_key(key: str) -> ChannelExtract:
    def extract(data: TimelineBundle) -> np.ndarray | None:
        arr = data.get(key)
        if arr is None:
            return None
        values = np.asarray(arr)
        if values.size == 0:
            return None
        return values

    return extract


def _b_mag_extract(data: TimelineBundle) -> np.ndarray | None:
    bx = _array_key("bx")(data)
    by = _array_key("by")(data)
    if bx is None or by is None:
        stored = data.get("b_mag")
        if stored is None:
            return None
        values = np.asarray(stored)
        return values if values.size else None
    return np.hypot(bx, by)


@dataclass(frozen=True)
class TimelineChannelSpec:
    key: str
    label: str
    family: str
    source_id: str | None = None
    requires_ic3: bool = False
    beam_off_quiet_threshold: float | None = None
    psd_unit: str | None = None
    replay_color: str | None = None
    replay_linewidth: float = 0.5
    beam_off_edges: bool = False
    replay_visible: bool = True
    fft_visible: bool = True
    extract: ChannelExtract | None = None

    def resolved_extract(self) -> ChannelExtract:
        if self.extract is not None:
            return self.extract
        return _array_key(self.key)


def _spec(
    key: str,
    label: str,
    family: str,
    *,
    source_id: str | None = None,
    requires_ic3: bool = False,
    beam_off_quiet_threshold: float | None = None,
    psd_unit: str | None = None,
    replay_color: str | None = None,
    replay_linewidth: float = 0.5,
    beam_off_edges: bool = False,
    replay_visible: bool = True,
    fft_visible: bool = True,
    extract: ChannelExtract | None = None,
) -> TimelineChannelSpec:
    return TimelineChannelSpec(
        key=key,
        label=label,
        family=family,
        extract=extract,
        source_id=source_id,
        requires_ic3=requires_ic3,
        beam_off_quiet_threshold=beam_off_quiet_threshold,
        psd_unit=psd_unit,
        replay_color=replay_color,
        replay_linewidth=replay_linewidth,
        beam_off_edges=beam_off_edges,
        replay_visible=replay_visible,
        fft_visible=fft_visible,
    )


TIMELINE_CHANNEL_SPECS: tuple[TimelineChannelSpec, ...] = (
    _spec(
        "ic1", "IC1", FAMILY_IC,
        source_id=SOURCE_IC_CURRENT,
        psd_unit="nA",
        replay_color="#1f77b4",
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic2", "IC2", FAMILY_IC,
        source_id=SOURCE_IC_CURRENT,
        psd_unit="nA",
        replay_color="#d62728",
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic3", "IC3 (A+B+C+D)", FAMILY_IC,
        source_id=SOURCE_IC_CURRENT,
        requires_ic3=True,
        psd_unit="nA",
        replay_color="#2ca02c",
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic1_ddose", "IC1 dDose/dt", FAMILY_DDOSE,
        psd_unit="nA",
        replay_color="#1f77b4",
        replay_linewidth=0.6,
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic2_ddose", "IC2 dDose/dt", FAMILY_DDOSE,
        psd_unit="nA",
        replay_color="#d62728",
        replay_linewidth=0.6,
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic3_ddose", "IC3 dDose/dt", FAMILY_DDOSE,
        requires_ic3=True,
        psd_unit="nA",
        replay_color="#2ca02c",
        replay_linewidth=0.6,
        beam_off_edges=True,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "sigma_ic1_x", "IC1 σx (mm)", FAMILY_SIGMA,
        source_id=SOURCE_SIGMA,
        psd_unit="mm",
        replay_color="#1f77b4",
        beam_off_edges=True,
    ),
    _spec("sigma_ic1_y", "IC1 σy (mm)", FAMILY_SIGMA, psd_unit="mm", replay_color="#aec7e8"),
    _spec("sigma_ic2_x", "IC2 σx (mm)", FAMILY_SIGMA, psd_unit="mm", replay_color="#d62728"),
    _spec("sigma_ic2_y", "IC2 σy (mm)", FAMILY_SIGMA, psd_unit="mm", replay_color="#ff9896"),
    _spec("bx", "Bx (G)", FAMILY_FIELD, psd_unit="G", replay_color="#1f77b4"),
    _spec("by", "By (G)", FAMILY_FIELD, psd_unit="G", replay_color="#d62728"),
    _spec(
        "b_mag", "|B|", FAMILY_FIELD,
        psd_unit="G",
        replay_visible=False,
        extract=_b_mag_extract,
    ),
    _spec(
        "amp_cmd_x", "Cmd X", FAMILY_AMPLIFIER,
        psd_unit="V",
        replay_visible=False,
    ),
    _spec("amp_cmd_y", "Cmd Y", FAMILY_AMPLIFIER, psd_unit="V", replay_visible=False),
    _spec("amp_rb_x", "Readback X", FAMILY_AMPLIFIER, psd_unit="V", replay_visible=False),
    _spec("amp_rb_y", "Readback Y", FAMILY_AMPLIFIER, psd_unit="V", replay_visible=False),
    _spec(
        "beam", "Beam Current", FAMILY_BEAM,
        psd_unit="nA",
        replay_visible=False,
        beam_off_quiet_threshold=10.0,
    ),
    _spec("ic1_x", "IC1 X", FAMILY_POSITION, psd_unit="mm", replay_visible=False),
    _spec("ic1_y", "IC1 Y", FAMILY_POSITION, psd_unit="mm", replay_visible=False),
    _spec("ic2_x", "IC2 X", FAMILY_POSITION, psd_unit="mm", replay_visible=False),
    _spec("ic2_y", "IC2 Y", FAMILY_POSITION, psd_unit="mm", replay_visible=False),
    _spec(
        "ic1_x_peak", "IC1 X Peak", FAMILY_PEAK,
        psd_unit="nA",
        replay_visible=False,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic1_y_peak", "IC1 Y Peak", FAMILY_PEAK,
        psd_unit="nA",
        replay_visible=False,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic2_x_peak", "IC2 X Peak", FAMILY_PEAK,
        psd_unit="nA",
        replay_visible=False,
        beam_off_quiet_threshold=10.0,
    ),
    _spec(
        "ic2_y_peak", "IC2 Y Peak", FAMILY_PEAK,
        psd_unit="nA",
        replay_visible=False,
        beam_off_quiet_threshold=10.0,
    ),
)

TIMELINE_CHANNEL_BY_KEY: dict[str, TimelineChannelSpec] = {
    spec.key: spec for spec in TIMELINE_CHANNEL_SPECS
}

REPLAY_CHANNEL_SPECS: tuple[TimelineChannelSpec, ...] = tuple(
    spec for spec in TIMELINE_CHANNEL_SPECS if spec.replay_visible
)

FFT_CHANNEL_SPECS: tuple[TimelineChannelSpec, ...] = tuple(
    spec for spec in TIMELINE_CHANNEL_SPECS if spec.fft_visible
)


def channel_available(session: TimelineBundle, spec: TimelineChannelSpec) -> bool:
    if spec.requires_ic3 and not session.get("has_ic3", False):
        return False
    values = spec.resolved_extract()(session)
    return values is not None and len(values) > 0


def available_channel_keys(session_data: dict[str, TimelineBundle]) -> set[str]:
    available: set[str] = set()
    for data in session_data.values():
        for spec in REPLAY_CHANNEL_SPECS:
            if channel_available(data, spec):
                available.add(spec.key)
    return available


def fft_label(spec: TimelineChannelSpec) -> str:
    """Short label for FFT channel pickers (strip trailing units when present)."""
    if spec.label.endswith("(mm)") or spec.label.endswith("(G)"):
        return spec.label.rsplit(" (", 1)[0]
    return spec.label
