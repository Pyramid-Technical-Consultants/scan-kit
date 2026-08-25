"""Signal metrics and channels for the FFT Explorer viewer."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, DataFilterSelection
from ..data.timeline_channels import (
    FAMILY_AMPLIFIER,
    FAMILY_BEAM,
    FAMILY_DDOSE,
    FAMILY_FIELD,
    FAMILY_IC,
    FAMILY_PEAK,
    FAMILY_POSITION,
    FAMILY_SIGMA,
    FFT_CHANNEL_SPECS,
    TIMELINE_CHANNEL_BY_KEY,
    fft_label,
)

METRIC_IC_CURRENT = "ic_current"
METRIC_DDOSE = "ddose"
METRIC_SIGMA = "sigma"
METRIC_MAG_FIELD = "mag_field"
METRIC_AMPLIFIER = "amplifier"
METRIC_BEAM = "beam"
METRIC_POSITION = "position"
METRIC_PEAK_AMPLITUDE = "peak_amplitude"

PRESET_ALL_ICS = "all_ics"
PRESET_IC1_ONLY = "ic1_only"
PRESET_MAG_FIELD = "mag_field"
PRESET_AMPLIFIER = "amplifier"


@dataclass(frozen=True)
class FftChannelDef:
    id: str
    label: str
    psd_unit: str
    requires_ic3: bool = False


@dataclass(frozen=True)
class FftMetricDef:
    id: str
    label: str
    channels: tuple[FftChannelDef, ...]
    default_channel_ids: tuple[str, ...]
    beam_off_quiet_threshold: float | None = None


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    metric_id: str
    channels: tuple[str, ...]
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH
    annotate_peaks: bool = True


def _fft_channel(spec_key: str) -> FftChannelDef:
    spec = TIMELINE_CHANNEL_BY_KEY[spec_key]
    return FftChannelDef(
        spec.key,
        fft_label(spec),
        spec.psd_unit or "",
        requires_ic3=spec.requires_ic3,
    )


def _fft_metric(
    metric_id: str,
    label: str,
    family: str,
    default_channel_ids: tuple[str, ...],
    beam_off_quiet_threshold: float | None = None,
) -> FftMetricDef:
    channels = tuple(
        _fft_channel(spec.key)
        for spec in FFT_CHANNEL_SPECS
        if spec.family == family
    )
    return FftMetricDef(
        metric_id,
        label,
        channels,
        default_channel_ids,
        beam_off_quiet_threshold=beam_off_quiet_threshold,
    )


FFT_METRICS: tuple[FftMetricDef, ...] = (
    _fft_metric(
        METRIC_IC_CURRENT, "IC Current", FAMILY_IC,
        ("ic1", "ic2", "ic3"), beam_off_quiet_threshold=10.0,
    ),
    _fft_metric(
        METRIC_DDOSE, "dDose/dt", FAMILY_DDOSE,
        ("ic1_ddose", "ic2_ddose", "ic3_ddose"), beam_off_quiet_threshold=10.0,
    ),
    _fft_metric(
        METRIC_SIGMA, "Sigma", FAMILY_SIGMA,
        ("sigma_ic1_x", "sigma_ic1_y", "sigma_ic2_x", "sigma_ic2_y"),
    ),
    _fft_metric(
        METRIC_MAG_FIELD, "Magnetic Field", FAMILY_FIELD,
        ("bx", "by"),
    ),
    _fft_metric(
        METRIC_AMPLIFIER, "Amplifier", FAMILY_AMPLIFIER,
        ("amp_cmd_x", "amp_cmd_y", "amp_rb_x", "amp_rb_y"),
    ),
    _fft_metric(
        METRIC_BEAM, "Beam Current", FAMILY_BEAM,
        ("beam",), beam_off_quiet_threshold=10.0,
    ),
    _fft_metric(
        METRIC_POSITION, "Chamber Position", FAMILY_POSITION,
        ("ic1_x", "ic1_y", "ic2_x", "ic2_y"),
    ),
    _fft_metric(
        METRIC_PEAK_AMPLITUDE, "Peak Amplitude (G3)", FAMILY_PEAK,
        ("ic1_x_peak", "ic1_y_peak", "ic2_x_peak", "ic2_y_peak"),
        beam_off_quiet_threshold=10.0,
    ),
)

PRESETS: tuple[PresetDef, ...] = (
    PresetDef(
        PRESET_ALL_ICS,
        "All IC currents",
        METRIC_IC_CURRENT,
        ("ic1", "ic2", "ic3"),
        beam_state_filter=FILTER_BEAM_BOTH,
    ),
    PresetDef(
        PRESET_IC1_ONLY,
        "IC1 only",
        METRIC_IC_CURRENT,
        ("ic1",),
        beam_state_filter=FILTER_BEAM_BOTH,
    ),
    PresetDef(
        PRESET_MAG_FIELD,
        "Hall probes",
        METRIC_MAG_FIELD,
        ("bx", "by", "b_mag"),
    ),
    PresetDef(
        PRESET_AMPLIFIER,
        "Amplifier cmd/readback",
        METRIC_AMPLIFIER,
        ("amp_cmd_x", "amp_cmd_y", "amp_rb_x", "amp_rb_y"),
    ),
)

METRIC_BY_ID = {metric.id: metric for metric in FFT_METRICS}
CHANNEL_BY_ID = {
    channel.id: channel
    for metric in FFT_METRICS
    for channel in metric.channels
}
PRESET_BY_ID = {preset.id: preset for preset in PRESETS}


@dataclass
class FftConfig:
    metric_id: str = METRIC_IC_CURRENT
    channels: tuple[str, ...] = ("ic1", "ic2")
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH
    annotate_peaks: bool = True

    @property
    def metric(self) -> FftMetricDef | None:
        return METRIC_BY_ID.get(self.metric_id)

    @property
    def data_filter(self) -> DataFilterSelection:
        return DataFilterSelection(
            domain_filter=self.domain_filter,
            beam_state_filter=self.beam_state_filter,
        )

    @property
    def title(self) -> str:
        metric = self.metric
        if metric is None:
            return "FFT Explorer"
        labels = [
            CHANNEL_BY_ID[ch].label
            for ch in self.channels
            if ch in CHANNEL_BY_ID
        ]
        if not labels:
            return f"FFT — {metric.label}"
        if len(labels) == 1:
            return f"FFT — {labels[0]}"
        if len(labels) <= 3:
            return f"FFT — {metric.label}: " + ", ".join(labels)
        return f"FFT — {metric.label} ({len(labels)} channels)"

    @property
    def channel_defs(self) -> tuple[FftChannelDef, ...]:
        return tuple(
            CHANNEL_BY_ID[ch]
            for ch in self.channels
            if ch in CHANNEL_BY_ID
        )

    @property
    def column_keys(self) -> tuple[str, ...]:
        return self.channels
