"""Signal sources and display options for the FFT Explorer viewer."""

from __future__ import annotations

from dataclasses import dataclass

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, DataFilterSelection

SIGNAL_IC1 = "ic1"
SIGNAL_IC2 = "ic2"
SIGNAL_IC3 = "ic3"

PRESET_ALL_ICS = "all_ics"
PRESET_IC1_ONLY = "ic1_only"


@dataclass(frozen=True)
class FftSignalDef:
    id: str
    label: str
    ic_key: str
    column_key: str


@dataclass(frozen=True)
class PresetDef:
    id: str
    label: str
    signals: tuple[str, ...]
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH
    annotate_peaks: bool = True


FFT_SIGNALS: tuple[FftSignalDef, ...] = (
    FftSignalDef(SIGNAL_IC1, "IC1 Current", "ic1", "ic1_current"),
    FftSignalDef(SIGNAL_IC2, "IC2 Current", "ic2", "ic2_current"),
    FftSignalDef(SIGNAL_IC3, "IC3 Current (A+B+C+D)", "ic3", "ic3_current"),
)

PRESETS: tuple[PresetDef, ...] = (
    PresetDef(
        PRESET_ALL_ICS,
        "All IC currents",
        (SIGNAL_IC1, SIGNAL_IC2, SIGNAL_IC3),
        beam_state_filter=FILTER_BEAM_BOTH,
    ),
    PresetDef(
        PRESET_IC1_ONLY,
        "IC1 only",
        (SIGNAL_IC1,),
        beam_state_filter=FILTER_BEAM_BOTH,
    ),
)

SIGNAL_BY_ID = {s.id: s for s in FFT_SIGNALS}
PRESET_BY_ID = {p.id: p for p in PRESETS}


@dataclass
class FftConfig:
    signals: tuple[str, ...] = (SIGNAL_IC1, SIGNAL_IC2)
    domain_filter: str = FILTER_ALL
    beam_state_filter: str = FILTER_BEAM_BOTH
    annotate_peaks: bool = True

    @property
    def data_filter(self) -> DataFilterSelection:
        return DataFilterSelection(
            domain_filter=self.domain_filter,
            beam_state_filter=self.beam_state_filter,
        )

    @property
    def title(self) -> str:
        labels = [SIGNAL_BY_ID[s].label for s in self.signals if s in SIGNAL_BY_ID]
        if not labels:
            return "FFT Explorer"
        if len(labels) == 1:
            return f"FFT — {labels[0]}"
        return "FFT — " + ", ".join(labels)

    @property
    def ic_keys(self) -> tuple[str, ...]:
        return tuple(
            SIGNAL_BY_ID[s].ic_key
            for s in self.signals
            if s in SIGNAL_BY_ID
        )

    @property
    def ic_labels(self) -> tuple[str, ...]:
        return tuple(
            SIGNAL_BY_ID[s].label
            for s in self.signals
            if s in SIGNAL_BY_ID
        )

    @property
    def column_keys(self) -> tuple[str, ...]:
        return tuple(
            SIGNAL_BY_ID[s].column_key
            for s in self.signals
            if s in SIGNAL_BY_ID
        )
