"""Qt shell for the FFT Explorer viewer."""

from __future__ import annotations

import logging
from typing import Sequence

from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from ..common.data_filter import FILTER_ALL, FILTER_BEAM_BOTH, FILTER_BEAM_ON
from ..common.settings import ViewSettings
from .async_refresh import DebouncedBackgroundTask
from .fft_catalog import FFT_SIGNALS, PRESET_BY_ID, PRESETS, FftConfig
from .fft_data import default_config, load_sessions_fft, probe_signal_availability
from .fft_ui import render_fft
from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    new_headless_figure,
    run_view_window,
)
from .unified_view_controls import DataFilterPanel, sync_data_filter_panel

_log = logging.getLogger(__name__)


class FftExplorerWindow(PlotViewWindow):
    """Configurable frequency-domain explorer for timeslice signals."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="FFT Explorer",
            figsize=(16, 9),
            side_panel_min_width=240,
            side_panel_default_width=300,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._session_data: dict[str, dict] = {}
        self._signal_availability: dict[str, bool] = {}
        self._signal_checks: dict[str, QCheckBox] = {}
        self._filter_panel: DataFilterPanel | None = None
        self._peaks_box: QCheckBox | None = None
        self._refresh_generation = 0
        self._updating = False
        self._pending_preset = initial_preset

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(80)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._render_task = DebouncedBackgroundTask(debounce_ms=50, parent=self)
        self._render_task.finished.connect(self._on_render_finished)

        self.set_side_panel(self._build_controls())
        self._show_status_message("Loading timeslice data…")
        self._start_initial_load()

    def _show_status_message(self, message: str) -> None:
        self.figure.clear()
        self.figure.text(0.5, 0.5, message, ha="center", va="center")
        self.draw_idle()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [(preset.id, preset.label, True) for preset in PRESETS],
                self._apply_preset,
            )
        )

        signal_group = QGroupBox("Signals")
        signal_layout = QVBoxLayout(signal_group)
        for signal in FFT_SIGNALS:
            box = QCheckBox(signal.label)
            box.toggled.connect(self._schedule_refresh)
            signal_layout.addWidget(box)
            self._signal_checks[signal.id] = box
        layout.addWidget(signal_group)

        self._filter_panel = DataFilterPanel(
            on_selection_changed=self._schedule_refresh,
            domain_current=FILTER_ALL,
            beam_current=FILTER_BEAM_ON,
        )
        layout.addWidget(self._filter_panel)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        self._peaks_box = QCheckBox("Annotate Peaks")
        self._peaks_box.setChecked(True)
        self._peaks_box.toggled.connect(self._schedule_refresh)
        options_layout.addWidget(self._peaks_box)
        layout.addWidget(options_group)

        layout.addStretch(1)
        return panel

    def _start_initial_load(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir
        settings = self._settings

        def loader() -> tuple[dict[str, dict], dict[str, bool]]:
            session_data = load_sessions_fft(session_ids, base_dir, settings=settings)
            availability = probe_signal_availability(
                session_ids, base_dir, session_data=session_data,
            )
            return session_data, availability

        self._load_task.schedule(loader)

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID[preset_id]
        self._set_config(
            FftConfig(
                signals=preset.signals,
                domain_filter=preset.domain_filter,
                beam_state_filter=preset.beam_state_filter,
                annotate_peaks=preset.annotate_peaks,
            )
        )
        self._schedule_refresh()

    def _set_config(self, config: FftConfig) -> None:
        self._updating = True
        try:
            for signal in FFT_SIGNALS:
                box = self._signal_checks.get(signal.id)
                if box is None:
                    continue
                enabled = self._signal_availability.get(signal.id, False)
                if not enabled and self._session_data:
                    col = signal.column_key
                    enabled = any(
                        len(data.get(col, ())) > 0
                        for data in self._session_data.values()
                    )
                box.setEnabled(enabled)
                box.setChecked(enabled and signal.id in config.signals)
            if self._filter_panel is not None:
                self._filter_panel.set_domain(config.domain_filter)
                self._filter_panel.set_beam_state(config.beam_state_filter)
            if self._peaks_box is not None:
                self._peaks_box.setChecked(config.annotate_peaks)
        finally:
            self._updating = False

    def _read_config(self) -> FftConfig:
        signals = tuple(
            signal.id
            for signal in FFT_SIGNALS
            if self._signal_checks.get(signal.id) is not None
            and self._signal_checks[signal.id].isChecked()
        )
        return FftConfig(
            signals=signals,
            domain_filter=(
                self._filter_panel.selected_domain()
                if self._filter_panel is not None
                else FILTER_ALL
            ) or FILTER_ALL,
            beam_state_filter=(
                self._filter_panel.selected_beam_state()
                if self._filter_panel is not None
                else FILTER_BEAM_BOTH
            ) or FILTER_BEAM_BOTH,
            annotate_peaks=(
                self._peaks_box.isChecked() if self._peaks_box is not None else True
            ),
        )

    def _schedule_refresh(self) -> None:
        if self._updating:
            return
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        gen = self._refresh_generation
        config = self._read_config()
        self.setWindowTitle(config.title)
        self._schedule_render(gen, config)

    def _schedule_render(self, gen: int, config: FftConfig) -> None:
        session_data = self._session_data
        base_dir = self._base_dir
        figsize = tuple(float(v) for v in self.figure.get_size_inches())

        def render_fn() -> tuple[int, Figure | None]:
            fig = new_headless_figure(figsize)
            try:
                render_fft(fig, config, session_data, base_dir)
            except Exception:
                _log.exception("FFT Explorer render failed")
                return gen, None
            return gen, fig

        self._render_task.schedule(render_fn)

    @Slot(int, object)
    def _on_load_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            self._show_status_message("Failed to load timeslice data")
            _log.error("FFT load returned invalid result: %r", result)
            return
        session_data, availability = result
        self._session_data = session_data
        self._signal_availability = availability

        if not session_data:
            self._show_status_message("No timeslice IC current data for selected sessions")
            self._set_config(FftConfig(signals=()))
            return

        sync_data_filter_panel(
            self._filter_panel,
            supports_filter=True,
            has_beam_state=True,
        )

        if self._pending_preset and self._pending_preset in PRESET_BY_ID:
            preset_id = self._pending_preset
            self._pending_preset = None
            self._apply_preset(preset_id)
        else:
            self._set_config(default_config(availability))
            self._schedule_refresh()

    @Slot(int, object)
    def _on_render_finished(self, _task_generation: int, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 2:
            return
        gen, fig = result
        if gen != self._refresh_generation:
            if isinstance(fig, Figure):
                import matplotlib.pyplot as plt
                plt.close(fig)
            return
        if not isinstance(fig, Figure):
            self._show_status_message("FFT render failed — see log for details")
            return
        previous = self.figure
        self.figure = fig
        self.canvas.figure = fig
        self.draw_idle()
        if previous is not fig:
            import matplotlib.pyplot as plt
            plt.close(previous)


def run_fft_explorer_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        print("No sessions selected")
        return

    run_view_window(
        lambda: FftExplorerWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
    )
