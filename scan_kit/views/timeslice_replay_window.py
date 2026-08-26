"""Qt shell hosting Matplotlib timeslice replay with channel controls."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .plot_view_shell import (
    PlotViewWindow,
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .timeslice_replay_channels import (
    PRESET_CHANNELS,
    PRESET_DDOSE,
    PRESET_LABELS,
    available_channel_keys,
    build_replay_config,
    channel_defs_by_family,
    default_selected_keys,
    filter_available_keys,
    load_sessions_catalog,
)
from .timeslice_replay_ui import render_timeslice_replay


class TimesliceReplayWindow(PlotViewWindow):
    """Interactive timeslice replay with Qt channel / option controls."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        initial_bg_subtract: bool = False,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            title="Timeslice Replay",
            figsize=(16, 9),
            side_panel_min_width=220,
            side_panel_default_width=280,
            parent=parent,
        )
        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._bg_subtract = initial_bg_subtract
        self._session_data = load_sessions_catalog(
            self._session_ids,
            self._base_dir,
            bg_subtract=self._bg_subtract,
        )
        self._available = available_channel_keys(self._session_data)
        self._channel_checks: dict[str, QCheckBox] = {}
        self._updating = False

        self.set_side_panel(self._build_controls())
        self._init_selection(initial_preset)
        self._refresh_plot()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [
                    (
                        preset_id,
                        label,
                        bool(filter_available_keys(
                            PRESET_CHANNELS[preset_id], self._available,
                        )),
                    )
                    for preset_id, label in PRESET_LABELS.items()
                ],
                self._apply_preset,
            )
        )

        for family, channels in channel_defs_by_family():
            group = QGroupBox(family)
            group_layout = QVBoxLayout(group)
            for channel in channels:
                check = QCheckBox(channel.label)
                enabled = channel.key in self._available
                check.setEnabled(bool(enabled))
                if not enabled:
                    check.setToolTip("Not available in the selected session(s)")
                check.stateChanged.connect(self._on_controls_changed)
                self._channel_checks[channel.key] = check
                group_layout.addWidget(check)
            layout.addWidget(group)

        options = QGroupBox("Options")
        opt_layout = QVBoxLayout(options)
        self._bg_check = QCheckBox("Background Subtract")
        self._bg_check.setChecked(self._bg_subtract)
        self._bg_check.toggled.connect(self._on_bg_subtract_toggled)
        opt_layout.addWidget(self._bg_check)

        self._peer_check = QCheckBox("Peer Overlay (Single Session)")
        self._peer_check.setChecked(False)
        self._peer_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._peer_check)

        self._edges_check = QCheckBox("Beam-Off Edges")
        self._edges_check.setChecked(True)
        self._edges_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._edges_check)

        self._digital_check = QCheckBox("Digital Lanes")
        self._digital_check.setChecked(True)
        self._digital_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._digital_check)

        self._beam_check = QCheckBox("Beam Current Twin Axis")
        self._beam_check.setChecked(True)
        self._beam_check.toggled.connect(self._on_controls_changed)
        opt_layout.addWidget(self._beam_check)
        layout.addWidget(options)

        if not self._session_data:
            note = QLabel("No timeslice data found for the selected sessions.")
            note.setWordWrap(True)
            layout.addWidget(note)

        layout.addStretch(1)
        return panel

    def _init_selection(self, initial_preset: str | None) -> None:
        if initial_preset and initial_preset in PRESET_CHANNELS:
            keys = filter_available_keys(
                PRESET_CHANNELS[initial_preset], self._available,
            )
            if initial_preset == PRESET_DDOSE and len(self._session_data) <= 1:
                self._peer_check.setChecked(True)
        else:
            keys = default_selected_keys(self._available)
        self._set_selected_keys(keys)

    def _set_selected_keys(self, keys: Sequence[str]) -> None:
        selected = set(keys)
        self._updating = True
        try:
            for key, check in self._channel_checks.items():
                check.setChecked(key in selected)
        finally:
            self._updating = False

    def _selected_keys(self) -> list[str]:
        return [
            key for key, check in self._channel_checks.items() if check.isChecked()
        ]

    def _apply_preset(self, preset_id: str) -> None:
        keys = filter_available_keys(PRESET_CHANNELS[preset_id], self._available)
        self._set_selected_keys(keys)
        self._updating = True
        try:
            self._peer_check.setChecked(
                preset_id == PRESET_DDOSE and len(self._session_data) <= 1
            )
        finally:
            self._updating = False
        self._refresh_plot()

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._refresh_plot()

    def _on_bg_subtract_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._bg_subtract = checked
        selected = self._selected_keys()
        self._session_data = load_sessions_catalog(
            self._session_ids,
            self._base_dir,
            bg_subtract=self._bg_subtract,
        )
        self._available = available_channel_keys(self._session_data)
        self._updating = True
        try:
            for key, check in self._channel_checks.items():
                enabled = key in self._available
                check.setEnabled(bool(enabled))
                if not enabled:
                    check.setChecked(False)
            for key in selected:
                if key in self._available:
                    self._channel_checks[key].setChecked(True)
            if not self._selected_keys():
                self._set_selected_keys(default_selected_keys(self._available))
        finally:
            self._updating = False
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        config = build_replay_config(
            self._selected_keys(),
            self._session_data,
            peer_overlay=self._peer_check.isChecked(),
            show_digital=self._digital_check.isChecked(),
            show_beam_twin=self._beam_check.isChecked(),
            beam_off_edges=self._edges_check.isChecked(),
        )
        render_timeslice_replay(
            self.figure, config, self._session_data, self._base_dir,
        )
        self.draw_idle()


def run_timeslice_replay_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    bg_subtract: bool = False,
    initial_preset: str | None = None,
) -> None:
    """Create the Qt timeslice replay window and run its event loop."""
    if not session_ids:
        print("No sessions selected")
        return

    run_view_window(
        lambda: TimesliceReplayWindow(
            session_ids,
            base_dir,
            initial_bg_subtract=bg_subtract,
            initial_preset=initial_preset,
        ),
    )
