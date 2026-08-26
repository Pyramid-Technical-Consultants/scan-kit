"""Qt shell for the 3D IC beam trajectory viewer (visPy)."""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..common import ViewSettings
from ..common.session_notes import load_notes
from .async_refresh import DebouncedBackgroundTask
from .plot_view_shell import (
    make_presets_menu_button,
    make_side_panel_column,
    run_view_window,
)
from .trajectory_catalog import (
    PRESETS,
    PRESET_BY_ID,
    TrajectoryConfig,
)
from .trajectory_data import (
    format_session_summary,
    load_trajectory_sessions,
    probe_trajectory_availability,
)
from .trajectory_vispy import TrajectoryScene, default_session_colors


class TrajectoryWindow(QMainWindow):
    """3D IC beam trajectory viewer with unified side controls."""

    def __init__(
        self,
        session_ids: Sequence[str],
        base_dir: str,
        *,
        settings: ViewSettings | None = None,
        initial_preset: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("IC Beam Trajectory (3D)")
        self.resize(1400, 900)

        from vispy import scene
        from vispy.app import use_app

        use_app("pyside6")
        self._vispy_canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor="#1a1a1a",
            size=(1200, 800),
            show=False,
        )
        self._scene = TrajectoryScene(self._vispy_canvas)

        plot_host = QWidget()
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(6, 6, 0, 6)
        plot_layout.addWidget(self._vispy_canvas.native)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(plot_host)

        side_panel = self._build_controls()
        self._splitter.addWidget(side_panel)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setSizes([1100, 300])

        self.setCentralWidget(self._splitter)

        self._session_ids = list(session_ids)
        self._base_dir = base_dir
        self._settings = settings
        self._sessions: dict[str, object] = {}
        self._notes = load_notes(base_dir)
        self._pending_preset = initial_preset
        self._refresh_generation = 0
        self._updating = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(60)
        self._refresh_timer.timeout.connect(self._start_refresh)

        self._load_task = DebouncedBackgroundTask(debounce_ms=0, parent=self)
        self._load_task.finished.connect(self._on_load_finished)

        self._show_status("Loading trajectory data…")
        self._start_load()

    def _build_controls(self) -> QWidget:
        panel, layout = make_side_panel_column()

        layout.addWidget(
            make_presets_menu_button(
                [(p.id, p.label, True) for p in PRESETS],
                self._apply_preset,
            )
        )

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        self._show_spots = QCheckBox("Measured spot trajectories")
        self._show_spots.setChecked(True)
        self._show_plan = QCheckBox("Plan rays (iso ↔ pivot)")
        self._show_plan.setChecked(True)
        self._show_pivot = QCheckBox("X/Y scan dipole pivots")
        self._show_pivot.setChecked(True)
        self._show_iso = QCheckBox("Fitted iso planes (per axis)")
        self._show_iso.setChecked(True)
        self._show_ic = QCheckBox("IC chamber planes")
        self._show_ic.setChecked(True)
        for box in (
            self._show_spots,
            self._show_plan,
            self._show_pivot,
            self._show_iso,
            self._show_ic,
        ):
            box.toggled.connect(self._on_controls_changed)
            display_layout.addWidget(box)
        layout.addWidget(display_group)

        extend_group = QGroupBox("Fallback extent (mm)")
        extend_layout = QVBoxLayout(extend_group)
        extend_layout.addWidget(
            QLabel(
                "Rays span furthest upstream dipole pivot to furthest iso plane. "
                "Fallback when fits are missing.",
            ),
        )
        up_row, self._extend_up_spin = self._make_extend_spin("Upstream of IC2", 2000.0)
        down_row, self._extend_down_spin = self._make_extend_spin("Downstream of IC1", 2000.0)
        extend_layout.addWidget(up_row)
        extend_layout.addWidget(down_row)
        layout.addWidget(extend_group)

        info_group = QGroupBox("Session fits")
        info_layout = QVBoxLayout(info_group)
        self._info_label = QLabel("—")
        self._info_label.setWordWrap(True)
        self._info_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse,
        )
        info_layout.addWidget(self._info_label)
        layout.addWidget(info_group)

        layout.addStretch(1)
        return panel

    def _make_extend_spin(self, label: str, default: float) -> tuple[QWidget, QDoubleSpinBox]:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(100.0, 5000.0)
        spin.setSingleStep(100.0)
        spin.setValue(default)
        spin.valueChanged.connect(self._on_controls_changed)
        row_layout.addWidget(spin)
        return row, spin

    def _read_config(self) -> TrajectoryConfig:
        return TrajectoryConfig(
            show_spots=self._show_spots.isChecked(),
            show_plan=self._show_plan.isChecked(),
            show_pivot_markers=self._show_pivot.isChecked(),
            show_iso_planes=self._show_iso.isChecked(),
            show_ic_planes=self._show_ic.isChecked(),
            extend_upstream_mm=self._extend_up_spin.value(),
            extend_downstream_mm=self._extend_down_spin.value(),
        )

    def _set_config(self, config: TrajectoryConfig) -> None:
        self._updating = True
        try:
            self._show_spots.setChecked(config.show_spots)
            self._show_plan.setChecked(config.show_plan)
            self._show_pivot.setChecked(config.show_pivot_markers)
            self._show_iso.setChecked(config.show_iso_planes)
            self._show_ic.setChecked(config.show_ic_planes)
            self._extend_up_spin.setValue(config.extend_upstream_mm)
            self._extend_down_spin.setValue(config.extend_downstream_mm)
        finally:
            self._updating = False

    def _apply_preset(self, preset_id: str) -> None:
        preset = PRESET_BY_ID.get(preset_id)
        if preset is None:
            return
        self._set_config(
            TrajectoryConfig(
                show_spots=preset.show_spots,
                show_plan=preset.show_plan,
                show_pivot_markers=preset.show_pivot_markers,
                show_iso_planes=preset.show_iso_planes,
                show_ic_planes=preset.show_ic_planes,
                extend_upstream_mm=preset.extend_upstream_mm,
                extend_downstream_mm=preset.extend_downstream_mm,
            )
        )
        self._schedule_refresh()

    def _on_controls_changed(self, *_args) -> None:
        if self._updating:
            return
        self._schedule_refresh()

    def _show_status(self, message: str) -> None:
        self._scene.clear()
        from vispy import scene

        text = scene.Text(
            message,
            color="white",
            font_size=14,
            pos=(0, 0, 50),
            parent=self._scene._view.scene,
        )
        self._scene._nodes.append(text)
        self._vispy_canvas.update()

    def _start_load(self) -> None:
        session_ids = list(self._session_ids)
        base_dir = self._base_dir

        def loader() -> dict:
            return load_trajectory_sessions(session_ids, base_dir)

        self._load_task.schedule(loader)

    @Slot(int, object)
    def _on_load_finished(self, _gen: int, result: object) -> None:
        if not isinstance(result, dict):
            self._show_status("Failed to load trajectory data")
            return
        self._sessions = result
        if not self._sessions:
            self._show_status("No raw IC1/IC2 position data found")
            return

        preset = self._pending_preset
        self._pending_preset = None
        if preset and preset in PRESET_BY_ID:
            self._apply_preset(preset)
        else:
            self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        self._refresh_generation += 1
        self._refresh_timer.start()

    def _start_refresh(self) -> None:
        config = self._read_config()
        self.setWindowTitle(config.title)
        loaded_ids = [sid for sid in self._session_ids if sid in self._sessions]
        colors = default_session_colors(len(loaded_ids))

        summaries = [
            format_session_summary(self._sessions[sid], notes=self._notes)
            for sid in loaded_ids
        ]
        self._info_label.setText("\n\n".join(summaries) if summaries else "—")

        self._scene.render(
            self._sessions,
            config,
            loaded_ids,
            colors,
        )


def run_trajectory_window(
    session_ids: Sequence[str],
    base_dir: str = "test_data",
    *,
    settings: ViewSettings | None = None,
    initial_preset: str | None = None,
) -> None:
    if not session_ids:
        return

    run_view_window(
        lambda: TrajectoryWindow(
            session_ids,
            base_dir,
            settings=settings,
            initial_preset=initial_preset,
        ),
        maximize=True,
    )
