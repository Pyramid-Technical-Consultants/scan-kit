"""Keep the pytest suite headless (no blocking matplotlib or Qt windows)."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

# Must be set before matplotlib is imported anywhere in the test process.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"
G3_SESSION = "1091134775"
G2_SESSION = "590658542"
G3_LARGE_SESSION = "1242721320"
HV_SESSION = "447945951"
AMP_G3_SESSION = "1943968267"
AMP_G3_OLD_SESSION = "1262268206"
AMP_G3_CONST_SESSION = "845596095"
AMP_G3_STUCK_SESSION = "863788396"

# First line of Qt-heavy tests in large modules (auto-marked slow below).
_PLAN_SYNTHESIS_UI_START_LINE = 724
_CONFIG_TUNING_UI_START_LINE = 459


def wait_for_qt(
    qapp,
    predicate: Callable[[], bool],
    *,
    timeout_ms: int = 5000,
) -> bool:
    """Pump the Qt event loop until *predicate* is true or *timeout_ms* elapses."""
    from PySide6.QtCore import QElapsedTimer

    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < timeout_ms:
        if predicate():
            return True
        qapp.processEvents()
    return predicate()


@pytest.fixture(scope="session")
def test_data_dir() -> str:
    return str(TEST_DATA)


@pytest.fixture(scope="session")
def g3_session_id() -> str:
    return G3_SESSION


@pytest.fixture(scope="session")
def g3_spot_summary(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.binned_summary_data import load_sessions_summary

    return load_sessions_summary([G3_SESSION], test_data_dir)


@pytest.fixture(scope="session")
def g3_dose_rate(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.binned_summary_data import load_sessions_dose_rate

    return load_sessions_dose_rate([G3_SESSION], test_data_dir)


@pytest.fixture(scope="session")
def g3_current_ratios(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.binned_summary_data import load_sessions_current_ratios

    return load_sessions_current_ratios([G3_SESSION], test_data_dir)


_TIMELINE_CACHE: dict[tuple[str, str], dict] = {}


def _cached_timeline_catalog(session_id: str, test_data_dir: str) -> dict:
    key = (session_id, test_data_dir)
    if key not in _TIMELINE_CACHE:
        from scan_kit.views.timeslice_replay_channels import load_session_timeline_catalog

        _TIMELINE_CACHE[key] = (
            load_session_timeline_catalog(session_id, test_data_dir) or {}
        )
    return _TIMELINE_CACHE[key]


@pytest.fixture(scope="session")
def g3_binned_availability(
    test_data_dir: str,
    g3_spot_summary: dict[str, dict],
    g3_source_availability: dict[str, bool],
) -> dict[str, bool]:
    from scan_kit.views.binned_summary_data import probe_view_option_availability

    availability = probe_view_option_availability(
        [G3_SESSION],
        test_data_dir,
        spot_data=g3_spot_summary,
    )
    # Registry keys are shared with g3_source_availability; merge for consistency.
    availability.update(g3_source_availability)
    return availability


@pytest.fixture(scope="session")
def g3_distribution_availability(g3_source_availability: dict[str, bool]) -> dict[str, bool]:
    from scan_kit.views.distribution_data import _extend_mode_availability

    return _extend_mode_availability(dict(g3_source_availability))


@pytest.fixture(scope="session")
def g3_fft_data(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.fft_data import load_sessions_fft

    return load_sessions_fft([G3_SESSION], test_data_dir)


@pytest.fixture(scope="session")
def g3_large_fft_data(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.fft_data import load_sessions_fft

    return load_sessions_fft([G3_LARGE_SESSION], test_data_dir)


@pytest.fixture(scope="session")
def g3_source_availability(test_data_dir: str) -> dict[str, bool]:
    from scan_kit.data.availability import probe_sessions

    return probe_sessions([G3_SESSION], test_data_dir)


@pytest.fixture(scope="session")
def g3_timeslice_catalog(test_data_dir: str) -> dict[str, dict]:
    data = _cached_timeline_catalog(G3_SESSION, test_data_dir)
    return {G3_SESSION: data} if data else {}


@pytest.fixture(scope="session")
def g3_timeline_catalog(test_data_dir: str) -> dict:
    return _cached_timeline_catalog(G3_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def g2_timeslice_catalog(test_data_dir: str) -> dict[str, dict]:
    data = _cached_timeline_catalog(G2_SESSION, test_data_dir)
    return {G2_SESSION: data} if data else {}


@pytest.fixture(scope="session")
def g2_timeline_catalog(test_data_dir: str) -> dict:
    return _cached_timeline_catalog(G2_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def g3_spot_summary_chamber(test_data_dir: str) -> dict[str, dict]:
    from scan_kit.views.binned_summary_data import load_sessions_summary
    from scan_kit.views.unified_catalog import REFERENCE_CHAMBER

    return load_sessions_summary(
        [G3_SESSION], test_data_dir, reference_frame=REFERENCE_CHAMBER,
    )


@pytest.fixture(scope="session")
def g3_timeslice_summary_table(test_data_dir: str) -> dict:
    from scan_kit.views.binned_summary_data import load_session_timeslice_summary_table

    data = load_session_timeslice_summary_table(G3_SESSION, test_data_dir)
    return data or {}


@pytest.fixture(scope="session")
def hv_session_data(test_data_dir: str):
    from scan_kit.views.ic_hv_transient import _load_session_hv

    return _load_session_hv(HV_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def amp_samples_g2(test_data_dir: str):
    from scan_kit.views.amplifier_correlation import _load_session_samples

    return _load_session_samples(G2_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def amp_samples_g3(test_data_dir: str):
    from scan_kit.views.amplifier_correlation import _load_session_samples

    return _load_session_samples(AMP_G3_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def amp_samples_g3_old(test_data_dir: str):
    from scan_kit.views.amplifier_correlation import _load_session_samples

    return _load_session_samples(AMP_G3_OLD_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def amp_samples_g3_const(test_data_dir: str):
    from scan_kit.views.amplifier_correlation import _load_session_samples

    return _load_session_samples(AMP_G3_CONST_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def amp_samples_g3_stuck(test_data_dir: str):
    from scan_kit.views.amplifier_correlation import _load_session_samples

    return _load_session_samples(AMP_G3_STUCK_SESSION, test_data_dir)


@pytest.fixture(scope="session")
def g2_position_errors(test_data_dir: str):
    from scan_kit.common.timeslice_position_error import load_session_beam_on_position_errors

    return load_session_beam_on_position_errors(G2_SESSION, test_data_dir)


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-mark heavy integration and Qt widget tests as slow."""
    slow_marker = pytest.mark.slow
    ui_start_lines = {
        "test_plan_synthesis.py": _PLAN_SYNTHESIS_UI_START_LINE,
        "test_config_tuning_xml.py": _CONFIG_TUNING_UI_START_LINE,
    }
    slow_modules = frozenset({
        "test_g2_timeslice_position.py",
        "test_g3_iso_timeslice_position.py",
        "test_timeslice_chamber_position.py",
    })
    for item in items:
        path_name = item.path.name
        if path_name in slow_modules:
            item.add_marker(slow_marker)
            continue
        start_line = ui_start_lines.get(path_name)
        if start_line is not None and path_name == item.path.name:
            if item.location[1] >= start_line:
                item.add_marker(slow_marker)


@pytest.fixture
def qt_wait(qapp):
    """Asserting Qt wait helper for smoke tests."""

    def _wait(predicate: Callable[[], bool], *, timeout_ms: int = 5000) -> None:
        assert wait_for_qt(qapp, predicate, timeout_ms=timeout_ms), (
            "Qt condition not met within timeout"
        )

    return _wait


@pytest.fixture(scope="session")
def qapp():
    """One ``QApplication`` for the whole test process (widgets need it, not ``QGuiApplication``)."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def _headless_matplotlib():
    """Prevent ``plt.show()`` from opening a blocking GUI window during tests."""
    captured: list[plt.Figure] = []

    def _capture_show(*args, **kwargs) -> None:
        del args, kwargs
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if fig not in captured:
                captured.append(fig)

    real_show = plt.show
    plt.show = _capture_show
    try:
        yield
    finally:
        plt.show = real_show
        plt.close("all")


@pytest.fixture(autouse=True)
def _headless_qt_windows():
    """Layout-only ``QWidget.show()`` calls without flashing real windows."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QWidget
    except ImportError:
        yield
        return

    real_show = QWidget.show

    def _show_offscreen(self, *args, **kwargs):
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        return real_show(self, *args, **kwargs)

    QWidget.show = _show_offscreen
    try:
        yield
    finally:
        QWidget.show = real_show
