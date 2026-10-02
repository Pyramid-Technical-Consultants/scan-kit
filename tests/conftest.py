"""Keep the pytest suite headless (no blocking matplotlib or Qt windows)."""

from __future__ import annotations

import inspect
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
# xdist workers do not always inherit the repo root on sys.path; several tests import
# shared constants via ``from tests.conftest import ...``.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Must be set before matplotlib is imported anywhere in the test process.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

TEST_DATA = _ROOT / "test_data"
HAS_TEST_DATA = TEST_DATA.is_dir() and os.environ.get("SCAN_KIT_SKIP_TEST_DATA") != "1"
G3_SESSION = "1091134775"
G2_SESSION = "590658542"
G3_LARGE_SESSION = "1242721320"
HV_SESSION = "447945951"
AMP_G3_SESSION = "1943968267"
AMP_G3_OLD_SESSION = "1262268206"
AMP_G3_CONST_SESSION = "845596095"
AMP_G3_STUCK_SESSION = "863788396"

_SESSION_FIXTURES = frozenset({
    "test_data_dir",
    "g3_session_id",
    "g2_position_errors",
})

_SKIP_NO_TEST_DATA = pytest.mark.skip(
    reason="test_data/ is not available (gitignored; run integration tests locally)",
)

_HELPERS_USING_TEST_DATA: dict[str, frozenset[str]] = {}
_PATH_CONSTANTS_USING_TEST_DATA: dict[str, frozenset[str]] = {}


def _module_helpers_using_test_data(module) -> frozenset[str]:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return frozenset()
    cached = _HELPERS_USING_TEST_DATA.get(module_file)
    if cached is not None:
        return cached
    names: set[str] = set()
    for name, obj in vars(module).items():
        if not name.startswith("_") or not callable(obj):
            continue
        try:
            helper_source = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if "_TEST_DATA" in helper_source or "TEST_DATA" in helper_source:
            names.add(name)
    cached = frozenset(names)
    _HELPERS_USING_TEST_DATA[module_file] = cached
    return cached


def _module_path_constants_using_test_data(module) -> frozenset[str]:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return frozenset()
    cached = _PATH_CONSTANTS_USING_TEST_DATA.get(module_file)
    if cached is not None:
        return cached
    names: set[str] = set()
    for name, value in vars(module).items():
        if name in ("TEST_DATA", "_TEST_DATA"):
            continue
        if isinstance(value, Path) and "test_data" in value.parts:
            names.add(name)
    cached = frozenset(names)
    _PATH_CONSTANTS_USING_TEST_DATA[module_file] = cached
    return cached


def _test_function_uses_test_data(item: pytest.Item) -> bool:
    """Skip tests that load the gitignored fixture tree (directly or via helpers)."""
    try:
        source = inspect.getsource(item.obj)
    except (OSError, TypeError):
        return False
    if "_TEST_DATA" in source or "TEST_DATA" in source:
        return True
    for name in _module_path_constants_using_test_data(item.module):
        if re.search(rf"\b{name}\b", source):
            return True
    return any(f"{name}(" in source for name in _module_helpers_using_test_data(item.module))


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
    if not HAS_TEST_DATA:
        pytest.skip("test_data/ is not available")
    return str(TEST_DATA)


@pytest.fixture(scope="session")
def g3_session_id() -> str:
    return G3_SESSION



@pytest.fixture(scope="session")
def g2_position_errors(test_data_dir: str):
    from scan_kit.common.timeslice_position_error import load_session_beam_on_position_errors

    return load_session_beam_on_position_errors(G2_SESSION, test_data_dir)


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-mark heavy integration and Qt widget tests as slow."""
    slow_marker = pytest.mark.slow
    slow_modules = frozenset({
        "test_g2_timeslice_position.py",
        "test_g3_iso_timeslice_position.py",
        "test_timeslice_chamber_position.py",
    })
    for item in items:
        if not HAS_TEST_DATA:
            if _SESSION_FIXTURES.intersection(item.fixturenames):
                item.add_marker(_SKIP_NO_TEST_DATA)
            elif _test_function_uses_test_data(item):
                item.add_marker(_SKIP_NO_TEST_DATA)

        path_name = item.path.name
        if path_name in slow_modules:
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

    from scan_kit.common.gui_gc import collect_on_gui_thread

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    collect_on_gui_thread(app)
    yield app


@pytest.fixture(autouse=True)
def _collect_on_main_thread():
    """Many tests never spin the event loop, so collect between them, on this thread."""
    yield
    from scan_kit.common.gui_gc import collect_due

    collect_due()


@pytest.fixture(scope="session")
def gpu():
    """The WebGPU adapter; skips without a hardware one (software adapters are too slow to transport)."""
    from scan_kit.gpu import GpuUnavailable, adapter_info

    try:
        info = adapter_info()
    except GpuUnavailable as exc:
        pytest.skip(f"WebGPU unavailable: {exc}")
    if info["adapter_type"] == "CPU":
        pytest.skip("software WebGPU adapter is too slow for transport")
    return info


@pytest.fixture(autouse=True)
def _isolate_user_store(tmp_path_factory, monkeypatch):
    """Keep the app SQLite DB out of the developer's ``~/.scan-kit``."""
    root = tmp_path_factory.mktemp("scan-kit-user")
    monkeypatch.setattr("scan_kit.common.user_store.user_data_dir", lambda: root)
    from scan_kit.common.user_store import reset_connection

    reset_connection()
    try:
        yield
    finally:
        reset_connection()


@pytest.fixture(autouse=True)
def _restore_cwd():
    """Undo working-directory changes so relative ``test_data`` lookups keep resolving.

    ``prepare_linux_frozen_env`` chdirs by design, so the tests covering it leak the
    process cwd into whichever test the xdist worker picks up next.
    """
    original = os.getcwd()
    try:
        yield
    finally:
        if os.getcwd() != original:
            os.chdir(original)


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
