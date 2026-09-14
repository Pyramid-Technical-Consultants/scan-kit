"""Per-view import smoke tests that mirror frozen exe subprocess startup.

Warm workers pre-init QtAgg before importing view modules. Tk-only views must
never go through that path. These tests catch regressions like numpy trapz removal
or TkAgg/Qt backend clashes before a PyInstaller release.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from scan_kit.views import TK_ONLY_VIEW_MODULES, VIEWS, view_module_name

_QT_VIEW_MODULES = [
    view_module_name(entry)
    for entry in VIEWS
    if view_module_name(entry) not in TK_ONLY_VIEW_MODULES
]
_TK_VIEW_MODULES = sorted(TK_ONLY_VIEW_MODULES)


def _run_import_script(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("module_name", _QT_VIEW_MODULES)
def test_qt_view_imports_after_frozen_matplotlib_init(module_name: str) -> None:
    """Each Qt/matplotlib view must import after the warm-worker QtAgg setup."""
    code = textwrap.dedent(
        f"""
        import sys
        sys.frozen = True
        from scan_kit.common.matplotlib_backend import init_matplotlib_for_views
        init_matplotlib_for_views()
        import importlib
        mod = importlib.import_module("scan_kit.views.{module_name}")
        assert callable(mod.run)
        """
    )
    result = _run_import_script(code)
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("module_name", _TK_VIEW_MODULES)
def test_tk_view_imports_without_qt_init(module_name: str) -> None:
    """Tk views must import in a process that never initialized Qt."""
    code = textwrap.dedent(
        f"""
        import importlib
        mod = importlib.import_module("scan_kit.views.{module_name}")
        assert callable(mod.run)
        """
    )
    result = _run_import_script(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_ic_hv_transient_trapz_imports_on_numpy2() -> None:
    """ic_hv_transient must not touch removed np.trapz at import time."""
    code = textwrap.dedent(
        """
        import sys
        sys.frozen = True
        from scan_kit.common.matplotlib_backend import init_matplotlib_for_views
        init_matplotlib_for_views()
        from scan_kit.views import ic_hv_transient
        assert callable(ic_hv_transient.run)
        """
    )
    result = _run_import_script(code)
    assert result.returncode == 0, result.stderr or result.stdout


def test_tk_only_views_excluded_from_warm_worker_pool() -> None:
    assert TK_ONLY_VIEW_MODULES == frozenset({"ic_audio_export"})
