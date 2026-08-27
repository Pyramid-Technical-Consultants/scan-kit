"""Smoke tests that launcher view modules and heavy deps import cleanly.

These catch missing PyInstaller hiddenimports / data bundles before a frozen
release (e.g. vispy GLSL shaders, data registry sources).
"""

from __future__ import annotations

import importlib

import pytest

from scan_kit.data.registry import REGISTRY
from scan_kit.views import TK_ONLY_VIEW_MODULES, VIEWS, view_module_name


@pytest.mark.parametrize(
    "module_name",
    [
        view_module_name(entry)
        for entry in VIEWS
        if view_module_name(entry) not in TK_ONLY_VIEW_MODULES
    ],
)
def test_launcher_view_module_imports(module_name: str) -> None:
    mod = importlib.import_module(f"scan_kit.views.{module_name}")
    assert callable(mod.run)


def test_tk_only_view_module_imports() -> None:
    for module_name in TK_ONLY_VIEW_MODULES:
        mod = importlib.import_module(f"scan_kit.views.{module_name}")
        assert callable(mod.run)


def test_data_registry_registers_all_builtin_sources() -> None:
    import scan_kit.data  # noqa: F401 — populates REGISTRY

    expected = {
        "confidence",
        "current_ratio",
        "dose_rate",
        "gaussian_fit_filter",
        "ic12_pos_diff",
        "ic_current",
        "position",
        "position_error",
        "sigma",
        "sigma_error",
    }
    assert expected <= set(REGISTRY.keys())


def test_trajectory_vispy_import_chain() -> None:
    importlib.import_module("scan_kit.views.trajectory_vispy")
    from vispy import scene  # noqa: F401
    from vispy.visuals.line.line import _AggLineVisual  # noqa: F401


def test_vispy_glsl_tree_is_discoverable() -> None:
    from pathlib import Path

    import vispy

    glsl_dir = Path(vispy.__file__).resolve().parent / "glsl"
    assert glsl_dir.is_dir()
    assert any(glsl_dir.rglob("*.vert")) or any(glsl_dir.rglob("*.glsl"))


def test_pyinstaller_vispy_collect_all_includes_glsl() -> None:
    from PyInstaller.utils.hooks import collect_all

    datas, _, _ = collect_all("vispy")
    glsl_entries = [
        src
        for src, _dest in datas
        if "/glsl/" in src.replace("\\", "/")
    ]
    assert len(glsl_entries) >= 20
