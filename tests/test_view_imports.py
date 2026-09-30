"""Smoke tests that launcher view modules and heavy deps import cleanly.

These catch missing PyInstaller hiddenimports / data bundles before a frozen
release (e.g. vispy GLSL shaders, data registry sources).
"""

from __future__ import annotations

from scan_kit.data.registry import REGISTRY


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
