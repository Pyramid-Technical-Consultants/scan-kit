"""Smoke tests that heavy deps and their data bundles are present.

These catch missing PyInstaller hiddenimports / data bundles before a frozen
release (e.g. vispy GLSL shaders).
"""

from __future__ import annotations


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
