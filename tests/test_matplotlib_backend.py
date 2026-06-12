from __future__ import annotations

import sys

import pytest


def test_init_matplotlib_for_views_noop_when_not_frozen(monkeypatch) -> None:
    import scan_kit.common.matplotlib_backend as mod

    monkeypatch.setattr(mod, "_initialized", False)
    monkeypatch.setattr(mod.sys, "frozen", False, raising=False)
    mod.init_matplotlib_for_views()
    assert mod._initialized is True


def test_init_matplotlib_for_views_creates_qapp_and_selects_qtagg(
    monkeypatch,
) -> None:
    import matplotlib
    import scan_kit.common.matplotlib_backend as mod

    monkeypatch.setattr(mod, "_initialized", False)
    monkeypatch.setattr(mod.sys, "frozen", True, raising=False)
    monkeypatch.delenv("MPLBACKEND", raising=False)

    created: list[object] = []
    calls: list[tuple[str, bool]] = []

    class FakeApp:
        @staticmethod
        def instance():
            return None

        def __init__(self, _argv):
            created.append(self)

    monkeypatch.setattr("PySide6.QtWidgets.QApplication", FakeApp)
    monkeypatch.setattr(matplotlib, "use", lambda name, force=False: calls.append((name, force)))

    mod.init_matplotlib_for_views()

    assert len(created) == 1
    assert calls == [("QtAgg", True)]
