"""Textual TUI for scan-kit analysis views."""

import os
import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, SelectionList, Static
from textual.widgets.selection_list import Selection

from .common.sessions import discover_sessions
from .views import VIEWS

MAX_SESSIONS = 3
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ScanKitApp(App[None]):
    """Scan-kit analysis launcher."""

    CSS = """
    Screen { align: center middle; background: #0a0e14 }
    #main { width: 98%; height: 100%; padding: 0 }

    #left-panel, #right-panel {
        height: 100%; padding: 0; background: #0d1117;
    }
    #left-panel { width: 38% }
    #right-panel { width: 62% }
    #right-panel > Static { padding: 0; height: 1 }

    #session-label, #base-dir-section, #views-label, #status {
        color: #00d4aa; padding: 0;
    }
    #session-label { height: 1 }
    #status { border: round #00d4aa; background: #0a0e14; min-height: 1 }

    #session-list {
        height: 1fr; border: round #00d4aa; padding: 1; background: #0a0e14;
        scrollbar-color: #00d4aa #0d1117;
    }
    #session-list:focus { border: round #00d4ff }
    #session-list .selection-list--button { min-height: 2 }
    #session-list .selection-list--button-selected {
        color: #00d4ff; background: #001a1f;
    }
    #session-list .selection-list--button-highlighted { background: #001a1f }

    #base-dir-input {
        background: #0a0e14; color: #00d4ff; border: round #00d4aa;
        padding: 0 1; height: 3;
    }
    #base-dir-input:focus { border: round #00d4ff }

    #buttons-section {
        height: 1fr; overflow-y: auto; padding: 0;
        scrollbar-color: #00d4aa #00d4aa;
    }

    .view-button {
        width: 100%; background: #0a0e14; color: #00d4aa;
        border: round #00d4aa; padding: 0 1; height: 3;
    }
    .view-button:hover { background: #0d1117; color: #00d4ff; border: round #00d4ff }
    .view-button:focus { background: #001a1f; color: #00d4ff; border: round #00d4ff }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._base_dir = str(PROJECT_ROOT / "test_data")
        self._sessions: list[str] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Static("DATA SOURCE", id="base-dir-section")
                yield Input(
                    placeholder="Path to session ZIPs...",
                    value=str(PROJECT_ROOT / "test_data"),
                    id="base-dir-input",
                )
                yield Static("SELECT UP TO 3", id="session-label")
                yield SelectionList[str](id="session-list")
                yield Static("", id="status")
            with VerticalScroll(id="right-panel"):
                yield Static("RUN ANALYSIS", id="views-label")
                with Vertical(id="buttons-section"):
                    for display_name, _module_name, _run in VIEWS:
                        yield Button(
                            display_name,
                            id=f"view-{_module_name}",
                            classes="view-button",
                        )

    def on_mount(self) -> None:
        self._refresh_sessions()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle base directory path change."""
        if event.input.id == "base-dir-input":
            path = event.value.strip()
            if path:
                self._base_dir = path
                self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        """Refresh session list from disk."""
        self._sessions = discover_sessions(
            base_dirs=(self._base_dir,),
            project_root=PROJECT_ROOT,
        )
        session_list = self.query_one("#session-list", SelectionList)
        session_list.clear_options()
        for sid in self._sessions:
            session_list.add_option(Selection(sid, sid))
        self._update_status()

    def _update_status(self) -> None:
        """Update status bar with selection count."""
        session_list = self.query_one("#session-list", SelectionList)
        selected = session_list.selected
        count = len(selected)
        status = self.query_one("#status", Static)
        if count == 0:
            status.update("> SELECT 1-3 SESSIONS")
        else:
            status.update(f"> READY: {count} | {', '.join(selected)}")

    def _enforce_max_sessions(self) -> None:
        """Deselect oldest if more than MAX_SESSIONS selected."""
        session_list = self.query_one("#session-list", SelectionList)
        selected = session_list.selected
        while len(selected) > MAX_SESSIONS:
            session_list.deselect(selected[0])
            selected = session_list.selected

    def on_selection_list_selected_changed(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        """Handle selection change; enforce max 3 sessions."""
        self._enforce_max_sessions()
        self._update_status()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run the selected analysis view."""
        button_id = event.button.id
        if not button_id or not button_id.startswith("view-"):
            return

        module_name = button_id.removeprefix("view-")
        session_list = self.query_one("#session-list", SelectionList)
        selected = session_list.selected

        if not selected:
            self.notify("Select 1-3 sessions first", severity="warning")
            return

        session_ids = list(selected)[:MAX_SESSIONS]
        base_dir = self._base_dir

        code = (
            f"from scan_kit.views.{module_name} import run; "
            f"run({session_ids!r}, {base_dir!r})"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (
            os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )

        try:
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=PROJECT_ROOT,
                env=env,
            )
        except Exception as e:
            self.notify(f"Failed to run analysis: {e}", severity="error")
        else:
            self.notify("Analysis closed", severity="information")


def main() -> None:
    """Entry point for the scan-kit TUI."""
    app = ScanKitApp()
    app.run()


if __name__ == "__main__":
    main()
