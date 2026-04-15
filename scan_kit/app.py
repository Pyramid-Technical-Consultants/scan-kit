"""Textual TUI for scan-kit analysis views."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, SelectionList, Static
from textual.widgets.selection_list import Selection

from .common import SessionMeta
from .common.session_notes import load_notes, save_note
from .common.session_source import resolve_session_source, load_session_termination_summary
from .common.sessions import discover_sessions
from .views import VIEWS

MAX_SESSIONS = 3
FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SORT_MODES = ["date", "id", "mu"]
_SORT_LABELS = {
    "date": "Date",
    "id": "ID",
    "mu": "MU",
}

_EPOCH = datetime(1970, 1, 1)
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_READY_SENTINEL = "__SCAN_KIT_PLOT_READY__"


def _sort_key(
    item: tuple[str, str, SessionMeta | None], mode: str
) -> tuple:
    sid, _zp, meta = item
    if mode == "date":
        return (meta.date if meta and meta.date else _EPOCH,)
    if mode == "mu":
        return (meta.primary_mu if meta and meta.primary_mu is not None else 0.0,)
    return (sid,)


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

    #base-dir-section, #views-label, #status {
        color: #00d4aa; padding: 0;
    }
    #status-row { height: 3; width: 100% }
    #status { border: round #00d4aa; background: #0a0e14; min-height: 1; width: 1fr }

    #clear-btn {
        width: 5; min-width: 5; height: 3;
        background: #0a0e14; color: #aa0030; border: round #aa0030; padding: 0;
    }
    #clear-btn:hover { color: #ff0040; border: round #ff0040; background: #1a0008 }

    #session-list {
        height: 1fr; border: round #00d4aa; padding: 1; background: #0a0e14;
        scrollbar-color: #00d4aa 30%;
        scrollbar-color-active: #00d4ff;
        scrollbar-color-hover: #00d4ff;
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

    #sort-row { height: 3; width: 100% }

    .sort-btn {
        width: 1fr; background: #0a0e14; color: #555;
        border: round #333; padding: 0 1; height: 3; min-width: 6;
    }
    .sort-btn:hover { color: #00d4ff; border: round #00d4ff }
    .sort-btn.active { color: #00d4ff; background: #001a1f; border: round #00d4ff }

    #buttons-section {
        height: 1fr; overflow-y: auto; padding: 0;
        scrollbar-color: #00d4aa 30%;
        scrollbar-color-active: #00d4ff;
        scrollbar-color-hover: #00d4ff;
    }

    .view-button {
        width: 100%; background: #0a0e14; color: #00d4aa;
        border: round #00d4aa; padding: 0 1; height: 3;
    }
    .view-button:hover { background: #0d1117; color: #00d4ff; border: round #00d4ff }
    .view-button:focus { background: #001a1f; color: #00d4ff; border: round #00d4ff }
    .view-button.loading { color: #ffaa00; border: round #ffaa00; background: #1a1200 }
    .view-button.loading:hover { color: #ffaa00; border: round #ffaa00 }

    #note-input {
        background: #0a0e14; color: #00d4ff; border: round #00d4aa;
        padding: 0 1; height: 3;
    }
    #note-input:focus { border: round #00d4ff }
    """

    BINDINGS = [
        Binding("escape", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        if FROZEN:
            self._base_dir = str(Path.cwd())
        else:
            self._base_dir = str(PROJECT_ROOT / "test_data")
        self._sessions: list[str] = []
        self._discovered: list[tuple[str, str, SessionMeta | None]] = []
        self._sort_mode: str = "date"
        self._hydrate_generation: int = 0
        self._meta_loading: bool = False
        self._child_procs: list[subprocess.Popen] = []
        self._running_views: dict[str, tuple[subprocess.Popen, str]] = {}
        self._spinner_frame: int = 0
        self._poll_timer = None
        self._notes: dict[str, str] = {}
        self._highlighted_sid: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Static("DATA SOURCE", id="base-dir-section")
                yield Input(
                    placeholder="Path to session ZIPs...",
                    value=self._base_dir,
                    id="base-dir-input",
                )
                with Horizontal(id="sort-row"):
                    for mode in _SORT_MODES:
                        btn = Button(
                            _SORT_LABELS[mode],
                            id=f"sort-{mode}",
                            classes="sort-btn",
                        )
                        if mode == self._sort_mode:
                            btn.add_class("active")
                        yield btn
                yield SelectionList[str](id="session-list")
                with Horizontal(id="status-row"):
                    yield Static("", id="status")
                    yield Button("✕", id="clear-btn")
                yield Input(
                    placeholder="Highlight a session to edit its note…",
                    id="note-input",
                )
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

    def _set_sort(self, mode: str) -> None:
        """Switch to the given sort mode and update button highlights."""
        self._sort_mode = mode
        for m in _SORT_MODES:
            btn = self.query_one(f"#sort-{m}", Button)
            if m == mode:
                btn.add_class("active")
            else:
                btn.remove_class("active")
        self._repopulate_list()

    def _refresh_sessions(self) -> None:
        """Refresh session list from disk (fast), then load metadata in background."""
        self._notes = load_notes(self._base_dir)
        self._hydrate_generation += 1
        gen = self._hydrate_generation
        self._discovered = discover_sessions(
            base_dirs=(self._base_dir,),
            project_root=PROJECT_ROOT,
        )
        self._meta_loading = bool(self._discovered)
        self._repopulate_list()
        if not self._discovered:
            self._meta_loading = False
            self._update_status()
            return
        snapshot = list(self._discovered)
        base_dir = self._base_dir
        threading.Thread(
            target=self._hydrate_metadata_worker,
            args=(gen, snapshot, base_dir),
            daemon=True,
        ).start()

    def _hydrate_metadata_worker(
        self,
        gen: int,
        snapshot: list[tuple[str, str, SessionMeta | None]],
        base_dir: str,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        base = Path(base_dir)
        n = len(snapshot)
        max_workers = max(4, min(24, n, (os.cpu_count() or 4) * 3))

        def _load_one(idx: int, row: tuple[str, str, SessionMeta | None]):
            sid, path_str, _ = row

            def _on_extracting(session_id: str) -> None:
                self.call_from_thread(
                    self._show_extracting_status, gen, session_id,
                )

            src = resolve_session_source(sid, base, on_extracting=_on_extracting)
            meta = load_session_termination_summary(src) if src else None
            return idx, sid, path_str, meta

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_load_one, i, row): i
                for i, row in enumerate(snapshot)
            }
            for future in as_completed(futures):
                if gen != self._hydrate_generation:
                    return
                idx, sid, path_str, meta = future.result()
                self.call_from_thread(
                    self._apply_single_metadata, gen, idx, sid, path_str, meta,
                )

        self.call_from_thread(self._finish_hydration, gen)

    def _apply_single_metadata(
        self,
        gen: int,
        idx: int,
        sid: str,
        path_str: str,
        meta: SessionMeta | None,
    ) -> None:
        if gen != self._hydrate_generation:
            return
        for i, (s, p, _) in enumerate(self._discovered):
            if s == sid:
                self._discovered[i] = (sid, path_str, meta)
                break
        self._repopulate_list()

    def _show_extracting_status(self, gen: int, session_id: str) -> None:
        if gen != self._hydrate_generation:
            return
        status = self.query_one("#status", Static)
        status.update(f"> Extracting {session_id}… (one-time)")

    def _finish_hydration(self, gen: int) -> None:
        if gen != self._hydrate_generation:
            return
        self._meta_loading = False
        self._update_status()

    def _repopulate_list(self) -> None:
        """Sort and display sessions using the current sort mode."""
        session_list = self.query_one("#session-list", SelectionList)
        prev_selected = list(session_list.selected)
        prev_scroll_y = session_list.scroll_y
        prev_highlighted = session_list.highlighted

        reverse = self._sort_mode == "date"
        ordered = sorted(self._discovered, key=lambda t: _sort_key(t, self._sort_mode), reverse=reverse)
        self._sessions = [sid for sid, _zp, _meta in ordered]

        sid_w = max((len(s) for s, _, _ in ordered), default=0)
        mu_w = max(
            (len(m.short_mu) for _, _, m in ordered if m is not None),
            default=1,
        )
        time_w = max(
            (len(m.short_time) for _, _, m in ordered if m is not None),
            default=1,
        )

        session_list.clear_options()
        for sid, _zp, meta in ordered:
            if meta is not None:
                label = (
                    f"{sid:<{sid_w}}  "
                    f"{meta.short_date}  "
                    f"{meta.short_mu:>{mu_w}} MU  "
                    f"{meta.short_time:>{time_w}}s"
                )
            else:
                label = sid
            note = self._notes.get(sid, "")
            if note:
                preview = note if len(note) <= 30 else note[:28] + "…"
                label += f"  | {preview}"
            session_list.add_option(Selection(label, sid))
        for sid in prev_selected:
            if sid in self._sessions:
                try:
                    session_list.select(sid)
                except Exception:
                    pass
        if prev_highlighted is not None:
            try:
                session_list.highlighted = prev_highlighted
            except Exception:
                pass
        session_list.scroll_target_y = prev_scroll_y
        self._update_status()

    def _update_status(self) -> None:
        """Update status bar with selection count."""
        session_list = self.query_one("#session-list", SelectionList)
        selected = session_list.selected
        count = len(selected)
        status = self.query_one("#status", Static)
        extra = "  (loading session details…)" if self._meta_loading else ""
        if count == 0:
            status.update(f"> SELECT 1-3 SESSIONS{extra}")
        else:
            status.update(f"> READY: {count} | {', '.join(selected)}{extra}")

    def _clear_selection(self) -> None:
        """Deselect all sessions."""
        session_list = self.query_one("#session-list", SelectionList)
        for sid in list(session_list.selected):
            session_list.deselect(sid)
        self._update_status()

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

    def on_selection_list_selection_highlighted(
        self, event: SelectionList.SelectionHighlighted
    ) -> None:
        """Populate the note input when the user highlights a session."""
        sid = str(event.selection.value)
        self._highlighted_sid = sid
        note_input = self.query_one("#note-input", Input)
        note_input.value = self._notes.get(sid, "")
        note_input.placeholder = f"Note for {sid}…"

    def on_input_changed(self, event: Input.Changed) -> None:
        """Auto-save note as the user types."""
        if event.input.id != "note-input":
            return
        sid = self._highlighted_sid
        if sid is None:
            return
        current = self._notes.get(sid, "")
        if event.value == current:
            return
        text = event.value
        if text.strip():
            self._notes[sid] = text
        else:
            self._notes.pop(sid, None)
        save_note(self._base_dir, sid, event.value)
        self._refresh_session_label(sid)

    def _refresh_session_label(self, sid: str) -> None:
        """Update one session's label in-place to reflect a note change."""
        self._repopulate_list()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses (sort buttons + analysis views)."""
        button_id = event.button.id
        if button_id == "clear-btn":
            self._clear_selection()
            return
        if button_id and button_id.startswith("sort-"):
            mode = button_id.removeprefix("sort-")
            if mode in _SORT_MODES:
                self._set_sort(mode)
            return
        if not button_id or not button_id.startswith("view-"):
            return

        module_name = button_id.removeprefix("view-")

        if module_name in self._running_views:
            self.notify("Already running", severity="warning")
            return

        session_list = self.query_one("#session-list", SelectionList)
        selected = session_list.selected

        if not selected:
            self.notify("Select 1-3 sessions first", severity="warning")
            return

        session_ids = list(selected)[:MAX_SESSIONS]
        base_dir = self._base_dir

        btn = event.button
        original_label = str(btn.label)
        env = os.environ.copy()

        if FROZEN:
            cmd = [
                sys.executable,
                "--run-view", module_name,
                "--sessions", ",".join(session_ids),
                "--base-dir", base_dir,
            ]
        else:
            code = (
                "import matplotlib.pyplot as plt\n"
                "_real_show = plt.show\n"
                "def _show(*a, **kw):\n"
                "    for mgr in plt.get_fignums():\n"
                "        try: plt.figure(mgr).canvas.manager.window.state('zoomed')\n"
                "        except Exception:\n"
                "            try: plt.figure(mgr).canvas.manager.window.showMaximized()\n"
                "            except Exception: pass\n"
                f"    print('{_READY_SENTINEL}', flush=True); _real_show(*a, **kw)\n"
                "plt.show = _show\n"
                f"from scan_kit.views.{module_name} import run\n"
                f"run({session_ids!r}, {base_dir!r})"
            )
            env["PYTHONPATH"] = str(PROJECT_ROOT) + (
                os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
            )
            cmd = [sys.executable, "-c", code]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
            )
            self._child_procs.append(proc)
            self._reap_children()
        except Exception as e:
            self.notify(f"Failed to run analysis: {e}", severity="error")
            return

        self._running_views[module_name] = (proc, original_label)
        btn.add_class("loading")
        self._update_spinner()
        if self._poll_timer is None:
            self._poll_timer = self.set_interval(0.12, self._poll_running_views)

        threading.Thread(
            target=self._watch_subprocess_ready,
            args=(proc, module_name),
            daemon=True,
        ).start()

    def _update_spinner(self) -> None:
        """Update spinner character on all loading buttons."""
        frame = _SPINNER[self._spinner_frame % len(_SPINNER)]
        for module_name, (proc, original_label) in self._running_views.items():
            try:
                btn = self.query_one(f"#view-{module_name}", Button)
                btn.label = f"{frame} {original_label}"
            except Exception:
                pass

    def _watch_subprocess_ready(
        self, proc: subprocess.Popen, module_name: str
    ) -> None:
        """Read subprocess stdout in a background thread for the ready sentinel."""
        try:
            for line in proc.stdout:
                if _READY_SENTINEL.encode() in line:
                    self.call_from_thread(self._mark_view_ready, module_name)
                    break
        except Exception:
            pass
        try:
            for _ in proc.stdout:
                pass
        except Exception:
            pass

    def _mark_view_ready(self, module_name: str) -> None:
        """Called on the UI thread when a view's plot window has appeared."""
        if module_name not in self._running_views:
            return
        _proc, original_label = self._running_views.pop(module_name)
        try:
            btn = self.query_one(f"#view-{module_name}", Button)
            btn.label = original_label
            btn.remove_class("loading")
        except Exception:
            pass
        if not self._running_views:
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None

    def _poll_running_views(self) -> None:
        """Check for finished subprocesses and update button states."""
        self._spinner_frame += 1
        finished = [
            name for name, (proc, _) in self._running_views.items()
            if proc.poll() is not None
        ]
        for module_name in finished:
            proc, original_label = self._running_views.pop(module_name)
            try:
                btn = self.query_one(f"#view-{module_name}", Button)
                btn.label = original_label
                btn.remove_class("loading")
            except Exception:
                pass
            if proc.returncode != 0:
                self.notify(f"{module_name} exited with error", severity="error")

        if self._running_views:
            self._update_spinner()
        else:
            if self._poll_timer is not None:
                self._poll_timer.stop()
                self._poll_timer = None
        self._reap_children()

    def _reap_children(self) -> None:
        """Remove finished processes from the tracking list."""
        self._child_procs = [p for p in self._child_procs if p.poll() is None]

    def action_quit(self) -> None:
        """Terminate child analysis processes, then quit."""
        self._hydrate_generation += 1
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        self._running_views.clear()
        for proc in self._child_procs:
            try:
                proc.kill()
            except Exception:
                pass
        self._child_procs.clear()
        self.exit()


def _run_view_subprocess(module_name: str, session_ids: list[str], base_dir: str) -> None:
    """Execute a single analysis view (used by frozen exe in --run-view mode)."""
    import importlib
    import matplotlib.pyplot as plt

    _real_show = plt.show

    def _patched_show(*a, **kw):
        for mgr in plt.get_fignums():
            try:
                plt.figure(mgr).canvas.manager.window.state("zoomed")
            except Exception:
                try:
                    plt.figure(mgr).canvas.manager.window.showMaximized()
                except Exception:
                    pass
        print(_READY_SENTINEL, flush=True)
        _real_show(*a, **kw)

    plt.show = _patched_show

    mod = importlib.import_module(f"scan_kit.views.{module_name}")
    mod.run(session_ids, base_dir)


def main() -> None:
    """Entry point for the scan-kit TUI."""
    multiprocessing.freeze_support()

    if "--run-view" in sys.argv:
        idx = sys.argv.index("--run-view")
        module_name = sys.argv[idx + 1]
        sessions_idx = sys.argv.index("--sessions")
        session_ids = sys.argv[sessions_idx + 1].split(",")
        base_idx = sys.argv.index("--base-dir")
        base_dir = sys.argv[base_idx + 1]
        _run_view_subprocess(module_name, session_ids, base_dir)
        return

    app = ScanKitApp()

    def _force_exit(*_args) -> None:
        for proc in app._child_procs:
            try:
                proc.kill()
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGINT, _force_exit)
    atexit.register(_force_exit)

    app.run()


if __name__ == "__main__":
    main()
