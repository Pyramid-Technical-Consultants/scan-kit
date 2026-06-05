from pathlib import Path

path = Path("scan_kit/qt_launcher.py")
text = path.read_text(encoding="utf-8")
start = text.index("    def _refresh_sessions(self) -> None:")
end = text.index("    def _notify(self, message: str, *, error: bool = False) -> None:")
replacement = """    def _refresh_sessions(self) -> None:
        if self._session_browser is None:
            return
        self._session_browser.refresh(
            restored_selection=list(self._settings.selected_sessions or [])[:MAX_SESSIONS],
        )

    def _selected_sids_in_order(self) -> list[str]:
        if self._session_browser is None:
            return []
        return self._session_browser.selected_session_ids()

    def _persist_selected_sessions(self, session_ids: list[str] | None = None) -> None:
        \"\"\"Save the current session selection into the persistent settings file.\"\"\"
        if session_ids is None:
            session_ids = self._selected_sids_in_order()
        self._settings.selected_sessions = session_ids
        try:
            self._settings.save(self._base_dir)
        except Exception:
            pass

    def _session_meta_by_sid(self) -> dict[str, SessionMeta | None]:
        if self._session_browser is None:
            return {}
        return self._session_browser.session_meta_by_id()

"""
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
