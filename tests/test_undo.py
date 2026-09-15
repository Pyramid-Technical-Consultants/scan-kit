"""Tests for the Qt-based session-note undo/redo support."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from scan_kit.common.session_browser import (
    _COL_CONFIG,
    _COL_NOTE,
    _COL_USE,
    SessionBrowserWidget,
)
from scan_kit.common.session_meta import SessionMeta
from scan_kit.common.user_store import notes_for_library


def _make_widget(tmp_path) -> SessionBrowserWidget:
    widget = SessionBrowserWidget(initial_base_dir=str(tmp_path), editable_notes=True)
    widget._table.setRowCount(1)
    widget._set_session_row_widgets(0, "S1", None, use_checked=False)
    widget._rebuild_session_row_index()
    return widget


def test_session_note_edit_is_undoable(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        note_cell = widget._table.item(0, _COL_NOTE)

        note_cell.setText("important note")
        assert widget.notes()["S1"] == "important note"
        assert notes_for_library(tmp_path)["S1"] == "important note"

        note_cell.setText("clobbered")
        assert widget.notes()["S1"] == "clobbered"

        assert widget.undo() is True
        assert widget.notes()["S1"] == "important note"
        assert note_cell.text() == "important note"
        assert notes_for_library(tmp_path)["S1"] == "important note"

        assert widget.redo() is True
        assert widget.notes()["S1"] == "clobbered"
        assert note_cell.text() == "clobbered"
    finally:
        widget.shutdown()


def test_session_note_undo_restores_empty(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        note_cell = widget._table.item(0, _COL_NOTE)

        note_cell.setText("a note")
        assert "S1" in widget.notes()

        assert widget.undo() is True
        assert "S1" not in widget.notes()
        assert note_cell.text() == ""
        assert "S1" not in notes_for_library(tmp_path)
    finally:
        widget.shutdown()


def test_undo_redo_noop_when_empty(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        assert widget.undo() is False
        assert widget.redo() is False
    finally:
        widget.shutdown()


def test_each_commit_is_a_separate_undo_step(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        note_cell = widget._table.item(0, _COL_NOTE)
        note_cell.setText("first")
        note_cell.setText("second")
        note_cell.setText("third")

        assert widget.undo() is True and note_cell.text() == "second"
        assert widget.undo() is True and note_cell.text() == "first"
        assert widget.undo() is True and note_cell.text() == ""
        assert widget.undo() is False
    finally:
        widget.shutdown()


def test_refresh_clears_undo_history(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        widget._table.item(0, _COL_NOTE).setText("note")
        assert widget._undo_stack.canUndo()

        widget.refresh()
        assert not widget._undo_stack.canUndo()
        assert widget.undo() is False
    finally:
        widget.shutdown()


def test_undo_actions_exposed_with_dynamic_text(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        undo_action = widget.undo_action()
        redo_action = widget.redo_action()
        assert undo_action is not None and redo_action is not None
        assert not undo_action.isEnabled()

        widget._table.item(0, _COL_NOTE).setText("hello")
        assert undo_action.isEnabled()
        assert "edit note for S1" in undo_action.text()
    finally:
        widget.shutdown()


def test_actions_absent_when_notes_readonly(qapp, tmp_path) -> None:
    widget = SessionBrowserWidget(initial_base_dir=str(tmp_path), editable_notes=False)
    try:
        assert widget.undo_action() is None
        assert widget.redo_action() is None
    finally:
        widget.shutdown()


def test_config_column_after_room_elides_with_tooltip(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        headers = [
            widget._table.horizontalHeaderItem(i).text()
            for i in range(widget._table.columnCount())
        ]
        assert headers == [
            "Use", "Session ID", "Date", "MU", "Time", "RM", "Config", "Note",
        ]
        assert widget._table.textElideMode() == Qt.TextElideMode.ElideRight

        name = "working_hvtt_new_cal_gates_tuned"
        meta = SessionMeta(
            date=None,
            primary_mu=None,
            treatment_time_s=None,
            room_number=1,
            config_name=name,
        )
        widget._set_session_row_widgets(0, "S1", meta, use_checked=False)
        cell = widget._table.item(0, _COL_CONFIG)
        assert cell is not None
        assert cell.text() == name
        assert cell.toolTip() == name
    finally:
        widget.shutdown()


def test_use_column_width_stable_when_table_clears(qapp, tmp_path) -> None:
    widget = _make_widget(tmp_path)
    try:
        hh = widget._table.horizontalHeader()
        widget._refresh_use_column_swatches(["S1"])
        before = hh.sectionSize(_COL_USE)
        widget._table.setRowCount(0)
        assert hh.sectionSize(_COL_USE) == before
    finally:
        widget.shutdown()
