import { useCallback, useMemo, useRef } from "react";
import {
  DataEditor,
  GridCellKind,
  type EditableGridCell,
  type GridCell,
  type GridColumn,
  type GridSelection,
  type Item,
  type Theme,
} from "@glideapps/glide-data-grid";

import { useResizableColumns } from "@/grid-theme";
import { selectedSessionIds, selectWholeRows } from "@/grid-selection";
import { sessionColor } from "@/session-colors";
import { drawSessionCheckbox, type CheckPaint } from "@/session-checkbox";
import { sessionMenuPoint } from "@/session-menu";
import { headerCheck, shouldToggleRow } from "@/session-checks";

export type SortKey =
  | "session_id"
  | "date"
  | "mu"
  | "extent"
  | "layers"
  | "time"
  | "room"
  | "config"
  | "note";

export type Sort = { key: SortKey; direction: "asc" | "desc" };

const COLUMN_SORT: Array<SortKey | null> = [
  null,
  "session_id",
  "date",
  "mu",
  "extent",
  "layers",
  "time",
  "room",
  "config",
  "note",
];

const COLUMN_TITLES = [
  "",
  "Session ID",
  "Date",
  "MU",
  "Ext.",
  "Lyr.",
  "Time",
  "RM",
  "Config",
  "Note",
];

/** Double-click selects a row. The note column edits in place instead. */
export function clickEditsNote(column: number): boolean {
  return COLUMN_SORT[column] === "note";
}

export type LibraryRow = {
  session_id: string;
  folder_id?: string;
  library?: string;
  storage_path: string;
  selected: boolean;
  note: string;
  date: string;
  date_iso: string | null;
  mu: string;
  mu_value: number | null;
  extent: string;
  extent_value: number | null;
  layers: string;
  layers_value: number | null;
  time: string;
  time_value: number | null;
  room: string;
  room_value: number | null;
  config: string;
};

export type NoteEdit = { sessionId: string; library: string; before: string; after: string };

export type SessionMenu = { sessionId: string; library: string; x: number; y: number; rowIds: string[] };

function paintedColor(
  className: string,
  property: "color" | "backgroundColor" | "borderTopColor",
): string {
  const probe = document.createElement("span");
  probe.className = className;
  document.body.append(probe);
  const resolved = getComputedStyle(probe)[property];
  probe.remove();
  return resolved;
}

export function checkPaint(): CheckPaint {
  return {
    border: paintedColor("border border-input", "borderTopColor"),
    idle: paintedColor("bg-input/30", "backgroundColor"),
    mark: paintedColor("text-primary-foreground", "color"),
    header: paintedColor("bg-primary", "backgroundColor"),
  };
}

function sortValue(row: LibraryRow, key: SortKey): string | number | null {
  switch (key) {
    case "date":
      return row.date_iso;
    case "mu":
      return row.mu_value;
    case "extent":
      return row.extent_value;
    case "layers":
      return row.layers_value;
    case "time":
      return row.time_value;
    case "room":
      return row.room_value;
    case "session_id":
      return row.session_id;
    case "config":
      return row.config;
    case "note":
      return row.note;
  }
}

export function compareRows(left: LibraryRow, right: LibraryRow, sort: Sort): number {
  const a = sortValue(left, sort.key);
  const b = sortValue(right, sort.key);
  if (a == null && b == null) {
    return 0;
  }
  if (a == null) {
    return 1;
  }
  if (b == null) {
    return -1;
  }
  const order =
    typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
  return sort.direction === "asc" ? order : -order;
}

function textCell(value: string, editable: boolean): GridCell {
  return {
    kind: GridCellKind.Text,
    data: value,
    displayData: value,
    allowOverlay: editable,
    readonly: !editable,
    copyData: value,
  };
}

function libraryColumns(sort: Sort): GridColumn[] {
  return COLUMN_TITLES.map((title, index) => {
    const key = COLUMN_SORT[index];
    const marked =
      key != null && sort.key === key ? `${title} ${sort.direction === "desc" ? "↓" : "↑"}` : title;
    return {
      title: marked,
      id: key ?? "use",
      width: index === 0 ? 48 : index === 8 || index === 9 ? 220 : 110,
      grow: index === 8 || index === 9 ? 1 : 0,
    };
  });
}

export function LibraryTable({
  rows,
  order,
  displayIds,
  sort,
  theme,
  width,
  height,
  checks,
  selectedIds,
  folder,
  gridSelection,
  onGridSelectionChange,
  onSort,
  onRowCheck,
  onHeaderCheck,
  onCommitNote,
  onOpenMenu,
  pendingNotes,
}: {
  rows: LibraryRow[];
  order: number[];
  displayIds: string[];
  sort: Sort;
  theme: Theme;
  width: number;
  height: number;
  checks: CheckPaint | null;
  selectedIds: string[];
  folder: string | null;
  gridSelection: GridSelection;
  onGridSelectionChange: (next: GridSelection) => void;
  onSort: (key: SortKey) => void;
  onRowCheck: (sessionId: string, checked: boolean) => void;
  onHeaderCheck: () => void;
  onCommitNote: (edit: NoteEdit) => Promise<boolean>;
  onOpenMenu: (menu: SessionMenu) => void;
  // Shared with the undo/redo writes in App — a note write from any path
  // stays pending until rows reflect it.
  pendingNotes: { current: Map<string, string> };
}) {
  const lastRowToggle = useRef<{ row: string; at: number } | null>(null);
  const baseColumns = useMemo(() => libraryColumns(sort), [sort]);
  const { columns, onColumnResize } = useResizableColumns(baseColumns);
  const changeSelection = useCallback(
    (next: GridSelection) => {
      onGridSelectionChange(selectWholeRows(next, columns.length));
    },
    [columns.length, onGridSelectionChange],
  );
  // Count only visible rows — a selection holding ids outside this list
  // must not paint the header fully checked.
  const selectedVisible = rows.filter((row) => selectedIds.includes(row.session_id)).length;
  const { checked: headerChecked, indeterminate: headerMixed } = headerCheck(
    rows.length,
    selectedVisible,
  );

  const noteEdit = useRef<{ cell: Item; id: string; note: string; library: string } | null>(null);
  // A committed note stays "after" until rows catch up with the write —
  // a second edit opened in that window must not capture the stale cell
  // as `before` (it would prefill the old text and corrupt the undo chain).
  // session_id is not unique across locations — the pending map keys on
    // the same composite noteEdit captures.
  const noteKey = (library: string, id: string) => `${library}\0${id}`;
  const onCellActivated = useCallback(
    (cell: Item) => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      // rows caught up after a commit — drop entries the catalog reflects.
      for (const [key, note] of pendingNotes.current) {
        const at = key.indexOf("\0");
        const current = rows.find(
          (entry) =>
            entry.session_id === key.slice(at + 1) &&
            (entry.library ?? folder ?? "") === key.slice(0, at),
        );
        // Only a row that reflects the write clears the entry — an absent
        // row is not proof the write landed, and dropping the entry here
        // reopens the stale-before hole the map exists to close.
        if (current != null && current.note === note) {
          pendingNotes.current.delete(key);
        }
      }
      // Capture the session and its note when the editor opens — an
      // in-flight catalog refresh can move a different session into this
      // row before commit, and undo restores `before` verbatim.
      const pending =
        row == null
          ? undefined
          : pendingNotes.current.get(noteKey(row.library ?? folder ?? "", row.session_id));
      noteEdit.current =
        clickEditsNote(col) && row != null
          ? {
              cell,
              id: row.session_id,
              note: pending ?? row.note,
              library: row.library ?? folder ?? "",
            }
          : null;
    },
    [folder, order, pendingNotes, rows],
  );
  const onCellEdited = useCallback(
    (cell: Item, newValue: EditableGridCell) => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      if (row == null || folder == null) {
        return;
      }
      if (col === 0 && newValue.kind === GridCellKind.Boolean) {
        // Keyboard toggling lands here — clicks are handled by onCellClicked.
        onRowCheck(row.session_id, newValue.data === true);
        return;
      }
      // The activation capture only covers the cell that was activated —
      // a bulk paste fires per target cell without activating each, so a
      // mismatched cell must fall back to its own row, not the capture.
      const edit =
        noteEdit.current?.cell[0] === col && noteEdit.current.cell[1] === rowIndex
          ? noteEdit.current
          : null;
      const before = edit?.note ?? row.note;
      if (clickEditsNote(col) && newValue.kind === GridCellKind.Text && newValue.data !== before) {
        const key = noteKey(
          edit?.library ?? row.library ?? folder ?? "",
          edit?.id ?? row.session_id,
        );
        pendingNotes.current.set(key, newValue.data);
        // A rejected write must not poison the entry — an identical retry
        // would compare against a note that never persisted.
        void onCommitNote({
          sessionId: edit?.id ?? row.session_id,
          library: edit?.library ?? row.library ?? folder ?? "",
          before,
          after: newValue.data,
        }).then((ok) => {
          // Only clear the marker this commit left — a newer pending
          // write to the same cell must survive a stale rejection.
          if (ok === false && pendingNotes.current.get(key) === newValue.data) {
            pendingNotes.current.delete(key);
          }
        });
      }
    },
    [folder, onCommitNote, onRowCheck, order, pendingNotes, rows],
  );

  const getCellContent = useCallback(
    (cell: Item): GridCell => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      if (row == null) {
        return textCell("", false);
      }
      if (col === 0) {
        const checked = selectedIds.includes(row.session_id);
        return {
          kind: GridCellKind.Boolean,
          data: checked,
          allowOverlay: false,
          readonly: true,
          copyData: "",
          cursor: "pointer",
        };
      }
      const value = [
        row.session_id,
        row.date,
        row.mu,
        row.extent,
        row.layers,
        row.time,
        row.room,
        row.config,
        row.note,
      ][col - 1];
      return textCell(value ?? "", clickEditsNote(col));
    },
    [order, rows, selectedIds],
  );

  const drawGridCell = useCallback(
    (
      args: {
        ctx: CanvasRenderingContext2D;
        rect: { x: number; y: number; width: number; height: number };
        col: number;
        row: number;
      },
      drawContent: () => void,
    ) => {
      if (args.col !== 0 || checks == null) {
        drawContent();
        return;
      }
      const id = displayIds[args.row];
      const on = id != null && selectedIds.includes(id);
      drawSessionCheckbox(
        args.ctx,
        args.rect,
        on ? "on" : "off",
        on && id != null ? sessionColor(selectedIds.indexOf(id)) : checks.header,
        on ? { ...checks, mark: "#fff" } : checks,
      );
    },
    [checks, displayIds, selectedIds],
  );

  const drawGridHeader = useCallback(
    (
      args: {
        ctx: CanvasRenderingContext2D;
        columnIndex: number;
        rect: { x: number; y: number; width: number; height: number };
      },
      drawContent: () => void,
    ) => {
      drawContent();
      if (args.columnIndex !== 0 || checks == null) {
        return;
      }
      const mode = headerMixed ? "mixed" : headerChecked ? "on" : "off";
      drawSessionCheckbox(args.ctx, args.rect, mode, checks.header, checks);
    },
    [checks, headerChecked, headerMixed],
  );

  return (
    <DataEditor
      width={width}
      height={height}
      columns={columns}
      rows={rows.length}
      rowHeight={28}
      headerHeight={28}
      getCellContent={getCellContent}
      drawCell={drawGridCell}
      drawHeader={drawGridHeader}
      onCellActivated={onCellActivated}
      onCellEdited={onCellEdited}
      onCellClicked={([col, rowIndex], event) => {
        const source = order[rowIndex];
        const row = source == null ? undefined : rows[source];
        if (row == null || clickEditsNote(col)) {
          return;
        }
        const recent = lastRowToggle.current;
        // A re-sorted table reuses the index for a different row — key on the
        // session id so a different session's double-click is not eaten.
        const justToggled =
          recent != null &&
          recent.row === row.session_id &&
          performance.now() - recent.at < 500;
        if (event.isDoubleClick) {
          event.preventDefault();
        }
        if (shouldToggleRow(col, event.isDoubleClick === true, justToggled)) {
          lastRowToggle.current = { row: row.session_id, at: performance.now() };
          onRowCheck(row.session_id, !selectedIds.includes(row.session_id));
        } else if (event.isDoubleClick !== true) {
          // A non-toggling click provably opens a new gesture — the next
          // dblclick tail must not be deduped against the stale record.
          lastRowToggle.current = null;
        }
      }}
      onCellContextMenu={([, rowIndex], event) => {
        event.preventDefault();
        const source = order[rowIndex];
        const row = source == null ? undefined : rows[source];
        if (row == null) {
          return;
        }
        const point = sessionMenuPoint(event.bounds, event.localEventX, event.localEventY);
        if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
          return;
        }
        onOpenMenu({
          sessionId: row.session_id,
          library: row.library ?? folder ?? "",
          x: point.x,
          y: point.y,
          rowIds: selectedSessionIds(displayIds, gridSelection, rowIndex),
        });
      }}
      onHeaderClicked={(col) => {
        if (col === 0) {
          onHeaderCheck();
          return;
        }
        const key = COLUMN_SORT[col];
        if (key != null) {
          onSort(key);
        }
      }}
      theme={theme}
      gridSelection={gridSelection}
      onGridSelectionChange={changeSelection}
      columnSelect="none"
      rowMarkers="none"
      smoothScrollX
      smoothScrollY
      onColumnResize={onColumnResize}
    />
  );
}
