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

export type NoteEdit = { sessionId: string; before: string; after: string };

export type SessionMenu = { sessionId: string; x: number; y: number; rowIds: string[] };

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
  onCommitNote: (edit: NoteEdit) => void;
  onOpenMenu: (menu: SessionMenu) => void;
}) {
  const lastRowToggle = useRef<{ row: number; at: number } | null>(null);
  const columns = useMemo(() => libraryColumns(sort), [sort]);
  const changeSelection = useCallback(
    (next: GridSelection) => {
      onGridSelectionChange(selectWholeRows(next, columns.length));
    },
    [columns.length, onGridSelectionChange],
  );
  const { checked: headerChecked, indeterminate: headerMixed } = headerCheck(
    rows.length,
    selectedIds.length,
  );

  const onCellEdited = useCallback(
    (cell: Item, newValue: EditableGridCell) => {
      const [col, rowIndex] = cell;
      const source = order[rowIndex];
      const row = source == null ? undefined : rows[source];
      if (row == null || folder == null) {
        return;
      }
      if (clickEditsNote(col) && newValue.kind === GridCellKind.Text && newValue.data !== row.note) {
        onCommitNote({ sessionId: row.session_id, before: row.note, after: newValue.data });
      }
    },
    [folder, onCommitNote, order, rows],
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
          copyData: checked ? "true" : "false",
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
      onCellEdited={onCellEdited}
      onCellClicked={([col, rowIndex], event) => {
        const source = order[rowIndex];
        const row = source == null ? undefined : rows[source];
        if (row == null || clickEditsNote(col)) {
          return;
        }
        const recent = lastRowToggle.current;
        const justToggled =
          recent != null && recent.row === rowIndex && performance.now() - recent.at < 500;
        if (event.isDoubleClick) {
          event.preventDefault();
        }
        if (shouldToggleRow(col, event.isDoubleClick === true, justToggled)) {
          lastRowToggle.current = { row: rowIndex, at: performance.now() };
          onRowCheck(row.session_id, !selectedIds.includes(row.session_id));
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
    />
  );
}
