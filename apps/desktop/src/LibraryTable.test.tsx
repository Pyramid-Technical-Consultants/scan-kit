import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import type { GridCell, GridSelection, Theme } from "@glideapps/glide-data-grid";

import {
  checkPaint,
  clickEditsNote,
  compareRows,
  LibraryTable,
  type LibraryRow,
  type SortKey,
} from "./LibraryTable";

type EditorProps = {
  getCellContent: (cell: readonly [number, number]) => GridCell;
  onHeaderClicked: (column: number) => void;
  onCellClicked: (cell: readonly [number, number], event: { isDoubleClick?: boolean; preventDefault: () => void }) => void;
  onCellContextMenu: (
    cell: readonly [number, number],
    event: {
      preventDefault: () => void;
      bounds: { x: number; y: number };
      localEventX: number;
      localEventY: number;
    },
  ) => void;
  onCellEdited: (cell: readonly [number, number], value: { kind: string; data: string }) => void;
  onGridSelectionChange: (selection: GridSelection) => void;
  drawCell: (
    args: {
      ctx: CanvasRenderingContext2D;
      rect: { x: number; y: number; width: number; height: number };
      col: number;
      row: number;
    },
    drawContent: () => void,
  ) => void;
  drawHeader: (
    args: {
      ctx: CanvasRenderingContext2D;
      columnIndex: number;
      rect: { x: number; y: number; width: number; height: number };
    },
    drawContent: () => void,
  ) => void;
};

let editor: EditorProps | null = null;

vi.mock("@glideapps/glide-data-grid", () => ({
  DataEditor: (props: EditorProps) => {
    editor = props;
    return <div data-grid="" />;
  },
  GridCellKind: { Text: "text", Boolean: "boolean" },
}));

function row(id: string, extra: Partial<LibraryRow> = {}): LibraryRow {
  return {
    session_id: id,
    storage_path: "",
    selected: false,
    note: "note",
    date: "date",
    date_iso: "2020-01-02",
    mu: "1",
    mu_value: 1,
    extent: "2",
    extent_value: 2,
    layers: "3",
    layers_value: 3,
    time: "4",
    time_value: 4,
    room: "5",
    room_value: 5,
    config: "cfg",
    ...extra,
  };
}

const paint = {
  border: "#111",
  idle: "#222",
  mark: "#fff",
  header: "#333",
};

const selection = { current: null, columns: { items: [] }, rows: { items: [] } } as unknown as GridSelection;

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  editor = null;
  document.body.replaceChildren();
});

it("orders every column and keeps nulls last", () => {
  const left = row("a", {
    date_iso: "2020-01-01",
    mu_value: 1,
    extent_value: 1,
    layers_value: 1,
    time_value: 1,
    room_value: 1,
    config: "a",
    note: "a",
  });
  const right = row("b", {
    date_iso: "2020-01-02",
    mu_value: 2,
    extent_value: 2,
    layers_value: 2,
    time_value: 2,
    room_value: 2,
    config: "b",
    note: "b",
  });
  const keys: SortKey[] = ["session_id", "date", "mu", "extent", "layers", "time", "room", "config", "note"];
  for (const key of keys) {
    expect(compareRows(left, right, { key, direction: "asc" })).toBeLessThan(0);
    expect(compareRows(right, left, { key, direction: "desc" })).toBeLessThan(0);
  }
  const blank = row("a", { date_iso: null, mu_value: null });
  expect(compareRows(blank, blank, { key: "date", direction: "asc" })).toBe(0);
  expect(compareRows(blank, row("b"), { key: "date", direction: "asc" })).toBeGreaterThan(0);
  expect(compareRows(row("b"), blank, { key: "mu", direction: "asc" })).toBeLessThan(0);
  expect(clickEditsNote(9)).toBe(true);
  expect(clickEditsNote(1)).toBe(false);
  expect(checkPaint().mark).toBeTypeOf("string");
});

it("draws checks and forwards clicks, notes, and the menu", async () => {
  const checks: string[] = [];
  const notes: string[] = [];
  const menus: number[] = [];
  const sorts: string[] = [];
  let header = 0;
  const host = document.createElement("div");
  document.body.append(host);
  const ctx = {
    beginPath() {},
    roundRect() {},
    fill() {},
    stroke() {},
    moveTo() {},
    lineTo() {},
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    lineJoin: "miter",
  } as unknown as CanvasRenderingContext2D;
  const rows = [row("s1"), row("s2", { note: "other" })];
  await act(async () => {
    root = createRoot(host);
    root.render(
      <LibraryTable
        rows={rows}
        order={[1, 0]}
        displayIds={["s2", "s1"]}
        sort={{ key: "session_id", direction: "desc" }}
        theme={{} as Theme}
        width={640}
        height={240}
        checks={paint}
        selectedIds={["s1"]}
        folder="library"
        gridSelection={selection}
        onGridSelectionChange={() => {}}
        onSort={(key) => sorts.push(key)}
        onRowCheck={(id) => checks.push(id)}
        onHeaderCheck={() => {
          header += 1;
        }}
        onCommitNote={(edit) => notes.push(edit.after)}
        onOpenMenu={(menu) => menus.push(menu.x)}
      />,
    );
  });
  const point = { preventDefault() {} };
  const box = { x: 0, y: 0, width: 16, height: 16 };
  let drawn = 0;
  editor?.drawCell({ ctx, rect: box, col: 0, row: 0 }, () => {
    drawn += 1;
  });
  editor?.drawCell({ ctx, rect: box, col: 0, row: 1 }, () => {
    drawn += 1;
  });
  editor?.drawCell({ ctx, rect: box, col: 2, row: 0 }, () => {
    drawn += 1;
  });
  editor?.drawHeader({ ctx, columnIndex: 0, rect: box }, () => {
    drawn += 1;
  });
  editor?.drawHeader({ ctx, columnIndex: 1, rect: box }, () => {
    drawn += 1;
  });
  expect(drawn).toBe(3);
  const boolean = editor?.getCellContent([0, 0]) as { data?: boolean };
  const note = editor?.getCellContent([9, 0]) as { displayData?: string };
  const empty = editor?.getCellContent([1, 9]) as { displayData?: string };
  expect(boolean.data).toBe(false);
  expect(note.displayData).toBe("other");
  expect(empty.displayData).toBe("");
  await act(async () => {
    editor?.onCellClicked([0, 0], { ...point, isDoubleClick: false });
    editor?.onCellClicked([1, 0], { ...point, isDoubleClick: true });
    editor?.onCellClicked([1, 0], { ...point, isDoubleClick: true });
    editor?.onCellClicked([9, 0], { ...point, isDoubleClick: false });
    editor?.onCellClicked([1, 9], { ...point, isDoubleClick: false });
    editor?.onHeaderClicked(0);
    editor?.onHeaderClicked(2);
    editor?.onHeaderClicked(99);
    editor?.onCellEdited([9, 0], { kind: "text", data: "next" });
    editor?.onCellEdited([9, 0], { kind: "text", data: "other" });
    editor?.onCellContextMenu([0, 0], { ...point, bounds: { x: 10, y: 20 }, localEventX: 3, localEventY: 4 });
    editor?.onCellContextMenu([0, 0], { ...point, bounds: { x: Number.NaN, y: 0 }, localEventX: 0, localEventY: 0 });
    editor?.onCellContextMenu([0, 9], { ...point, bounds: { x: 1, y: 1 }, localEventX: 0, localEventY: 0 });
    editor?.onGridSelectionChange(selection);
  });
  expect(checks).toContain("s2");
  expect(notes).toEqual(["next"]);
  expect(menus).toEqual([13]);
  expect(sorts).toEqual(["date"]);
  expect(header).toBe(1);
});
