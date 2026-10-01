import type { GridSelection } from "@glideapps/glide-data-grid";

/** Rows the context menu should toggle. A click inside the highlight uses that span; otherwise just the clicked row. */
export function selectedSessionIds(
  displayIds: readonly string[],
  selection: GridSelection,
  clickedRow: number,
): string[] {
  const indexes = new Set<number>();
  const add = (range: { y: number; height: number }) => {
    for (let offset = 0; offset < range.height; offset += 1) {
      indexes.add(range.y + offset);
    }
  };
  const current = selection.current;
  if (current != null) {
    add(current.range);
    for (const range of current.rangeStack) {
      add(range);
    }
  }
  const rows = indexes.has(clickedRow) ? [...indexes] : [clickedRow];
  rows.sort((left, right) => left - right);
  return rows.flatMap((index) => {
    const id = displayIds[index];
    return id == null ? [] : [id];
  });
}

type Range = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

function widen(range: Range, columnCount: number): Range {
  return { x: 0, y: range.y, width: columnCount, height: range.height };
}

function isFullWidth(range: Range, columnCount: number): boolean {
  return range.x === 0 && range.width === columnCount;
}

/** Glide `rowSelect` only affects the row-marker gutter. Widen a cell range so the row is the selection. */
export function selectWholeRows(selection: GridSelection, columnCount: number): GridSelection {
  const current = selection.current;
  if (current == null || columnCount <= 0) {
    return selection;
  }
  if (
    isFullWidth(current.range, columnCount) &&
    current.rangeStack.every((range) => isFullWidth(range, columnCount))
  ) {
    return selection;
  }
  return {
    ...selection,
    current: {
      cell: current.cell,
      range: widen(current.range, columnCount),
      rangeStack: current.rangeStack.map((range) => widen(range, columnCount)),
    },
  };
}
