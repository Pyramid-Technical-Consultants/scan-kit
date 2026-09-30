import type { GridSelection } from "@glideapps/glide-data-grid";

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
