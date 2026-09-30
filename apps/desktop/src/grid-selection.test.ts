import { CompactSelection, type GridSelection } from "@glideapps/glide-data-grid";
import { expect, it } from "vitest";

import { selectWholeRows } from "./grid-selection";

function selection(
  cell: readonly [number, number],
  range: { x: number; y: number; width: number; height: number },
  rangeStack: { x: number; y: number; width: number; height: number }[] = [],
): GridSelection {
  return {
    columns: CompactSelection.empty(),
    rows: CompactSelection.empty(),
    current: { cell, range, rangeStack },
  };
}

it("widens a cell click to the whole row", () => {
  const next = selectWholeRows(selection([3, 2], { x: 3, y: 2, width: 1, height: 1 }), 10);
  expect(next.current?.cell).toEqual([3, 2]);
  expect(next.current?.range).toEqual({ x: 0, y: 2, width: 10, height: 1 });
});

it("keeps a dragged row span and widens each extra range", () => {
  const next = selectWholeRows(
    selection([1, 4], { x: 1, y: 2, width: 2, height: 3 }, [{ x: 4, y: 0, width: 1, height: 1 }]),
    10,
  );
  expect(next.current?.range).toEqual({ x: 0, y: 2, width: 10, height: 3 });
  expect(next.current?.rangeStack).toEqual([{ x: 0, y: 0, width: 10, height: 1 }]);
});

it("leaves a selection that is already full width alone", () => {
  const current = selection([3, 2], { x: 0, y: 2, width: 10, height: 1 });
  expect(selectWholeRows(current, 10)).toBe(current);
  expect(selectWholeRows({ columns: CompactSelection.empty(), rows: CompactSelection.empty() }, 10)).toEqual({
    columns: CompactSelection.empty(),
    rows: CompactSelection.empty(),
  });
});
