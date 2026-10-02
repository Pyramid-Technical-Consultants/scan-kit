import { expect, it } from "vitest";

import { checkboxRect } from "./session-checkbox";

it("centers a 16px checkbox in the cell", () => {
  expect(checkboxRect({ x: 10, y: 20, width: 48, height: 32 })).toEqual({
    x: 26,
    y: 28,
    size: 16,
  });
});
