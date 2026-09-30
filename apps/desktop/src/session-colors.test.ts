import { expect, it } from "vitest";

import { selectionFromLibrary, sessionColor, sessionSwatch, UNCHECKED_SWATCH } from "./session-colors";

it("assigns the seaborn palette by check order and wraps", () => {
  expect(sessionColor(0)).toBe("#4C72B0");
  expect(sessionColor(1)).toBe("#DD8452");
  expect(sessionColor(8)).toBe("#4C72B0");
  expect(sessionSwatch(["b", "a"], "a")).toEqual({
    color: "#DD8452",
    label: "Session color in plots: #DD8452 (2 of 2)",
  });
  expect(sessionSwatch(["b"], "missing").color).toBe(UNCHECKED_SWATCH);
  expect(
    selectionFromLibrary(
      [
        { session_id: "a", selected: true },
        { session_id: "b", selected: true },
      ],
      ["b", "a"],
    ),
  ).toEqual(["b", "a"]);
});
