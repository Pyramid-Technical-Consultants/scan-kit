import { expect, it } from "vitest";

import { pageLoadFraction } from "./page-load";

it("shows nothing until a load starts, then sweeps or fills", () => {
  expect(pageLoadFraction([])).toBeNull();
  expect(pageLoadFraction([{ done: 0, total: 4 }])).toEqual({ done: 0, total: 0 });
  expect(pageLoadFraction([{ done: 1, total: 4 }])).toEqual({ done: 1, total: 4 });
  expect(
    pageLoadFraction([
      { done: 1, total: 4 },
      { done: 0, total: 0 },
    ]),
  ).toEqual({ done: 0, total: 0 });
});
