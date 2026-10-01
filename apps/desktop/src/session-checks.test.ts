import { expect, it } from "vitest";

import {
  headerCheck,
  headerWillFill,
  nextSessionSelection,
  shouldToggleRow,
  toggleListedSessions,
} from "./session-checks";

it("selects at most five sessions from the header", () => {
  const ids = ["a", "b", "c", "d", "e", "f"];
  expect(nextSessionSelection(ids, [], "all", true)).toEqual({
    ids: ["a", "b", "c", "d", "e"],
    capped: true,
  });
  expect(headerWillFill(6, 5)).toBe(false);
  expect(nextSessionSelection(ids, ["a"], "f", true).capped).toBe(false);
  expect(nextSessionSelection(ids, ["a", "b", "c", "d", "e"], "f", true)).toEqual({
    ids: ["a", "b", "c", "d", "e"],
    capped: true,
  });
});

it("clears the header once every visible choice is selected", () => {
  expect(headerCheck(3, 0)).toEqual({ checked: false, indeterminate: false });
  expect(headerCheck(3, 2)).toEqual({ checked: false, indeterminate: true });
  expect(headerCheck(3, 3)).toEqual({ checked: true, indeterminate: false });
  expect(headerWillFill(3, 2)).toBe(true);
  expect(headerWillFill(3, 3)).toBe(false);
});

it("toggles a row on double-click, and only once when the checkbox already did", () => {
  expect(shouldToggleRow(3, true, false)).toBe(true);
  expect(shouldToggleRow(0, true, true)).toBe(false);
  expect(shouldToggleRow(0, false, false)).toBe(true);
  expect(shouldToggleRow(3, false, false)).toBe(false);
});

it("checks listed rows up to the cap and clears them when every one is on", () => {
  expect(toggleListedSessions(["a"], ["c", "b"])).toEqual({ ids: ["a", "c", "b"], capped: false });
  expect(toggleListedSessions(["a", "c"], ["a", "b"])).toEqual({ ids: ["a", "c", "b"], capped: false });
  expect(toggleListedSessions(["a", "b", "c", "d"], ["d", "e", "f"])).toEqual({
    ids: ["a", "b", "c", "d", "e"],
    capped: true,
  });
  expect(toggleListedSessions(["b", "a", "c"], ["a", "c"])).toEqual({ ids: ["b"], capped: false });
  expect(toggleListedSessions([], [])).toEqual({ ids: [], capped: false });
});
