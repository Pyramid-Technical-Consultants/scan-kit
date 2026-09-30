import { createRoot } from "react-dom/client";
import { act } from "react";
import { expect, it } from "vitest";

import {
  headerCheck,
  headerWillFill,
  nextSessionSelection,
  checkboxFrame,
  UseCheckLayer,
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

it("ignores checkbox frames that are not on screen", () => {
  const origin = { left: 0, top: 0, right: 400, bottom: 300 };
  expect(checkboxFrame(undefined, origin)).toBeNull();
  expect(checkboxFrame({ x: Number.NaN, y: 0, width: 48, height: 32 }, origin)).toBeNull();
  expect(checkboxFrame({ x: 16, y: 40, width: 48, height: 32 }, origin)).toEqual({
    left: 16,
    top: 40,
    width: 48,
    height: 32,
  });
});

it("renders the shadcn checkbox for the session column", async () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  await act(async () => {
    root.render(
      <UseCheckLayer
        boxes={[
          { key: "header", kind: "header", left: 0, top: 0, width: 48, height: 40 },
          { key: "s1", kind: "row", sessionId: "s1", left: 0, top: 40, width: 48, height: 40 },
          { key: "s2", kind: "row", sessionId: "s2", left: 0, top: 80, width: 48, height: 40 },
        ]}
        order={["s1"]}
        header={{ checked: false, indeterminate: true }}
        onRow={() => {}}
        onHeader={() => {}}
      />,
    );
  });
  const boxes = document.querySelectorAll("[data-slot=checkbox]");
  expect(boxes).toHaveLength(3);
  expect(boxes[0]?.getAttribute("aria-label")).toBe("Select all");
  expect(boxes[1]?.getAttribute("aria-label")).toBe("Select row");
  expect(boxes[1]?.getAttribute("aria-checked")).toBe("true");
  expect(boxes[1]?.getAttribute("title")).toBe("Session color in plots: #4C72B0 (1 of 1)");
  expect((boxes[1] as HTMLElement).style.backgroundColor).toBe("#4C72B0");
  expect(boxes[2]?.getAttribute("title")).toBe(
    "Not used in plots — check to assign a plot color by order",
  );
  expect((boxes[2] as HTMLElement).style.backgroundColor).toBe("");
  await act(async () => {
    root.unmount();
  });
  host.remove();
});
