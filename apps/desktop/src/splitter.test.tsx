import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { Splitter } from "./splitter";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("reports pointer travel from the drag start", () => {
  const moves: number[] = [];
  let starts = 0;
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(
      <Splitter
        orientation="vertical"
        label="Resize column"
        now={50}
        min={18}
        onStart={() => {
          starts += 1;
        }}
        onMove={(delta) => {
          moves.push(delta);
        }}
      />,
    );
  });
  const handle = host.querySelector("[role='separator']");
  if (handle == null) {
    throw new Error("missing splitter");
  }
  handle.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, clientX: 10, pointerId: 1 }));
  handle.dispatchEvent(new PointerEvent("pointermove", { bubbles: true, clientX: 26, pointerId: 1 }));
  handle.dispatchEvent(new PointerEvent("pointermove", { bubbles: true, clientX: 18, pointerId: 1 }));
  expect(starts).toBe(1);
  expect(moves).toEqual([16, 8]);
  expect(handle.getAttribute("aria-orientation")).toBe("vertical");
  expect(handle.getAttribute("aria-valuemin")).toBe("18");
  expect(handle.getAttribute("aria-valuenow")).toBe("50");
});

it("steps from the keyboard", () => {
  const moves: number[] = [];
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(
      <Splitter
        orientation="horizontal"
        label="Resize row"
        now={40}
        min={15}
        onStart={() => undefined}
        onMove={(delta) => {
          moves.push(delta);
        }}
      />,
    );
  });
  const handle = host.querySelector("[role='separator']");
  handle?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "ArrowDown" }));
  handle?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "ArrowUp" }));
  handle?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "Home" }));
  expect(moves).toEqual([16, -16]);
  expect(handle?.getAttribute("aria-orientation")).toBe("horizontal");
});
