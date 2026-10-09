import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { DoseBoard } from "./DoseBoard";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

function board() {
  const plots = { current: [null, null, null, null, null, null] };
  return (
    <DoseBoard
      shown
      controls={[]}
      resolved={{}}
      plots={plots}
      acquire={async () => {
        throw new Error("the canvas has no size");
      }}
      onChange={() => undefined}
      onAction={() => undefined}
      onReady={() => undefined}
      onInput={() => undefined}
      onError={() => undefined}
      renderChoice={() => null}
    />
  );
}

it("keeps one column fraction across every row", () => {
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(board());
  });
  const handles = [...host.querySelectorAll("[aria-label='Resize column']")];
  expect(handles).toHaveLength(3);
  expect(handles.map((handle) => handle.getAttribute("aria-valuenow"))).toEqual(["50", "50", "50"]);
  act(() => {
    handles[0]?.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, key: "ArrowRight" }));
  });
  const next = handles.map((handle) => handle.getAttribute("aria-valuenow"));
  expect(new Set(next).size).toBe(1);
  expect(Number(next[0])).toBeGreaterThan(50);
});
