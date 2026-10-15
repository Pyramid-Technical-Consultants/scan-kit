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

it("tries a canvas again after its plot failed to open", async () => {
  const resized: ResizeObserverCallback[] = [];
  const realObserver = globalThis.ResizeObserver;
  const realRect = HTMLElement.prototype.getBoundingClientRect;
  globalThis.ResizeObserver = class {
    constructor(callback: ResizeObserverCallback) {
      resized.push(callback);
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
  HTMLElement.prototype.getBoundingClientRect = () => new DOMRect(0, 0, 200, 100);
  let attempts = 0;
  const ready: number[] = [];
  const errors: unknown[] = [];
  const plot = { solo: () => undefined, resize: () => undefined };
  const host = document.createElement("div");
  document.body.append(host);
  try {
    await act(async () => {
      root = createRoot(host);
      root.render(
        <DoseBoard
          shown
          controls={[]}
          resolved={{}}
          plots={{ current: [null, null, null, null, null, null] }}
          acquire={async () => {
            attempts += 1;
            if (attempts <= 6) {
              throw new Error("no context yet");
            }
            return plot as never;
          }}
          onChange={() => undefined}
          onAction={() => undefined}
          onReady={(panel) => ready.push(panel)}
          onInput={() => undefined}
          onError={(reason) => errors.push(reason)}
          renderChoice={() => null}
        />,
      );
    });
    expect(errors).toHaveLength(6);
    expect(ready).toEqual([]);
    await act(async () => {
      for (const callback of resized) {
        callback([], {} as ResizeObserver);
      }
    });
    expect(ready.sort()).toEqual([0, 1, 2, 3, 4, 5]);
  } finally {
    globalThis.ResizeObserver = realObserver;
    HTMLElement.prototype.getBoundingClientRect = realRect;
  }
});

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
