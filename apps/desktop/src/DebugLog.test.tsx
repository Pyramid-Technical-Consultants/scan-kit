import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { appendLog, clearLog } from "./debug-log";
import { DebugLog } from "./DebugLog";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  clearLog();
  document.body.replaceChildren();
});

it("copies the log and clears it", async () => {
  const written: string[] = [];
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: async (text: string) => { written.push(text); } },
  });
  appendLog("info", "app", "beam on");
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<DebugLog />);
  });
  expect(document.body.textContent).toContain("beam on");
  appendLog("warn", "app", "drift");
  await act(async () => {
    await Promise.resolve();
  });
  expect(document.body.textContent).toContain("drift");
  const copy = [...document.body.querySelectorAll("button")].find((item) => item.textContent?.includes("Copy"));
  const clear = [...document.body.querySelectorAll("button")].find((item) => item.textContent?.includes("Clear"));
  await act(async () => {
    copy?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    clear?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(written.some((text) => text.includes("beam on"))).toBe(true);
  expect(document.body.textContent).not.toContain("beam on");
});
