import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it } from "vitest";

import { AnalysisMenu, PRIMARY_ANALYSES } from "./analysis-menu";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
});

it("lists quick-open buttons in the core analysis order", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(<AnalysisMenu onOpen={() => undefined} />);
  });
  const trigger = host.querySelector("[aria-label='More analyses']");
  if (!(trigger instanceof HTMLElement)) {
    throw new Error("missing analyses menu");
  }
  await act(async () => {
    trigger.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const labels = [...document.querySelectorAll("[data-slot='dropdown-menu-label']")].map((node) =>
    node.textContent?.trim(),
  );
  const names = [...document.querySelectorAll("[data-slot='dropdown-menu-item']")].map((node) =>
    node.textContent?.trim(),
  );
  expect(labels).toEqual(["Core Analysis", "Specialized Analysis"]);
  expect(names).toEqual([
    "Timeline",
    "Bins",
    "Distribution",
    "Volumetric",
    "Session Log Compare",
    "IC HV Transient Test",
  ]);
  expect([...PRIMARY_ANALYSES]).toEqual(names.slice(0, PRIMARY_ANALYSES.length));
});
