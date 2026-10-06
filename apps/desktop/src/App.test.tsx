import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_data_dirs") {
      return [];
    }
    if (command === "scan_kit_last_main_tab") {
      return null;
    }
    if (command === "scan_kit_about") {
      return {
        title: "Scan Kit",
        product_line: "Scan Kit",
        description: "",
        version: "2.0.0-dev",
        commit: "",
      };
    }
    return null;
  }),
}));

vi.mock("@/wasm/scan_kit_plot.js", () => ({
  default: async () => undefined,
  WebPlot: { create: async () => ({}) },
}));

import App from "./App";

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
});

it("mounts the data view that opens every analysis plot", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(<App />);
  });
  expect(host.textContent).toContain("Add a data location to list sessions.");
});
