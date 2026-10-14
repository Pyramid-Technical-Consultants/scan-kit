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
      return { title: "Scan Kit", version: "2.0.0-dev" };
    }
    return null;
  }),
}));

vi.mock("@tauri-apps/api/window", () => ({
  availableMonitors: async () => [],
  getCurrentWindow: () => {
    throw new Error("no window");
  },
}));

vi.mock("@tauri-apps/api/dpi", () => ({
  PhysicalSize: class PhysicalSize {
    constructor(public width: number, public height: number) {}
  },
  PhysicalPosition: class PhysicalPosition {
    constructor(public x: number, public y: number) {}
  },
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => null),
}));

vi.mock("@/wasm/scan_kit_plot.js", () => ({
  default: async () => undefined,
  WebPlot: { create: async () => ({}) },
}));

afterEach(() => {
  document.body.replaceChildren();
});

it("mounts the shell into the root element", async () => {
  const host = document.createElement("div");
  host.id = "root";
  document.body.append(host);
  await act(async () => {
    await import("./main");
  });
  expect(host.textContent).toContain("Add a data location to list sessions.");
}, 20000);
