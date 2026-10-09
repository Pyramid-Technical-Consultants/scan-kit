import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";

const windowHooks = vi.hoisted(() => ({
  closeRequest: null as null | ((event: { preventDefault: () => void }) => Promise<void>),
  save: null as null | (() => void),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_data_dirs") {
      return [];
    }
    if (command === "scan_kit_last_main_tab") {
      return null;
    }
    if (command === "scan_kit_window_geometry") {
      return { width: 1200, height: 800, x: 80, y: 80 };
    }
    if (command === "scan_kit_about") {
      return {
        title: "Scan Kit",
        product_line: "Scan Kit",
        description: "Dose review",
        source_lead: "",
        github_label: "Code",
        github_url: "https://example.com",
        maintainer: "",
        website_label: "Web",
        website_url: "https://example.com",
        support_lead: "",
        support_email: "a@example.com",
        copyright: "2026",
        license_lead: "",
        license_label: "MIT",
        license_url: "https://example.com",
        version: "2.0.0-dev",
        commit: "",
      };
    }
    return null;
  }),
}));

vi.mock("@tauri-apps/api/dpi", () => ({
  PhysicalSize: class PhysicalSize {
    constructor(public width: number, public height: number) {}
  },
  PhysicalPosition: class PhysicalPosition {
    constructor(public x: number, public y: number) {}
  },
}));

vi.mock("@tauri-apps/api/window", () => ({
  availableMonitors: async () => [{ position: { x: 0, y: 0 }, size: { width: 1920, height: 1080 } }],
  getCurrentWindow: () => ({
    isMinimized: async () => false,
    isMaximized: async () => false,
    isFullscreen: async () => false,
    outerSize: async () => ({ width: 1200, height: 800 }),
    outerPosition: async () => ({ x: 80, y: 80 }),
    setSize: async () => {},
    setPosition: async () => {},
    close: async () => {},
    destroy: async () => {},
    onResized: async (save: () => void) => {
      windowHooks.save = save;
      return () => {};
    },
    onMoved: async () => () => {},
    onCloseRequested: async (handler: (event: { preventDefault: () => void }) => Promise<void>) => {
      windowHooks.closeRequest = handler;
      return () => {};
    },
  }),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => null),
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
  const press = (key: string, init: KeyboardEventInit = {}) => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init }));
  };
  await act(async () => {
    press("Escape");
    press("F5");
    press("o", { ctrlKey: true });
    press("q", { ctrlKey: true });
    press("1", { ctrlKey: true });
    press("6", { ctrlKey: true });
    press("z", { ctrlKey: true });
    press("y", { ctrlKey: true });
    press("Z", { ctrlKey: true, shiftKey: true });
    press("Escape", { repeat: true });
    const input = document.createElement("input");
    document.body.append(input);
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "o", ctrlKey: true, bubbles: true }));
    input.remove();
  });
  expect(host.textContent).toContain("Console, warnings, and uncaught errors");
  await act(async () => {
    press("1", { ctrlKey: true });
  });
  const exams = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Exams");
  await act(async () => {
    exams?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(host.textContent).toContain("Add a data location to list exams.");
  const locations = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Locations");
  await act(async () => {
    locations?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  windowHooks.save?.();
  await act(async () => {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 450);
    });
    await windowHooks.closeRequest?.({ preventDefault() {} });
  });
  const about = [...document.body.querySelectorAll("button")].find((item) => item.textContent?.includes("About"));
  await act(async () => {
    about?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
});
