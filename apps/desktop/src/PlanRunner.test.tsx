import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { PlanRunner } from "./PlanRunner";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => "C:/plans/input_map.csv"),
  save: vi.fn(async () => null),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_runner_catalog") {
      return {
        host: "192.168.100.184",
        file_dir: null,
        connected: false,
        view: {
          connected: false,
          state: "—",
          subtitle: "Connect, upload a plan, then press Start",
          progress: null,
          point_progress: null,
          point: "—",
          energy: "—",
          layer: "—",
          elapsed: "—",
          permit: "—",
          points: "—",
          enables: { start: false, pause: false, stop: false, reset: false },
          coach: "Enter the RCI IP and click Connect.",
        },
      };
    }
    if (command === "scan_kit_runner_connect") {
      return {
        connected: true,
        state: "LOCKED",
        subtitle: "Waiting for live status",
        progress: 0,
        point_progress: 0,
        point: "—",
        energy: "—",
        layer: "—",
        elapsed: "—",
        permit: "Held",
        points: "Invalid",
        enables: { start: false, pause: false, stop: false, reset: true },
        coach: "Connected. Browse to an input_map.csv, then upload it.",
        host: "192.168.100.184",
        version: "1.2.3",
        device_type: "RCI",
      };
    }
    if (command === "scan_kit_runner_upload") {
      return {
        connected: true,
        state: "LOCKED",
        subtitle: "Waiting for live status",
        progress: 0,
        point_progress: 0,
        point: "—",
        energy: "—",
        layer: "—",
        elapsed: "—",
        permit: "Held",
        points: "Invalid",
        enables: { start: false, pause: false, stop: false, reset: true },
        coach: "Connected. Browse to an input_map.csv, then upload it.",
        host: "192.168.100.184",
        version: "1.2.3",
        device_type: "RCI",
        message: "Uploaded the plan.",
      };
    }
    if (command === "scan_kit_runner_control") {
      return {
        connected: true,
        state: "IDLE",
        subtitle: "Reset",
        progress: 0,
        point_progress: 0,
        point: "—",
        energy: "—",
        layer: "—",
        elapsed: "—",
        permit: "Held",
        points: "Invalid",
        enables: { start: false, pause: false, stop: false, reset: true },
        coach: "Reset.",
        host: "192.168.100.184",
        version: "1.2.3",
        device_type: "",
      };
    }
    if (command === "scan_kit_runner_disconnect") {
      return {
        host: "192.168.100.184",
        file_dir: "C:/plans",
        connected: false,
        view: {
          connected: false,
          state: "—",
          subtitle: "Connect, upload a plan, then press Start",
          progress: null,
          point_progress: null,
          point: "—",
          energy: "—",
          layer: "—",
          elapsed: "—",
          permit: "—",
          points: "—",
          enables: { start: false, pause: false, stop: false, reset: false },
          coach: "Disconnected.",
        },
      };
    }
    if (command === "scan_kit_runner_remember") {
      return null;
    }
    throw new Error(command);
  }),
}));

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("connects with the remembered host and shows the controller state", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<PlanRunner />);
  });
  expect(document.body.textContent).toContain("Enter the RCI IP and click Connect.");
  const start = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Start");
  expect(start).toBeTruthy();
  expect((start as HTMLButtonElement).disabled).toBe(true);
  const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Connect");
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(invoke).toHaveBeenCalledWith("scan_kit_runner_connect", { host: "192.168.100.184" });
  expect(document.body.textContent).toContain("LOCKED");
  expect(document.body.textContent).toContain("Connected. Browse to an input_map.csv");
  const click = async (label: string) => {
    const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === label);
    expect(button).toBeTruthy();
    await act(async () => {
      button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  };
  const hostInput = document.querySelector("input");
  await act(async () => {
    hostInput?.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  });
  await click("Browse");
  await click("Upload to RCI");
  await click("Reset");
  const zipBrowse = [...document.body.querySelectorAll("button")].filter((item) => item.textContent === "Browse");
  await act(async () => {
    zipBrowse[1]?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const download = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Download");
  await act(async () => {
    download?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await click("Disconnect");
  expect(document.body.textContent).toContain("Disconnected.");
  expect(invoke).toHaveBeenCalledWith("scan_kit_runner_upload", { path: "C:/plans/input_map.csv" });
});
