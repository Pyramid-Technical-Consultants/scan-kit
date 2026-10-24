import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { DevicePicker } from "./device-picker";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async () => ({ devices: [] })),
}));

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
  vi.clearAllMocks();
});

function render(onChange = vi.fn()) {
  const host = document.createElement("div");
  document.body.append(host);
  act(() => {
    root = createRoot(host);
    root.render(
      <DevicePicker
        value="192.168.100.184"
        onChange={onChange}
        onConnect={() => undefined}
      />,
    );
  });
  return { host, onChange };
}

function button(host: HTMLElement, label: string | RegExp) {
  return [...host.querySelectorAll("button")].find((el) =>
    typeof label === "string" ? el.textContent?.includes(label) : label.test(el.textContent ?? ""),
  );
}

it("types a host and connects", () => {
  const { host } = render();
  const input = host.querySelector("input");
  expect(input?.placeholder).toBe("192.168.100.184");
  expect(button(host, "Connect")).toBeTruthy();
});

it("opens the scan panel with a subnet derived from the host", async () => {
  const { host } = render();
  const scanToggle = host.querySelector("button[aria-expanded]");
  await act(async () => {
    scanToggle?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const subnet = [...host.querySelectorAll("input")].find(
    (el) => el.value === "192.168.100.0/24",
  );
  expect(subnet).toBeTruthy();
});

it("scans and picks a discovered device", async () => {
  vi.mocked(invoke).mockResolvedValueOnce({
    devices: [
      { host: "192.168.100.184", version: "3.2.1", device_type: "ix256" },
      { host: "192.168.100.10", version: "3.2.1" },
    ],
  });
  const { host, onChange } = render();
  await act(async () => {
    host
      .querySelector("button[aria-expanded]")
      ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await act(async () => {
    button(host, "Scan")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();
  });
  expect(invoke).toHaveBeenCalledWith("scan_kit_discover_subnet", {
    subnet: "192.168.100.0/24",
  });
  const pick = button(host, /192\.168\.100\.184/);
  expect(pick?.textContent).toContain("ix256");
  await act(async () => {
    pick?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(onChange).toHaveBeenCalledWith("192.168.100.184");
});
