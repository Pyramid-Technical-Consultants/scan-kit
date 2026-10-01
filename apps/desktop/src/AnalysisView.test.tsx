import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";

import { AnalysisView } from "./AnalysisView";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command !== "scan_kit_open_plot") {
      throw new Error(command);
    }
    const header = {
      title: "Dose Ratios vs Energy",
      controls: [
        { id: "metric", label: "Y", options: ["Dose Ratios"], value: "Dose Ratios" },
        { id: "source", label: "Source", options: ["Spot — Isocenter"], value: "Spot — Isocenter" },
        { id: "x", label: "X", options: ["Energy", "Target MU", "Spot time", "Radius"], value: "Energy" },
        { id: "domain", label: "Domain", options: ["All", "Lower 95%", "Upper 95%", "MAD outliers"], value: "All" },
        { id: "beam", label: "Beam", options: ["Beam on", "Beam off", "Both"], value: "Beam on" },
        { id: "trend", label: "Trend", options: ["On", "Off"], value: "On" },
      ],
      table: null,
      samples: [],
      panels: [{}],
    };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const bytes = new Uint8Array(4 + json.length);
    new DataView(bytes.buffer).setUint32(0, json.length, true);
    bytes.set(json, 4);
    return bytes;
  }),
}));

vi.mock("@/wasm/scan_kit_plot.js", () => ({
  default: async () => undefined,
  WebPlot: {
    create: async () => ({
      backend: () => "test",
      load: () => undefined,
      render: () => undefined,
      resize: () => undefined,
      hover: () => null,
      zoom: () => undefined,
      pan: () => undefined,
      reset: () => undefined,
    }),
  },
}));

let root: Root | null = null;

afterEach(() => {
  act(() => {
    root?.unmount();
  });
  root = null;
  document.body.replaceChildren();
});

it("puts grouped controls on the right and returns to sessions", async () => {
  const onBack = vi.fn();
  const host = document.createElement("div");
  document.body.append(host);
  await act(() => {
    root = createRoot(host);
    root.render(
      <AnalysisView viewId="binned_summary" folder="C:/data" sessionIds={["1"]} onBack={onBack} />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });

  const legends = [...host.querySelectorAll("legend")].map((node) => node.textContent);
  expect(legends).toEqual(["Data Source", "Plot Style", "Filter Data"]);
  expect(host.querySelector("h2")).toBeNull();
  expect(host.textContent).not.toContain("Presets");
  expect(host.textContent).not.toContain("Dose Ratios vs Energy");

  const shell = host.firstElementChild;
  const plot = shell?.children[0];
  const handle = shell?.children[1];
  const aside = shell?.children[2];
  expect(plot?.querySelector("canvas")).not.toBeNull();
  expect(handle?.getAttribute("aria-label")).toBe("Resize configuration");
  expect(aside?.textContent).toContain("Sessions");
  expect(aside?.textContent).toContain("Domain");
  expect((aside as HTMLElement | undefined)?.style.width).toBe("350px");
  const segments = [...host.querySelectorAll("[data-slot='toggle-group-item']")].map((node) =>
    node.textContent?.trim(),
  );
  expect(segments).toEqual(["Beam on", "Beam off", "Both"]);
  expect(host.querySelector("[data-slot='radio-group']")).toBeNull();

  const back = [...host.querySelectorAll("button")].find((button) => button.textContent?.includes("Sessions"));
  await act(async () => {
    back?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(onBack).toHaveBeenCalledOnce();
});
