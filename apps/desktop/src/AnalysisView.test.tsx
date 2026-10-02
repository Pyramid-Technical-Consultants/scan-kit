import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { AnalysisView } from "./AnalysisView";
import { sessionColor } from "./session-colors";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string, args?: { view?: string }) => {
    if (command !== "scan_kit_open_plot") {
      throw new Error(command);
    }
    const distribution = args?.view === "distribution";
    const header = distribution
      ? {
          title: "Distribution",
          controls: [
            { id: "source", label: "Source", group: "Data Source", options: ["Spot", "Timeslice"], value: "Spot" },
            {
              id: "xy",
              label: "XY",
              group: "Data Source",
              options: [{ id: "position", label: "Position (mm)", detail: "Chamber or plan", icon: "position" }],
              value: "Position (mm)",
            },
            { id: "ic1", label: "IC1", group: "Data Source", kind: "check", options: ["Off", "On"], value: "On" },
            { id: "ic2", label: "IC2", group: "Data Source", kind: "check", options: ["Off", "On"], value: "Off" },
            { id: "plan", label: "Plan", group: "Data Source", kind: "check", options: ["Off", "On"], value: "On" },
          ],
          table: null,
          samples: [],
          panels: [{}],
        }
      : {
          title: "Dose Ratios vs Energy",
          controls: [
            { id: "source", label: "Source", group: "Data Source", options: ["Spot", "Timeslice"], value: "Spot" },
            {
              id: "y",
              label: "Y",
              group: "Data Source",
              options: [{ id: "dose_ratio", label: "Dose Ratios", detail: "Chambers over each other", icon: "dose_ratio" }],
              value: "Dose Ratios",
            },
            { id: "x", label: "X", group: "Data Source", options: ["Energy", "Target MU", "Spot time", "Radius"], value: "Energy" },
            { id: "bins", label: "Bins", group: "Data Source", options: ["Automatic", "8", "16", "32", "64"], value: "Automatic" },
            { id: "trend", label: "Trend", group: "Plot Style", options: ["Off", "Linear", "Polynomial"], value: "Off" },
            { id: "domain", label: "Domain", group: "Filter Data", options: ["All", "Lower 95%", "Upper 5%", "MAD Outliers"], value: "All" },
            { id: "beam", label: "Beam", group: "Filter Data", options: ["Beam On", "Beam Off", "Both"], value: "Beam On" },
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
      <AnalysisView
        viewId="binned_summary"
        folder="C:/data"
        sessions={[
          { id: "1093436476", note: "Morning" },
          { id: "1093436477", note: "" },
        ]}
        onBack={onBack}
        onOpenView={() => undefined}
      />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });

  const legends = [...host.querySelectorAll("legend")].map((node) => node.textContent);
  expect(legends).toEqual(["Sessions", "Data Source", "Plot Style", "Filter Data"]);
  expect(host.textContent).toContain("1093436476");
  expect(host.textContent).toContain("Morning");
  expect(host.textContent).toContain("1093436477");
  const noted = host.querySelector("label[for='session-list-1093436476']");
  const note = noted?.querySelector("span.truncate");
  expect(note?.textContent).toBe("Morning");
  expect(note?.getAttribute("title")).toBe("Morning");
  expect(noted?.contains(note ?? null)).toBe(true);
  expect(host.querySelector("label[for='session-list-1093436477']")?.querySelector(".truncate")).toBeNull();
  const sessionBoxes = [...host.querySelectorAll("fieldset")][0]?.querySelectorAll("[data-slot='checkbox']");
  expect(sessionBoxes?.length).toBe(2);
  expect(sessionBoxes?.[0]?.getAttribute("aria-checked")).toBe("true");
  expect(sessionBoxes?.[0]?.getAttribute("style")).toContain(sessionColor(0));
  expect(sessionBoxes?.[1]?.getAttribute("style")).toContain(sessionColor(1));
  expect(host.querySelector("h2")).toBeNull();
  expect(host.textContent).not.toContain("Presets");
  expect(host.textContent).not.toContain("Dose Ratios vs Energy");
  expect(host.textContent).toContain("Chambers over each other");

  const shell = host.firstElementChild;
  const plot = shell?.children[0];
  const handle = shell?.children[1];
  const aside = shell?.children[2];
  expect(plot?.querySelector("canvas")).not.toBeNull();
  expect(handle?.getAttribute("aria-label")).toBe("Resize configuration");
  expect(aside?.textContent).toContain("Sessions");
  expect(aside?.textContent).toContain("Domain");
  const fields = [...host.querySelectorAll("[data-slot='field']")];
  expect(fields.length).toBeGreaterThan(0);
  expect(fields.every((field) => field.getAttribute("data-orientation") === "horizontal")).toBe(true);
  expect((aside as HTMLElement | undefined)?.style.width).toBe("420px");
  const segments = [...host.querySelectorAll("[data-slot='toggle-group-item']")].map((node) =>
    node.textContent?.trim(),
  );
  expect(segments).toEqual([
    "Spot",
    "Timeslice",
    "Off",
    "Linear",
    "Polynomial",
    "Beam On",
    "Beam Off",
    "Both",
  ]);
  expect(host.textContent).not.toContain("Show Box Outliers");
  expect(host.textContent).not.toContain("Trend Line");
  const bins = [...host.querySelectorAll("[data-slot='select-trigger']")].find((node) =>
    node.textContent?.includes("Automatic"),
  );
  expect(bins?.hasAttribute("disabled") || bins?.getAttribute("aria-disabled") === "true").toBe(false);
  expect(bins?.getAttribute("data-size")).toBe("sm");
  expect(host.querySelector("[data-slot='toggle-group']")?.getAttribute("data-size")).toBe("sm");
  expect(host.querySelector("[data-slot='radio-group']")).toBeNull();

  const back = [...host.querySelectorAll("button")].find((button) => button.textContent?.includes("Sessions"));
  await act(async () => {
    back?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(onBack).toHaveBeenCalledOnce();
});

it("opens another analysis from the menu beside Sessions", async () => {
  const onOpenView = vi.fn();
  const host = document.createElement("div");
  document.body.append(host);
  await act(() => {
    root = createRoot(host);
    root.render(
      <AnalysisView
        viewId="binned_summary"
        folder="C:/data"
        sessions={[{ id: "a", note: "" }]}
        onBack={() => undefined}
        onOpenView={onOpenView}
      />,
    );
  });
  const back = [...host.querySelectorAll("button")].find((button) => button.textContent?.includes("Sessions"));
  const more = host.querySelector("[aria-label='More analyses']");
  if (!(back instanceof HTMLElement) || !(more instanceof HTMLElement)) {
    throw new Error("missing sessions controls");
  }
  expect(back.compareDocumentPosition(more) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  await act(async () => {
    more?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  const item = [...document.querySelectorAll("[data-slot='dropdown-menu-item']")].find((node) =>
    node.textContent?.includes("FFT Explorer"),
  );
  expect(item).toBeInstanceOf(HTMLElement);
  expect(
    [...document.querySelectorAll("[data-slot='dropdown-menu-item']")].some((node) =>
      node.textContent?.includes("Binned Summary"),
    ),
  ).toBe(false);
  await act(async () => {
    item?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(onOpenView).toHaveBeenCalledWith("ic_fft_analysis");
});

it("hides an unchecked session and keeps the other session's color", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(() => {
    root = createRoot(host);
    root.render(
      <AnalysisView
        viewId="binned_summary"
        folder="C:/data"
        sessions={[
          { id: "a", note: "first" },
          { id: "b", note: "second" },
        ]}
        onBack={() => undefined}
        onOpenView={() => undefined}
      />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const first = host.querySelector("[aria-label='Include a']");
  await act(async () => {
    first?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const calls = vi
    .mocked(invoke)
    .mock.calls.filter(([command]) => command === "scan_kit_open_plot")
    .map(([, args]) => args as { sessionIds: string[]; palette: number[][] });
  const last = calls[calls.length - 1];
  expect(last?.sessionIds).toEqual(["b"]);
  const kept = last?.palette[0];
  const original = calls[0]?.palette[1];
  expect(kept).toEqual(original);
});

it("lays distribution columns in plot order on one row with icons", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(() => {
    root = createRoot(host);
    root.render(
      <AnalysisView
        viewId="distribution"
        folder="C:/data"
        sessions={[{ id: "a", note: "" }]}
        onBack={() => undefined}
        onOpenView={() => undefined}
      />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const source = [...host.querySelectorAll("fieldset")].find(
    (node) => node.querySelector("legend")?.textContent === "Data Source",
  );
  expect(source?.textContent).toContain("Chamber or plan");
  const row = [...(source?.querySelectorAll(":scope > div") ?? [])].find(
    (node) => node.querySelectorAll("[data-slot='checkbox']").length === 3,
  );
  expect(row?.className).toContain("flex-row");
  const labels = [...(row?.querySelectorAll("label") ?? [])].map((node) => node.textContent?.trim());
  expect(labels).toEqual(["IC1", "IC2", "Plan"]);
  expect(row?.querySelector(".lucide-target")).not.toBeNull();
  expect(row?.querySelectorAll(".lucide-zap").length).toBe(2);
  const checks = [...(row?.querySelectorAll("[data-slot='checkbox']") ?? [])];
  expect(checks.map((node) => node.getAttribute("aria-checked"))).toEqual(["true", "false", "true"]);
});
