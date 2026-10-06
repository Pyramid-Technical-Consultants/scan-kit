import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { AnalysisView } from "./AnalysisView";
import { sessionColor } from "./session-colors";

const { followPlot } = vi.hoisted(() => ({ followPlot: vi.fn() }));

function pollFrame(payload: Uint8Array): Uint8Array {
  const report = {
    task: 1,
    generation: 1,
    phase: "done",
    done: 1,
    total: 1,
    note: "",
    finished: true,
  };
  const json = new TextEncoder().encode(JSON.stringify(report));
  const out = new Uint8Array(4 + json.length + payload.byteLength);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4);
  out.set(payload, 4 + json.length);
  return out;
}

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string, args?: { view?: string }) => {
    if (command === "scan_kit_cancel") {
      return null;
    }
    if (command === "scan_kit_start") {
      return { task: 1, generation: 1 };
    }
    if (command !== "scan_kit_poll") {
      throw new Error(command);
    }
    const starts = vi.mocked(invoke).mock.calls.filter(([name]) => name === "scan_kit_start");
    const started = starts[starts.length - 1]?.[1] as { view?: string } | undefined;
    const distribution = (started?.view ?? args?.view) === "distribution";
    const header = distribution
      ? {
          title: "Distribution",
          controls: [
            {
              id: "source",
              label: "Source",
              group: "Data Source",
              options: [
                { id: "spot", label: "Spot", detail: "One row per spot", icon: "spot" },
                { id: "timeslice", label: "Timeslice", detail: "One row per millisecond", icon: "timeslice" },
              ],
              value: "Spot",
            },
            {
              id: "xy",
              label: "XY",
              group: "Data Source",
              options: [{ id: "position", label: "Position (mm)", detail: "Chamber or plan", icon: "position" }],
              value: "Position (mm)",
            },
            { id: "plan", label: "Plan", group: "Data Source", kind: "check", options: ["Off", "On"], value: "On" },
            { id: "ic1", label: "IC1", group: "Data Source", kind: "check", options: ["Off", "On"], value: "On" },
            { id: "ic2", label: "IC2", group: "Data Source", kind: "check", options: ["Off", "On"], value: "Off" },
          ],
          table: null,
          samples: [],
          panels: [{}],
        }
      : {
          title: "Dose Ratios vs Energy",
          controls: [
            {
              id: "source",
              label: "Source",
              group: "Data Source",
              options: [
                { id: "spot", label: "Spot", detail: "One row per spot", icon: "spot" },
                { id: "timeslice", label: "Timeslice", detail: "One row per millisecond", icon: "timeslice" },
              ],
              value: "Spot",
            },
            {
              id: "y",
              label: "Y",
              group: "Data Source",
              options: [{ id: "dose_ratio", label: "Dose Ratios", detail: "Chambers over each other", icon: "dose_ratio" }],
              value: "Dose Ratios",
            },
            { id: "x", label: "X", group: "Data Source", options: ["Energy", "Target MU", "Spot time", "Radius"], value: "Energy" },
            { id: "bins", label: "Bins", group: "Data Source", options: ["Auto", "8", "16", "32", "64"], value: "Auto" },
            { id: "trend", label: "Trend", group: "Plot Style", options: ["Off", "Linear", "Polynomial"], value: "Off" },
            {
              id: "segments",
              label: "Segments",
              group: "Filter Data",
              kind: "segments",
              options: [
                { id: "beam", label: "Beam", detail: "", icon: "" },
                { id: "rank", label: "Rank", detail: "", icon: "" },
              ],
              value: JSON.stringify([
                { kind: "beam", state: "on" },
                { kind: "rank", which: "all" },
              ]),
            },
            {
              id: "scrub",
              label: "Timeline",
              kind: "scrub",
              options: [],
              value: JSON.stringify({
                on: false,
                at: 0,
                end: 12.5,
                speed: 1,
                window: "second",
                layers: [1, 4],
              }),
            },
          ],
          table: null,
          samples: [],
          panels: [{}],
        };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const bytes = new Uint8Array(4 + json.length);
    new DataView(bytes.buffer).setUint32(0, json.length, true);
    bytes.set(json, 4);
    return pollFrame(bytes);
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
      follow: (...args: unknown[]) => followPlot(...args),
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
  followPlot.mockClear();
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
  expect(host.textContent).not.toContain("Chambers over each other");
  expect(host.textContent).not.toContain("One row per spot");
  const dose = [...host.querySelectorAll("[data-slot='select-trigger']")].find((node) =>
    node.textContent?.includes("Dose Ratios"),
  );
  expect(dose?.querySelector("[title]")?.getAttribute("title")).toBe("Chambers over each other");
  const spot = [...host.querySelectorAll("[data-slot='toggle-group-item']")].find((node) =>
    node.textContent?.trim() === "Spot",
  );
  expect(spot?.getAttribute("title")).toBe("One row per spot");

  const shell = host.firstElementChild;
  const plot = shell?.children[0];
  const handle = shell?.children[1];
  const aside = shell?.children[2];
  expect(plot?.querySelector("canvas")).not.toBeNull();
  expect(handle?.getAttribute("aria-label")).toBe("Resize configuration");
  expect(aside?.textContent).toContain("Sessions");
  expect(aside?.textContent).toContain("Rank");
  const fields = [...host.querySelectorAll("[data-slot='field']")];
  expect(fields.length).toBeGreaterThan(0);
  expect(fields.every((field) => field.getAttribute("data-orientation") === "horizontal")).toBe(true);
  expect((aside as HTMLElement | undefined)?.style.width).toBe("420px");
  const segments = [...(aside?.querySelectorAll("[data-slot='toggle-group-item']") ?? [])].map(
    (node) => node.textContent?.trim(),
  );
  expect(segments).toEqual([
    "Spot",
    "Timeslice",
    "Auto",
    "8",
    "16",
    "32",
    "64",
    "Off",
    "Linear",
    "Polynomial",
    "Beam On",
    "Beam Off",
    "Both",
  ]);
  expect(host.textContent).not.toContain("Show Box Outliers");
  expect(host.textContent).not.toContain("Trend Line");
  const energy = [...host.querySelectorAll("[data-slot='select-trigger']")].find((node) =>
    node.textContent?.includes("Energy"),
  );
  expect(energy?.hasAttribute("disabled") || energy?.getAttribute("aria-disabled") === "true").toBe(false);
  expect(energy?.getAttribute("data-size")).toBe("sm");
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
    node.textContent?.includes("Dose Volume"),
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
  expect(onOpenView).toHaveBeenCalledWith("dose_volume");
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
    .mock.calls.filter(([command]) => command === "scan_kit_start")
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
  expect(source?.textContent).not.toContain("Chamber or plan");
  expect(source?.textContent).not.toContain("One row per millisecond");
  const xy = [...(source?.querySelectorAll("[data-slot='select-trigger']") ?? [])].find((node) =>
    node.textContent?.includes("Position"),
  );
  expect(xy?.querySelector("[title]")?.getAttribute("title")).toBe("Chamber or plan");
  const timeslice = [...(source?.querySelectorAll("[data-slot='toggle-group-item']") ?? [])].find(
    (node) => node.textContent?.trim() === "Timeslice",
  );
  expect(timeslice?.getAttribute("title")).toBe("One row per millisecond");
  const row = [...(source?.querySelectorAll(":scope > div") ?? [])].find(
    (node) => node.querySelectorAll("[data-slot='checkbox']").length === 3,
  );
  expect(row?.className).toContain("flex-row");
  const labels = [...(row?.querySelectorAll("label") ?? [])].map((node) => node.textContent?.trim());
  expect(labels).toEqual(["Plan", "IC1", "IC2"]);
  expect(row?.querySelector(".lucide-target")).not.toBeNull();
  expect(row?.querySelectorAll(".lucide-zap").length).toBe(2);
  const checks = [...(row?.querySelectorAll("[data-slot='checkbox']") ?? [])];
  expect(checks.map((node) => node.getAttribute("aria-checked"))).toEqual(["true", "true", "false"]);
});

it("adds and removes a segment through the plot options", async () => {
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
        onOpenView={() => undefined}
      />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const remove = [...host.querySelectorAll("button")].find(
    (button) => button.getAttribute("aria-label") === "Remove Rank",
  );
  expect(remove?.textContent?.trim()).toBe("");
  expect(remove?.querySelector("svg")).not.toBeNull();
  await act(async () => {
    remove?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const starts = () =>
    vi
      .mocked(invoke)
      .mock.calls.filter(([command]) => command === "scan_kit_start")
      .map(([, args]) => args as { options?: { segments?: string } });
  const dropped = JSON.parse(starts()[starts().length - 1]?.options?.segments ?? "null") as { kind: string }[];
  expect(dropped.map((item) => item.kind)).toEqual(["beam"]);
  const add = [...host.querySelectorAll("button")].find(
    (button) => button.getAttribute("aria-label") === "Add segment",
  );
  await act(async () => {
    add?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  const restored = JSON.parse(starts()[starts().length - 1]?.options?.segments ?? "null") as { kind: string }[];
  expect(restored.map((item) => item.kind)).toEqual(["beam", "rank"]);
});

it("keeps the playback bar on one row and arms it from the checkbox", async () => {
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
        onOpenView={() => undefined}
      />,
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });

  const bar = host.querySelector("[data-slot='scrub-bar']");
  expect(bar).not.toBeNull();
  expect(bar?.className).toContain("flex-row");
  expect(bar?.parentElement?.querySelectorAll("[data-slot='scrub-bar']").length).toBe(1);
  const box = bar?.querySelector("[data-slot='checkbox']");
  expect(box?.getAttribute("aria-checked")).toBe("false");
  const range = bar?.querySelector("input[type='range']");
  const skip = bar?.querySelector("[aria-label='Skip to start']");
  expect((range as HTMLInputElement | null)?.disabled).toBe(true);
  expect((skip as HTMLButtonElement | null)?.disabled).toBe(true);
  expect(bar?.querySelector("[data-slot='select-trigger']")).toBeNull();
  const speed = bar?.querySelector("[aria-label='Speed']");
  const windowMode = bar?.querySelector("[aria-label='Window']");
  const choiceName = (node: Element) =>
    node.getAttribute("aria-label") || node.textContent?.trim() || "";
  const pressed = (group: Element | null | undefined) => {
    const button = group?.querySelector("button[aria-pressed='true']");
    return button == null ? undefined : choiceName(button);
  };
  const labels = (group: Element | null | undefined) =>
    [...(group?.querySelectorAll("[data-slot='toggle-group-item']") ?? [])].map(choiceName);
  expect(labels(speed)).toEqual(["1/10", "1×", "10×"]);
  expect(labels(windowMode)).toEqual(["1 s", "Before"]);
  const before = [...(windowMode?.querySelectorAll("button") ?? [])].find(
    (item) => choiceName(item) === "Before",
  );
  expect(before?.textContent?.trim()).toBe("");
  expect(before?.querySelector("svg")).not.toBeNull();
  expect(pressed(speed)).toBe("1×");
  expect(pressed(windowMode)).toBe("1 s");
  const marks = () => [...(bar?.querySelectorAll("[data-slot='slider-mark']") ?? [])];
  expect(marks().map((mark) => mark.getAttribute("data-passed"))).toEqual(["false", "false"]);
  expect((speed?.querySelector("button") as HTMLButtonElement | null)?.disabled).toBe(true);
  expect((windowMode?.querySelector("button") as HTMLButtonElement | null)?.disabled).toBe(true);

  await act(async () => {
    box?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });

  expect(box?.getAttribute("aria-checked")).toBe("true");
  expect((range as HTMLInputElement | null)?.disabled).toBe(false);
  expect((skip as HTMLButtonElement | null)?.disabled).toBe(false);
  expect((speed?.querySelector("button") as HTMLButtonElement | null)?.disabled).toBe(false);
  const clickChoice = (group: Element | null | undefined, text: string) => {
    const button = [...(group?.querySelectorAll("button") ?? [])].find(
      (item) => choiceName(item) === text,
    );
    if (!(button instanceof HTMLButtonElement)) {
      throw new Error(`missing ${text}: ${[...(group?.querySelectorAll("button") ?? [])].map((item) => item.textContent).join("|")}`);
    }
    button.click();
  };
  await act(async () => {
    clickChoice(speed, "10×");
  });
  expect(pressed(speed)).toBe("10×");
  await act(async () => {
    clickChoice(windowMode, "Before");
  });
  expect(pressed(speed)).toBe("10×");
  expect(pressed(windowMode)).toBe("Before");

  const queued: FrameRequestCallback[] = [];
  const realFrame = window.requestAnimationFrame;
  window.requestAnimationFrame = (callback) => {
    queued.push(callback);
    return queued.length;
  };
  try {
    const play = () => bar?.querySelector("[aria-label='Play'], [aria-label='Pause']");
    await act(async () => {
      play()?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(play()?.getAttribute("aria-label")).toBe("Pause");

    await act(async () => {
      bar
        ?.querySelector("[aria-label='Skip to end']")
        ?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(play()?.getAttribute("aria-label")).toBe("Play");
    const thumb = bar?.querySelector("input[type='range']") as HTMLInputElement | null;
    expect(Number(thumb?.value)).toBeCloseTo(12.5);
    expect(marks().map((mark) => mark.getAttribute("data-passed"))).toEqual(["true", "true"]);
    expect(queued.length).toBeGreaterThan(0);
  } finally {
    window.requestAnimationFrame = realFrame;
  }
});

it("plays the timeslice window without reloading each step", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(() => {
    root = createRoot(host);
    root.render(
      <AnalysisView
        viewId="timeslice_replay"
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
  const starts = () =>
    vi.mocked(invoke).mock.calls.filter(([command]) => command === "scan_kit_start").length;
  const bar = host.querySelector("[data-slot='scrub-bar']");
  await act(async () => {
    bar?.querySelector("[data-slot='checkbox']")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 400));
  });
  const armed = starts();
  expect(armed).toBeGreaterThan(1);
  expect(followPlot).toHaveBeenCalledWith(true, 0, 0, true);

  const queued: FrameRequestCallback[] = [];
  const realFrame = window.requestAnimationFrame;
  window.requestAnimationFrame = (callback) => {
    queued.push(callback);
    return queued.length;
  };
  try {
    await act(async () => {
      bar?.querySelector("[aria-label='Play']")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    await act(async () => {
      let now = performance.now();
      for (let step = 0; step < 4; step += 1) {
        const batch = queued.splice(0, queued.length);
        now += 200;
        for (const frame of batch) {
          frame(now);
        }
      }
    });
    expect(starts()).toBe(armed);
    const highs = followPlot.mock.calls.map((call) => call[2] as number);
    expect(Math.max(...highs)).toBeGreaterThan(0.1);
    await act(async () => {
      bar?.querySelector("[aria-label='Pause']")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 300));
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 300));
    });
    expect(starts()).toBe(armed + 1);
  } finally {
    window.requestAnimationFrame = realFrame;
  }
});

it("keeps the timeslice picture when the next payload is empty", async () => {
  const invokeMock = vi.mocked(invoke);
  const original = invokeMock.getMockImplementation();
  let polls = 0;
  invokeMock.mockImplementation(async (command: string) => {
    if (command === "scan_kit_cancel") {
      return null;
    }
    if (command === "scan_kit_start") {
      return { task: 1, generation: 1 };
    }
    if (command !== "scan_kit_poll") {
      throw new Error(command);
    }
    polls += 1;
    const header = {
      title: "Timeslice Replay",
      controls: [
        {
          id: "scrub",
          label: "Timeline",
          kind: "scrub",
          options: [],
          value: JSON.stringify({ on: false, at: 0, end: 12.5, speed: 1, window: "second" }),
        },
      ],
      table: null,
      panels: polls === 1 ? [{}] : [],
    };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const bytes = new Uint8Array(4 + json.length);
    new DataView(bytes.buffer).setUint32(0, json.length, true);
    bytes.set(json, 4);
    return pollFrame(bytes);
  });
  try {
    const host = document.createElement("div");
    document.body.append(host);
    await act(() => {
      root = createRoot(host);
      root.render(
        <AnalysisView
          viewId="timeslice_replay"
          folder="C:/data"
          sessions={[{ id: "a", note: "" }]}
          onBack={() => undefined}
          onOpenView={() => undefined}
        />,
      );
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 250));
    });
    const frame = () => host.querySelector("canvas")?.parentElement;
    expect(frame()?.className).toContain("relative");
    expect(host.textContent).not.toContain("Loading plot");

    await act(async () => {
      host.querySelector("[data-slot='scrub-bar'] [data-slot='checkbox']")?.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 400));
    });
    expect(polls).toBeGreaterThan(1);
    expect(frame()?.className).toContain("relative");
    expect(frame()?.className).not.toContain("hidden");
    expect(host.textContent).not.toContain("Loading plot");
  } finally {
    if (original != null) {
      invokeMock.mockImplementation(original);
    }
  }
});
