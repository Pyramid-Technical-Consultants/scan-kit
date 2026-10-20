import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { ConfigTuning, fieldBoxClass, sourceColumn } from "./ConfigTuning";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_config_catalog") {
      return {
        config_dir: "C:/config",
        hide_unused: false,
        workflows: [
          {
            id: "sigma_tuning",
            name: "Sigma Tuning",
            description: "Fit spot-size factors to session data",
            sessions: true,
            params: [
              {
                key: "sigma_tolerance_percent",
                label: "Sigma Tolerance (%)",
                kind: "float",
                default: 20,
              },
            ],
          },
          {
            id: "config_diff",
            name: "Config Diff",
            description: "Diff all config files against selected sessions",
            sessions: true,
            diff: true,
            params: [],
          },
        ],
      };
    }
    if (command === "scan_kit_config_open") {
      return { path: "C:/config", files: ["devices.xml"], hide_unused: false };
    }
    if (command === "scan_kit_config_form") {
      return {
        path: "C:/config/devices.xml",
        xml: "<devices/>",
        integrity: { status: "HASH_FILE_NOT_EXIST", label: "No .md5 sidecar" },
        form: {
          root: "devices",
          nodes: [
            {
              kind: "fields",
              fields: [
                {
                  id: "0@type",
                  label: "Type",
                  kind: "string",
                  value: "device",
                  dead: false,
                },
                {
                  id: "0@version",
                  label: "Version",
                  kind: "int",
                  value: "1",
                  dead: false,
                },
              ],
            },
            {
              kind: "section",
              title: "IC 1 X",
              collapsible: true,
              nodes: [
                {
                  kind: "table",
                  id: "sigma",
                  title: "Beam sigma conversions (72 rows)",
                  columns: [{ name: "K0", label: "K0" }],
                  rows: [["3.002"]],
                },
              ],
            },
          ],
        },
      };
    }
    if (command === "scan_kit_config_diff") {
      return {
        base: "C:/config",
        edited: "map2map/devices.xml",
        summary: "sess: 1 of 2 differ",
        warnings: [],
        sessions: [
          {
            id: "sess",
            root: "C:/data/sess/config",
            files: [
              {
                rel: "map2map/devices.xml",
                status: "changed",
                edited: true,
                adds: 1,
                dels: 1,
                lines: [
                  { kind: "del", old_no: 1, text: "<a/>" },
                  { kind: "add", new_no: 1, text: "<b/>" },
                ],
              },
              { rel: "nozzle/n.json", status: "same", adds: 0, dels: 0, lines: [] },
            ],
          },
          {
            id: "other",
            warning: "No config/ tree in session other.",
            files: [],
          },
        ],
        changed: false,
        columns: [],
        rows: [],
      };
    }
    if (command === "scan_kit_config_tune") {
      return {
        xml: "<devices/>",
        form: { root: "devices", nodes: [] },
        summary: "Updated 1 beam_sigma band(s) from sess. Save the configuration to write devices.xml.",
        warnings: [],
        columns: ["Device", "K0 After"],
        rows: [[{ text: "IC_1_X" }, { text: "5.05", tone: "change" }]],
        changed: true,
      };
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
  vi.useRealTimers();
  vi.restoreAllMocks();
});

it("previews a tune, then applies it after confirmation", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const confirm = vi.fn(() => true);
  window.confirm = confirm;
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(
      <ConfigTuning folder="C:/data" sessions={[{ id: "sess", note: "" }]} selectedIds={["sess"]} />,
    );
  });
  expect(document.body.textContent).toContain("Fit spot-size factors");
  expect(document.querySelector("[aria-label='No .md5 sidecar']")).toBeTruthy();
  const version = [...document.querySelectorAll("[data-slot=field]")].find((node) =>
    node.textContent?.includes("Version"),
  );
  expect(version?.className).toContain("w-28");
  expect(document.querySelector("summary")).toBeNull();
  expect(document.body.textContent).not.toContain("Beam sigma");
  const chamber = [...document.body.querySelectorAll("button")].find((item) => item.textContent?.includes("IC 1 X"));
  expect(chamber?.getAttribute("aria-expanded")).toBe("false");
  await act(async () => {
    chamber?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(document.body.textContent).toContain("Beam sigma");
  await act(async () => {
    await vi.advanceTimersByTimeAsync(200);
  });
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_config_open",
    expect.objectContaining({ dataDir: "C:/data", sessionId: "sess" }),
  );
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_config_tune",
    expect.objectContaining({
      workflow: "sigma_tuning",
      dataDir: "C:/data",
      sessionIds: ["sess"],
    }),
  );
  expect(document.body.textContent).toContain("Updated 1 beam_sigma");
  const handle = document.querySelector("[aria-label='Resize configuration']");
  const previewHandle = document.querySelector("[aria-label='Resize preview']");
  const apply = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Apply");
  const save = document.querySelector("[aria-label='Save folder']");
  expect(apply).toBeTruthy();
  expect(handle?.previousElementSibling?.contains(save ?? null)).toBe(true);
  expect(handle?.nextElementSibling?.contains(save ?? null)).toBe(false);
  expect(handle?.nextElementSibling?.contains(apply ?? null)).toBe(false);
  expect(previewHandle?.nextElementSibling?.contains(apply ?? null)).toBe(true);
  expect(previewHandle?.previousElementSibling?.contains(apply ?? null)).toBe(false);
  const tuneCalls = vi.mocked(invoke).mock.calls.filter((call) => call[0] === "scan_kit_config_tune").length;
  await act(async () => {
    apply?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(confirm).toHaveBeenCalledWith(expect.stringContaining("beam sigma K0"));
  expect(vi.mocked(invoke).mock.calls.filter((call) => call[0] === "scan_kit_config_tune").length).toBe(tuneCalls);
  expect(document.body.textContent).not.toContain("Version");
  vi.useRealTimers();
});

it("diffs the loaded config against each selected session", async () => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(
      <ConfigTuning
        folder="C:/data"
        sessions={[{ id: "sess", note: "" }, { id: "other", note: "" }]}
        selectedIds={["sess", "other"]}
      />,
    );
  });
  const diff = [...document.body.querySelectorAll("button")].find((item) =>
    item.textContent?.includes("Config Diff"),
  );
  await act(async () => {
    diff?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await vi.advanceTimersByTimeAsync(200);
  });
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_config_diff",
    expect.objectContaining({
      configDir: "C:/config",
      file: "devices.xml",
      dataDir: "C:/data",
      sessionIds: ["sess", "other"],
    }),
  );
  expect(document.body.textContent).toContain("sess: 1 of 2 differ");
  expect(document.body.textContent).toContain("map2map/devices.xml");
  expect(document.body.textContent).toContain("Modified");
  expect(document.body.textContent).toContain("unsaved");
  expect(document.body.textContent).toContain("<b/>");
  expect(document.body.textContent).toContain("1 identical file");
  expect(document.body.textContent).toContain("No config/ tree in session other.");
  expect([...document.body.querySelectorAll("button")].some((item) => item.textContent === "Apply")).toBe(false);
  vi.useRealTimers();
});

it("keeps short xml values in a short field", () => {
  expect(fieldBoxClass({ kind: "int", value: "1" })).toBe("w-28");
  expect(fieldBoxClass({ kind: "string", value: "device" })).toBe("w-36");
  expect(fieldBoxClass({ kind: "string", value: "x".repeat(48) })).toBe("w-full max-w-lg");
});

it("maps a hidden unused column back to the full row", () => {
  const columns = [{ dead: true }, { dead: false }, { dead: true }, { dead: false }];
  expect(sourceColumn(columns, 0, true)).toBe(1);
  expect(sourceColumn(columns, 1, true)).toBe(3);
  expect(sourceColumn(columns, 1, false)).toBe(1);
});
