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
            description: "IC1/IC2 σ K0 to fit ±tolerance band",
            params: [
              {
                key: "sigma_tolerance_percent",
                label: "Sigma Tolerance (%)",
                kind: "float",
                default: 20,
              },
            ],
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
    if (command === "scan_kit_config_tune") {
      return {
        xml: "<devices/>",
        form: { root: "devices", nodes: [] },
        summary: "Updated 1 beam_sigma band(s) from sess. Save the configuration to write devices.xml.",
        warnings: [],
        columns: ["Device", "K0 After"],
        rows: [["IC_1_X", "5.05"]],
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
    root.render(<ConfigTuning folder="C:/data" selectedIds={["sess"]} />);
  });
  expect(document.body.textContent).toContain("IC1/IC2");
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
  const apply = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Apply");
  const save = document.querySelector("[aria-label='Save folder']");
  expect(apply).toBeTruthy();
  expect(handle?.previousElementSibling?.contains(save ?? null)).toBe(true);
  expect(handle?.nextElementSibling?.contains(apply ?? null)).toBe(true);
  expect(handle?.nextElementSibling?.contains(save ?? null)).toBe(false);
  const tuneCalls = vi.mocked(invoke).mock.calls.filter((call) => call[0] === "scan_kit_config_tune").length;
  await act(async () => {
    apply?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(confirm).toHaveBeenCalledWith(expect.stringContaining("beam sigma K0"));
  expect(vi.mocked(invoke).mock.calls.filter((call) => call[0] === "scan_kit_config_tune").length).toBe(tuneCalls);
  expect(document.body.textContent).not.toContain("Version");
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
