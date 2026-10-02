import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { ConfigTuning } from "./ConfigTuning";

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
                  id: "0#text",
                  label: "Devices",
                  kind: "string",
                  value: "",
                  dead: false,
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
});

it("tunes the open devices file from the selected sessions", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<ConfigTuning folder="C:/data" selectedIds={["sess"]} />);
  });
  expect(document.body.textContent).toContain("IC1/IC2");
  expect(document.body.textContent).toContain("No .md5 sidecar");
  const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Tune");
  expect(button).toBeTruthy();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_config_tune",
    expect.objectContaining({
      workflow: "sigma_tuning",
      dataDir: "C:/data",
      sessionIds: ["sess"],
    }),
  );
  expect(document.body.textContent).toContain("Updated 1 beam_sigma");
});
