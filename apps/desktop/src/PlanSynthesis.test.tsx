import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { PlanSynthesis } from "./PlanSynthesis";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(),
  save: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_plan_catalog") {
      return {
        save_dir: null,
        templates: [
          {
            id: "zero_field",
            name: "Zero Field",
            description: "Every spot at (0, 0) for each energy layer.",
            params: [
              {
                key: "selected_energies",
                label: "Energy Layers (MeV)",
                kind: "energy_multiselect",
                field_set: "energy",
                default: [70, 250],
                presets: [{ label: "Select All", energies: [70, 250] }],
              },
              {
                key: "spots_per_layer",
                label: "Spots per Layer (spots)",
                kind: "int",
                field_set: "geometry",
                default: 2,
              },
              {
                key: "fast_axis",
                label: "Fast Axis",
                kind: "button_group",
                field_set: "geometry",
                default: "x",
                choices: [
                  { value: "x", label: "X" },
                  { value: "y", label: "Y" },
                ],
              },
            ],
          },
        ],
      };
    }
    if (command === "scan_kit_synthesize_plan") {
      return {
        summary: "1 layer · 2 spots · 0.04 MU total · est. 0.1 s delivery",
        filename: "ZeroField.csv",
        columns: ["#NO", "ENERGY(MeV)"],
        rows: [
          ["1", "250"],
          ["2", "250"],
        ],
        csv: "#NO,ENERGY(MeV)\n1,250\n2,250\n",
        written: null,
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

it("generates a plan from the catalog", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<PlanSynthesis />);
  });
  expect(document.body.textContent).toContain("Every spot at (0, 0)");
  const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Generate");
  expect(button).toBeTruthy();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_synthesize_plan",
    expect.objectContaining({ template: "zero_field" }),
  );
  expect(document.body.textContent).toContain("2 spots");
});
