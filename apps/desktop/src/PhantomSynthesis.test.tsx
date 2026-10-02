import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

import { PhantomSynthesis } from "./PhantomSynthesis";

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: vi.fn(async () => "C:/out"),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(async (command: string) => {
    if (command === "scan_kit_phantom_catalog") {
      return {
        phantoms: [{ value: "slabs", label: "Water box, bone and lung slabs" }],
        positions: ["HFS"],
        energies: [120, 130, 140],
        defaults: {
          phantom: "slabs",
          position: "HFS",
          pixel: 2,
          slice: 2,
          energies: [140, 130, 120],
          gantry: 0,
          couch: 0,
          spot_pitch: 6,
          range_shifter_wet: 0,
          mu_per_spot: 0.02,
          fractions: 1,
        },
        save_dir: null,
      };
    }
    if (command === "scan_kit_phantom_preview") {
      return {
        summary: "CT 80×80×70 voxels at 2×2×2 mm; 3 layers, 75 spots, 1.5 MU per fraction.",
      };
    }
    if (command === "scan_kit_write_phantom") {
      return {
        status: "Wrote 70 CT slices, RTSTRUCT and RT Ion Plan to C:/out/synthetic_slabs.",
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

it("writes a phantom study from the catalog defaults", async () => {
  const host = document.createElement("div");
  document.body.append(host);
  await act(async () => {
    root = createRoot(host);
    root.render(<PhantomSynthesis />);
  });
  expect(document.body.textContent).toContain("CT 80×80×70");
  expect(document.body.textContent).toContain("Water box, bone and lung slabs");
  const button = [...document.body.querySelectorAll("button")].find((item) => item.textContent === "Write DICOM");
  expect(button).toBeTruthy();
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  expect(open).toHaveBeenCalled();
  expect(invoke).toHaveBeenCalledWith(
    "scan_kit_write_phantom",
    expect.objectContaining({
      parent: "C:/out",
      params: expect.objectContaining({ phantom: "slabs", energies: [140, 130, 120] }),
    }),
  );
  expect(document.body.textContent).toContain("Wrote 70 CT slices");
});
