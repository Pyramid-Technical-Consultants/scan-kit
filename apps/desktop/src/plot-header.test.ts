import { expect, it } from "vitest";

import { backingSize, plotHeader, sameChrome } from "./plot-header";

function payload(header: object, marks: number[]): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(4 + json.length + marks.length);
  new DataView(bytes.buffer).setUint32(0, json.length, true);
  bytes.set(json, 4);
  bytes.set(marks, 4 + json.length);
  return bytes;
}

it("reads the JSON header in front of the mark bytes", () => {
  const bytes = payload({ title: "Dose", controls: [], table: null, panels: [{}] }, [9, 9, 9]);
  const header = plotHeader(bytes.subarray(0));
  expect(header.title).toBe("Dose");
  expect(header.panels).toHaveLength(1);
  const shifted = new Uint8Array(bytes.length + 3);
  shifted.set(bytes, 3);
  expect(plotHeader(shifted.subarray(3)).title).toBe("Dose");
});

it("reads a string option and a noted option", () => {
  const header = plotHeader(
    payload(
      {
        title: "Dose",
        controls: [
          {
            id: "y",
            label: "Y",
            group: "Data Source",
            options: [
              "Energy",
              {
                id: "dose_error",
                label: "Dose Error (%)",
                detail: "Measured against target",
                icon: "dose_error",
              },
            ],
            value: "Dose Error (%)",
          },
        ],
        table: null,
        panels: [],
      },
      [],
    ).subarray(0),
  );
  expect(header.controls[0]?.group).toBe("Data Source");
  expect(header.controls[0]?.kind).toBe("");
  expect(header.controls[0]?.options[0]).toEqual({ id: "Energy", label: "Energy", detail: "", icon: "" });
  expect(header.controls[0]?.options[1]?.detail).toBe("Measured against target");
  expect(header.controls[0]?.options[1]?.icon).toBe("dose_error");
});

it("rejects a payload shorter than its header", () => {
  expect(() => plotHeader(payload({ title: "x" }, []).subarray(0, 8))).toThrow("truncated");
});

it("sizes the canvas in device pixels", () => {
  expect(backingSize(400.4, 300, 2)).toEqual({ width: 801, height: 600 });
  expect(backingSize(4, 4, 0)).toEqual({ width: 16, height: 16 });
});

it("keeps the chrome when only the plotted marks grew", () => {
  const current = plotHeader(
    payload(
      {
        title: "Dose",
        controls: [{ id: "y", label: "Y", value: "dose", options: [] }],
        table: { columns: ["a"], rows: [["1"]] },
        panels: [],
      },
      [1],
    ).subarray(0),
  );
  const grown = plotHeader(
    payload(
      {
        title: "Dose",
        controls: [{ id: "y", label: "Y", value: "dose", options: [] }],
        table: { columns: ["a"], rows: [["1"]] },
        panels: [{}],
        quality: "partial",
      },
      [1, 2, 3],
    ).subarray(0),
  );
  expect(sameChrome(current, grown)).toBe(true);
  expect(sameChrome(current, { ...grown, title: "Current" })).toBe(false);
  expect(sameChrome(null, grown)).toBe(false);
});
