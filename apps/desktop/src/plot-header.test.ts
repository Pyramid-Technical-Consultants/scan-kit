import { expect, it } from "vitest";

import { backingSize, plotHeader } from "./plot-header";

function payload(header: object, marks: number[]): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(4 + json.length + marks.length);
  new DataView(bytes.buffer).setUint32(0, json.length, true);
  bytes.set(json, 4);
  bytes.set(marks, 4 + json.length);
  return bytes;
}

it("reads the JSON header in front of the mark bytes", () => {
  const bytes = payload({ title: "Dose", controls: [], table: null, samples: [1], panels: [{}] }, [9, 9, 9]);
  const header = plotHeader(bytes.subarray(0));
  expect(header.title).toBe("Dose");
  expect(header.panels).toHaveLength(1);
  const shifted = new Uint8Array(bytes.length + 3);
  shifted.set(bytes, 3);
  expect(plotHeader(shifted.subarray(3)).samples).toEqual([1]);
});

it("rejects a payload shorter than its header", () => {
  expect(() => plotHeader(payload({ title: "x" }, []).subarray(0, 8))).toThrow("truncated");
});

it("sizes the canvas in device pixels", () => {
  expect(backingSize(400.4, 300, 2)).toEqual({ width: 801, height: 600 });
  expect(backingSize(4, 4, 0)).toEqual({ width: 16, height: 16 });
});
