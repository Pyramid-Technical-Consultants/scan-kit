import { expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";

import { acceptReport, driveTask, hairlineFraction, parsePoll, type Report } from "./task-client";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

function frame(report: Report, payload: Uint8Array | null): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(report));
  const extra = payload?.byteLength ?? 0;
  const out = new Uint8Array(4 + json.length + extra);
  new DataView(out.buffer).setUint32(0, json.length, true);
  out.set(json, 4);
  if (payload != null) {
    out.set(payload, 4 + json.length);
  }
  return out;
}

function report(generation: number, finished: boolean): Report {
  return {
    task: 1,
    generation,
    phase: finished ? "done" : "parse",
    done: finished ? 2 : 1,
    total: 2,
    note: finished ? "2 of 2" : "1 of 2",
    finished,
  };
}

it("a partial poll then a final poll loads twice, and a stale generation does not", () => {
  const loads: Uint8Array[] = [];
  const generation = 4;
  for (const finished of [false, true]) {
    const payload = new Uint8Array([finished ? 2 : 1]);
    const parsed = parsePoll(frame(report(generation, finished), payload));
    if (acceptReport(parsed.report, 1, generation) && parsed.payload != null) {
      loads.push(parsed.payload);
    }
  }
  expect(loads.map((bytes) => bytes[0])).toEqual([1, 2]);
  const stale = parsePoll(frame(report(generation, false), new Uint8Array([9])));
  expect(acceptReport(stale.report, 1, generation + 1)).toBe(false);
});

it("polls the next slice without waiting for a paint", async () => {
  let polls = 0;
  vi.mocked(invoke).mockImplementation(async (command: string) => {
    if (command === "scan_kit_start") {
      return { task: 1, generation: 1 };
    }
    polls += 1;
    return frame(report(1, polls > 1), new Uint8Array([polls]));
  });
  const seen: number[] = [];
  await driveTask(
    {},
    (_update, payload) => {
      seen.push(payload?.[0] ?? 0);
    },
    () => false,
  );
  expect(seen).toEqual([1, 2]);
  expect(polls).toBe(2);
});

it("the hairline sweeps when the length is unknown and fills a fraction", () => {
  expect(hairlineFraction(0, 0)).toBeNull();
  expect(hairlineFraction(0, 5)).toBeNull();
  expect(hairlineFraction(1, 4)).toBe(0.25);
  expect(hairlineFraction(9, 4)).toBe(1);
});
