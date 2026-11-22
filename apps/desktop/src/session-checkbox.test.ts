import { expect, it } from "vitest";

import { checkboxRect, drawSessionCheckbox, type CheckPaint } from "./session-checkbox";

const paint: CheckPaint = {
  border: "#111",
  idle: "#222",
  mark: "#fff",
  header: "#333",
};

function recorder() {
  const calls: string[] = [];
  const ctx = {
    beginPath() {
      calls.push("beginPath");
    },
    roundRect() {
      calls.push("roundRect");
    },
    fill() {
      calls.push(`fill:${ctx.fillStyle}`);
    },
    stroke() {
      calls.push(`stroke:${ctx.strokeStyle}`);
    },
    moveTo() {
      calls.push("moveTo");
    },
    lineTo() {
      calls.push("lineTo");
    },
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt" as const,
    lineJoin: "miter" as const,
  };
  return { calls, ctx: ctx as unknown as CanvasRenderingContext2D };
}

it("centers a 16px checkbox in the cell", () => {
  expect(checkboxRect({ x: 10, y: 20, width: 48, height: 32 })).toEqual({
    x: 26,
    y: 28,
    size: 16,
  });
});

it("draws an empty box, a dash, and a check", () => {
  const cell = { x: 0, y: 0, width: 16, height: 16 };

  const off = recorder();
  drawSessionCheckbox(off.ctx, cell, "off", "#4af", paint);
  expect(off.calls).toContain(`fill:${paint.idle}`);
  expect(off.calls).toContain(`stroke:${paint.border}`);
  expect(off.calls.filter((call) => call === "lineTo")).toHaveLength(0);

  const mixed = recorder();
  drawSessionCheckbox(mixed.ctx, cell, "mixed", "#4af", paint);
  expect(mixed.calls).toContain(`fill:${paint.header}`);
  expect(mixed.calls.filter((call) => call === "lineTo")).toHaveLength(1);
  expect(mixed.calls).toContain(`stroke:${paint.mark}`);

  const on = recorder();
  drawSessionCheckbox(on.ctx, cell, "on", "#4af", paint);
  expect(on.calls).toContain("fill:#4af");
  expect(on.calls.filter((call) => call === "lineTo")).toHaveLength(2);
  expect(on.calls).toContain(`stroke:${paint.mark}`);
});
