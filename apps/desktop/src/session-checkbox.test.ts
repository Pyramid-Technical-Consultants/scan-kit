import { expect, it } from "vitest";

import { checkboxRect, drawSessionCheckbox, type CheckPaint } from "./session-checkbox";

const paint: CheckPaint = {
  border: "#111",
  idle: "#222",
  mark: "#fff",
  header: "#333",
};

function context(): CanvasRenderingContext2D {
  return {
    beginPath() {},
    roundRect() {},
    fill() {},
    stroke() {},
    moveTo() {},
    lineTo() {},
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 1,
    lineCap: "butt",
    lineJoin: "miter",
  } as unknown as CanvasRenderingContext2D;
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
  for (const mode of ["off", "mixed", "on"] as const) {
    drawSessionCheckbox(context(), cell, mode, "#4af", paint);
  }
});
