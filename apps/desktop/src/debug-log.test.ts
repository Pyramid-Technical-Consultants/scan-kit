import { expect, it } from "vitest";

import { appendLog, clearLog, formatLogLine, logLines } from "./debug-log";

it("formats a debug line like the 1.8 panel", () => {
  const line = formatLogLine(
    "ERROR",
    "scan_kit.views.foo",
    "something broke",
    new Date(2026, 5, 12, 14, 30, 1),
  );
  expect(line).toBe("14:30:01 [ERROR] [scan_kit.views.foo] something broke");
});

it("keeps the newest five thousand lines", () => {
  clearLog();
  for (let i = 0; i < 5002; i += 1) {
    appendLog("INFO", "app", `line ${i}`);
  }
  const kept = logLines();
  expect(kept).toHaveLength(5000);
  expect(kept[0]).toContain("line 2");
  expect(kept[kept.length - 1]).toContain("line 5001");
  clearLog();
});
