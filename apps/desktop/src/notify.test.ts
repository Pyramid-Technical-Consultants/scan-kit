import { expect, it } from "vitest";

import { clearLog, logLines } from "./debug-log";
import { notifyError } from "./notify";

it("logs a repeated analysis failure once and a new action every time", () => {
  clearLog();
  notifyError("plot failed", "analysis");
  notifyError("plot failed", "analysis");
  expect(logLines().filter((line) => line.includes("plot failed"))).toHaveLength(1);
  notifyError("folder failed");
  notifyError("folder failed");
  expect(logLines().filter((line) => line.includes("folder failed"))).toHaveLength(2);
});
