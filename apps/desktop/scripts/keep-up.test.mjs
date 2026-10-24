import assert from "node:assert/strict";
import test from "node:test";

import { nextAttempt, shouldRestart } from "./keep-up.mjs";

test("a crash restarts and a clean quit or Ctrl+C does not", () => {
  assert.equal(shouldRestart(101, null, false, false), true);
  assert.equal(shouldRestart(1, null, false, false), true);
  assert.equal(shouldRestart(0, null, false, false), false);
  assert.equal(shouldRestart(0, null, false, true), true);
  assert.equal(shouldRestart(0, null, true, true), false);
  assert.equal(shouldRestart(130, null, false, true), false);
  assert.equal(shouldRestart(0xc000013a, null, false, true), false);
  assert.equal(shouldRestart(null, "SIGINT", false, true), false);
});

test("a session that stayed up can crash again, and a failed start gives up", () => {
  assert.equal(nextAttempt(101, null, false, 20_000, 3, false).action, "restart");
  assert.equal(nextAttempt(101, null, false, 20_000, 3, false).fast, 0);
  assert.equal(nextAttempt(1, null, false, 100, 3, false).action, "give-up");
  assert.equal(nextAttempt(0, null, false, 100, 0, false).action, "stop");
});
