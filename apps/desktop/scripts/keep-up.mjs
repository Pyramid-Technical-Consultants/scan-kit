// Tauri's dev CLI treats an app crash, a dead Vite process, and a dropped
// file-watcher channel as a normal quit: it kills the other half and exits.
// This restarts that process. Closing the window (exit 0) and Ctrl+C still stop.
import { spawn } from "node:child_process";

const FAST_MS = 15_000;
const FAST_LIMIT = 4;
const CTRL_C_EXIT = 0xc000013a;

export function shouldRestart(code, signal, requestedStop, restartOnClean) {
  if (requestedStop) {
    return false;
  }
  if (signal === "SIGINT" || signal === "SIGTERM" || signal === "SIGBREAK") {
    return false;
  }
  if (code === 130 || code === 143 || code === CTRL_C_EXIT) {
    return false;
  }
  if (code === 0) {
    return restartOnClean;
  }
  return true;
}

export function nextAttempt(code, signal, requestedStop, elapsedMs, fastFailures, restartOnClean) {
  if (!shouldRestart(code, signal, requestedStop, restartOnClean)) {
    return { action: "stop", code: code ?? 0, fast: fastFailures };
  }
  const fast = elapsedMs < FAST_MS ? fastFailures + 1 : 0;
  if (fast >= FAST_LIMIT) {
    return { action: "give-up", code: code ?? 1, fast };
  }
  return { action: "restart", code: code ?? 1, fast };
}

function restartDelay(fast) {
  if (fast <= 0) {
    return 1000;
  }
  return Math.min(8000, 500 * 2 ** (fast - 1));
}

function exitLabel(code, signal) {
  if (signal) {
    return signal;
  }
  return `exit ${code ?? "unknown"}`;
}

function run(command, args) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: "inherit" });
    child.on("error", () => resolve({ code: 1, signal: null }));
    child.on("exit", (code, signal) => resolve({ code, signal }));
  });
}

function watchStop() {
  let requestedStop = false;
  let notify = () => {};
  const stopped = new Promise((resolve) => {
    notify = resolve;
  });
  const requestStop = () => {
    requestedStop = true;
    notify();
  };
  process.on("SIGINT", requestStop);
  process.on("SIGTERM", requestStop);
  if (process.platform === "win32") {
    process.on("SIGBREAK", requestStop);
  }
  return {
    get requested() {
      return requestedStop;
    },
    pause(ms) {
      return new Promise((resolve) => {
        const timer = setTimeout(resolve, ms);
        stopped.then(() => {
          clearTimeout(timer);
          resolve();
        });
      });
    },
  };
}

export async function keepUp(command, args, { name, restartOnClean = false } = {}) {
  const stop = watchStop();
  let fast = 0;
  for (;;) {
    const started = Date.now();
    const result = await run(command, args);
    const attempt = nextAttempt(
      result.code,
      result.signal,
      stop.requested,
      Date.now() - started,
      fast,
      restartOnClean,
    );
    if (attempt.action === "stop") {
      process.exit(stop.requested ? 130 : (attempt.code ?? 0));
    }
    if (attempt.action === "give-up") {
      console.error(
        `Scan Kit: ${name} stopped ${FAST_LIMIT} times before it stayed up. Leaving it down.`,
      );
      process.exit(attempt.code ?? 1);
    }
    fast = attempt.fast;
    console.error(
      `Scan Kit: ${name} stopped (${exitLabel(result.code, result.signal)}). Starting it again.`,
    );
    await stop.pause(restartDelay(fast));
    if (stop.requested) {
      process.exit(130);
    }
  }
}
