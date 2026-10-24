const MAX_LOG_LINES = 5000;

export function formatLogLine(
  level: string,
  source: string,
  message: string,
  now = new Date(),
): string {
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  const sourcePart = source.trim() || "app";
  const text = message.replace(/\n+$/u, "");
  return `${hh}:${mm}:${ss} [${level}] [${sourcePart}] ${text}`;
}

const lines: string[] = [];
const listeners = new Set<() => void>();

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

export function logLines(): readonly string[] {
  return lines;
}

export function appendLog(level: string, source: string, message: string): void {
  const text = message.replace(/\n+$/u, "");
  if (text === "") {
    return;
  }
  for (const part of text.split("\n")) {
    if (part !== "") {
      lines.push(formatLogLine(level, source, part));
    }
  }
  if (lines.length > MAX_LOG_LINES) {
    lines.splice(0, lines.length - MAX_LOG_LINES);
  }
  notify();
}

export function clearLog(): void {
  lines.length = 0;
  notify();
}

export function subscribeLog(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function formatArg(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value instanceof Error) {
    return value.stack ?? value.message;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

let installed = false;

/** Mirror console output and uncaught errors into the debug log. Safe to call more than once. */
export function installDebugLog(): void {
  if (installed) {
    return;
  }
  installed = true;
  const methods = [
    ["debug", "DEBUG"],
    ["log", "INFO"],
    ["info", "INFO"],
    ["warn", "WARNING"],
    ["error", "ERROR"],
  ] as const;
  for (const [method, level] of methods) {
    const original = console[method].bind(console);
    console[method] = (...args: unknown[]) => {
      original(...args);
      appendLog(level, "console", args.map(formatArg).join(" "));
    };
  }
  window.addEventListener("error", (event) => {
    appendLog("ERROR", "uncaught", event.message);
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason: unknown = event.reason;
    const message = reason instanceof Error ? (reason.stack ?? reason.message) : String(reason);
    appendLog("ERROR", "uncaught", message);
  });
  appendLog("INFO", "launcher", "Debug log started");
}
