import { toast } from "sonner";

import { appendLog } from "@/debug-log";

const logged = new Set<string>();

function textOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Record a failure in the debug log without covering the window. */
export function logError(error: unknown): void {
  appendLog("ERROR", "app", textOf(error));
}

/** Status message. It floats above the window and does not move the layout. */
export function notify(message: string, id?: string): void {
  toast(message, id == null ? undefined : { id });
}

/** Failure toast. A repeated notice with the same id is logged once. */
export function notifyError(error: unknown, id?: string): void {
  const text = textOf(error);
  const key = id == null ? null : `${id}\0${text}`;
  if (key == null || !logged.has(key)) {
    if (key != null) {
      logged.add(key);
    }
    appendLog("ERROR", "app", text);
  }
  toast.error(text, id == null ? { duration: 8000 } : { id, duration: 8000 });
}

export function dismissNotice(id?: string): void {
  toast.dismiss(id);
  if (id == null) {
    logged.clear();
    return;
  }
  for (const key of logged) {
    if (key.startsWith(`${id}\0`)) {
      logged.delete(key);
    }
  }
}
