import { invoke } from "@tauri-apps/api/core";
import { FolderOpen } from "lucide-react";
import { createElement } from "react";

import { toast } from "@/components/ui/toast";
import { appendLog } from "@/debug-log";

const logged = new Set<string>();

function textOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function fileName(path: string): string {
  const parts = path.split(/[/\\]/).filter((part) => part.length > 0);
  return parts[parts.length - 1] ?? path;
}

/** Record a failure in the debug log without covering the window. */
export function logError(error: unknown): void {
  appendLog("ERROR", "app", textOf(error));
}

/** Status message. It floats above the window and does not move the layout. */
export function notify(message: string, id?: string): void {
  toast.add({ id, title: message });
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
  toast.add({
    id,
    type: "error",
    title: text,
    timeout: 8000,
    priority: "high",
  });
}

/** Success after a file or folder was written, with a way to open that place. */
export function notifySaved(
  path: string,
  options: { title: string; description?: string; folder?: boolean },
): void {
  toast.add({
    type: "success",
    title: options.title,
    description: createElement(
      "span",
      { className: "block truncate", title: path },
      options.description ?? fileName(path),
    ),
    timeout: 12000,
    actionProps: {
      children: [
        createElement(FolderOpen, { key: "icon", "aria-hidden": true }),
        options.folder ? "Open folder" : "Show in folder",
      ],
      onClick() {
        void invoke("scan_kit_reveal", { path }).catch((error: unknown) => notifyError(error));
      },
    },
  });
}

export function dismissNotice(id?: string): void {
  toast.close(id);
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
