import { invoke } from "@tauri-apps/api/core";

export type Report = {
  task: number;
  generation: number;
  phase: string;
  done: number;
  total: number;
  note: string;
  finished: boolean;
};

export function bytesOf(result: ArrayBuffer | Uint8Array): Uint8Array {
  return result instanceof Uint8Array ? result : new Uint8Array(result);
}

/// `u32` little-endian JSON length, the report, then an optional payload.
/// `scan-kit-core` `encode_poll` writes this. Rename both together.
export function parsePoll(bytes: Uint8Array): { report: Report; payload: Uint8Array | null } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = view.getUint32(0, true);
  const report = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length))) as Report;
  const rest = bytes.subarray(4 + length);
  return { report, payload: rest.byteLength > 0 ? rest : null };
}

/// `null` is an unknown length, so the hairline sweeps. A fraction is a width.
export function hairlineFraction(done: number, total: number): number | null {
  if (!(total > 0) || done <= 0) {
    return null;
  }
  return Math.min(1, Math.max(0, done / total));
}

export function acceptReport(report: Report, task: number, generation: number): boolean {
  return report.task === task && report.generation === generation;
}

export async function driveTask(
  args: Record<string, unknown>,
  onUpdate: (report: Report, payload: Uint8Array | null) => void,
  stop: () => boolean,
): Promise<boolean> {
  const started = await invoke<{ task: number; generation: number; superseded?: boolean }>(
    "scan_kit_start",
    args,
  );
  // A superseded ticket belongs to another view's task — polling it would
  // settle this view on unrelated work, and it is not ours to cancel.
  if (started.superseded) {
    return false;
  }
  if (stop()) {
    await invoke("scan_kit_cancel", { task: started.task, generation: started.generation });
    return true;
  }
  let misses = 0;
  for (;;) {
    if (stop()) {
      await invoke("scan_kit_cancel", { task: started.task, generation: started.generation });
      return true;
    }
    let parsed;
    try {
      const raw = await invoke<ArrayBuffer | Uint8Array>("scan_kit_poll", { task: started.task, generation: started.generation });
      parsed = parsePoll(bytesOf(raw));
      // A malformed frame must throw inside the cancel guard — dereferencing
      // a bad report outside it leaves the backend task running unpolled.
      if (parsed.report == null || typeof parsed.report !== "object") {
        throw new Error("bad poll frame");
      }
    } catch (reason) {
      // The backend task keeps running without a poller — cancel it, then propagate.
      await invoke("scan_kit_cancel", { task: started.task, generation: started.generation }).catch(() => {});
      throw reason;
    }
    // Supersession can land while the poll is in flight — the resolve then
    // carries the old generation's frame and acceptReport throws a routine
    // cancellation as a user-visible error. Check before accepting.
    if (stop()) {
      await invoke("scan_kit_cancel", { task: started.task, generation: started.generation });
      return true;
    }
    if (!acceptReport(parsed.report, started.task, started.generation)) {
      misses += 1;
      if (!parsed.report.finished && misses < 8) {
        await new Promise((resolve) => window.setTimeout(resolve, 40));
        continue;
      }
      throw new Error("the task was cancelled or superseded");
    }
    misses = 0;
    if (stop()) {
      await invoke("scan_kit_cancel", { task: started.task, generation: started.generation });
      return true;
    }
    try {
      onUpdate(parsed.report, parsed.payload);
    } catch (reason) {
      // Same shape as a poll error — do not leave the backend task running.
      await invoke("scan_kit_cancel", { task: started.task, generation: started.generation }).catch(() => {});
      throw reason;
    }
    if (parsed.report.finished) {
      if (parsed.report.phase === "failed") {
        throw new Error(parsed.report.note.length > 0 ? parsed.report.note : "failed");
      }
      if (parsed.report.phase !== "done") {
        throw new Error(`the task ended ${parsed.report.phase}`);
      }
      return true;
    }
    // An unpaced loop spins thousands of IPC round-trips per second for the
    // task's whole duration — each one serializes a report and schedules a
    // setState. 40 ms keeps progress live without saturating the bridge.
    await new Promise((resolve) => window.setTimeout(resolve, 40));
  }
}
