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
  if (!(total > 0)) {
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
): Promise<void> {
  const started = await invoke<{ task: number; generation: number }>("scan_kit_start", args);
  if (stop()) {
    await invoke("scan_kit_cancel", { task: started.task });
    return;
  }
  for (;;) {
    if (stop()) {
      await invoke("scan_kit_cancel", { task: started.task });
      return;
    }
    const raw = await invoke<ArrayBuffer | Uint8Array>("scan_kit_poll", { task: started.task });
    const parsed = parsePoll(bytesOf(raw));
    if (!acceptReport(parsed.report, started.task, started.generation)) {
      continue;
    }
    if (stop()) {
      await invoke("scan_kit_cancel", { task: started.task });
      return;
    }
    onUpdate(parsed.report, parsed.payload);
    if (parsed.report.finished) {
      if (parsed.report.phase === "failed") {
        throw new Error(parsed.report.note.length > 0 ? parsed.report.note : "failed");
      }
      return;
    }
  }
}
