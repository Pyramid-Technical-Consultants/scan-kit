type ViewChoice = {
  id: string;
  label: string;
  detail: string;
  icon: string;
};

export type ViewControl = {
  id: string;
  label: string;
  options: ViewChoice[];
  value: string;
  group: string;
  kind: string;
};

type ViewTable = {
  columns: string[];
  rows: string[][];
};

export type PlotHeader = {
  title: string;
  controls: ViewControl[];
  table: ViewTable | null;
  samples: number[];
  panels: unknown[];
  quality: string;
};

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function viewChoice(raw: unknown): ViewChoice | null {
  if (typeof raw === "string") {
    return raw.length === 0 ? null : { id: raw, label: raw, detail: "", icon: "" };
  }
  if (raw == null || typeof raw !== "object") {
    return null;
  }
  const row = raw as { id?: unknown; label?: unknown; detail?: unknown; icon?: unknown };
  const label = text(row.label);
  if (label.length === 0) {
    return null;
  }
  const id = text(row.id);
  return {
    id: id.length > 0 ? id : label,
    label,
    detail: text(row.detail),
    icon: text(row.icon),
  };
}

function viewControl(raw: unknown): ViewControl {
  const row = raw != null && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const options = Array.isArray(row.options)
    ? row.options.flatMap((item) => {
        const choice = viewChoice(item);
        return choice == null ? [] : [choice];
      })
    : [];
  return {
    id: text(row.id),
    label: text(row.label),
    options,
    value: text(row.value),
    group: text(row.group),
    kind: text(row.kind),
  };
}

/** True when the chrome can stay put and only the marks grew. */
export function sameChrome(current: PlotHeader | null, next: PlotHeader): boolean {
  if (current == null) {
    return false;
  }
  if (current.title !== next.title || current.samples.length !== next.samples.length) {
    return false;
  }
  if (current.controls.length !== next.controls.length) {
    return false;
  }
  for (let index = 0; index < current.controls.length; index += 1) {
    const left = current.controls[index];
    const right = next.controls[index];
    if (left == null || right == null || left.id !== right.id || left.value !== right.value) {
      return false;
    }
  }
  return sameTable(current.table, next.table);
}

function sameTable(left: ViewTable | null, right: ViewTable | null): boolean {
  if (left == null || right == null) {
    return left == null && right == null;
  }
  if (left.columns.length !== right.columns.length || left.rows.length !== right.rows.length) {
    return false;
  }
  for (let index = 0; index < left.columns.length; index += 1) {
    if (left.columns[index] !== right.columns[index]) {
      return false;
    }
  }
  for (let row = 0; row < left.rows.length; row += 1) {
    const a = left.rows[row];
    const b = right.rows[row];
    if (a == null || b == null || a.length !== b.length) {
      return false;
    }
    for (let column = 0; column < a.length; column += 1) {
      if (a[column] !== b[column]) {
        return false;
      }
    }
  }
  return true;
}

/** `scan_kit_open_plot` bytes: a little-endian `u32` JSON length, the JSON header, then marks. */
export function plotHeader(bytes: Uint8Array): PlotHeader {
  if (bytes.byteLength < 4) {
    throw new Error("plot payload is empty");
  }
  const length = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  if (bytes.byteLength < 4 + length) {
    throw new Error("plot payload is truncated");
  }
  const parsed = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length))) as {
    title?: unknown;
    controls?: unknown;
    table?: ViewTable | null;
    samples?: unknown;
    panels?: unknown;
    quality?: unknown;
  };
  const quality = text(parsed.quality);
  return {
    title: text(parsed.title),
    controls: Array.isArray(parsed.controls) ? parsed.controls.map(viewControl) : [],
    table: parsed.table ?? null,
    samples: Array.isArray(parsed.samples)
      ? parsed.samples.filter((sample): sample is number => typeof sample === "number")
      : [],
    panels: Array.isArray(parsed.panels) ? parsed.panels : [],
    quality: quality.length > 0 ? quality : "final",
  };
}

/** Canvas backing size in device pixels for a CSS box. */
export function backingSize(cssWidth: number, cssHeight: number, ratio: number): { width: number; height: number } {
  const scale = ratio > 0 ? ratio : 1;
  return {
    width: Math.max(16, Math.round(cssWidth * scale)),
    height: Math.max(16, Math.round(cssHeight * scale)),
  };
}
