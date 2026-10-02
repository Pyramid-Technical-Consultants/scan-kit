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
  };
  return {
    title: text(parsed.title),
    controls: Array.isArray(parsed.controls) ? parsed.controls.map(viewControl) : [],
    table: parsed.table ?? null,
    samples: Array.isArray(parsed.samples)
      ? parsed.samples.filter((sample): sample is number => typeof sample === "number")
      : [],
    panels: Array.isArray(parsed.panels) ? parsed.panels : [],
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
