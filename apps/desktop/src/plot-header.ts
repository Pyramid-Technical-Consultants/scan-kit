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
  panels: unknown[];
  quality: string;
  /** Polyline identity. Empty when this payload has no trace lines to reuse. */
  lineToken: string;
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
  if (current.title !== next.title) {
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

type LevelPaint = {
  label: string;
  min: string;
  max: string;
  step: string;
  value: string;
};

function levelPaint(raw: string): LevelPaint | null {
  try {
    const spec = JSON.parse(raw) as {
      label?: unknown;
      min?: unknown;
      max?: unknown;
      step?: unknown;
      value?: unknown;
    };
    if (
      typeof spec.label !== "string" ||
      typeof spec.min !== "string" ||
      typeof spec.max !== "string" ||
      typeof spec.step !== "string" ||
      typeof spec.value !== "string"
    ) {
      return null;
    }
    return {
      label: spec.label,
      min: spec.min,
      max: spec.max,
      step: spec.step,
      value: spec.value,
    };
  } catch {
    return null;
  }
}

/** Point the level slider at the window the picture is using. */
export function applyLevelPaint(header: PlotHeader | null, raw: string): PlotHeader | null {
  if (header == null || raw.length === 0) {
    return header;
  }
  const spec = levelPaint(raw);
  if (spec == null) {
    return header;
  }
  const index = header.controls.findIndex((control) => control.id === "level");
  if (index < 0) {
    return header;
  }
  const control = header.controls[index];
  if (control == null) {
    return header;
  }
  const bound = (id: string) => control.options.find((option) => option.id === id)?.label;
  if (
    control.label === spec.label &&
    control.value === spec.value &&
    bound("min") === spec.min &&
    bound("max") === spec.max &&
    bound("step") === spec.step
  ) {
    return header;
  }
  let options = control.options.map((option) => {
    if (option.id === "min") {
      return { ...option, label: spec.min };
    }
    if (option.id === "max") {
      return { ...option, label: spec.max };
    }
    if (option.id === "step") {
      return { ...option, label: spec.step };
    }
    return option;
  });
  for (const [id, label] of [
    ["min", spec.min],
    ["max", spec.max],
    ["step", spec.step],
  ] as const) {
    if (!options.some((option) => option.id === id)) {
      options = [...options, { id, label, detail: "", icon: "" }];
    }
  }
  const controls = header.controls.slice();
  controls[index] = { ...control, label: spec.label, value: spec.value, options };
  return { ...header, controls };
}

/**
 * Header to leave on screen. A payload with no panels keeps the picture already
 * up. A blank header does not swallow the first frame that actually has panels.
 */
export function shownHeader(current: PlotHeader | null, incoming: PlotHeader): PlotHeader | null {
  if (incoming.panels.length === 0) {
    return current;
  }
  if (current == null || current.panels.length === 0) {
    return incoming;
  }
  return sameChrome(current, incoming) ? current : incoming;
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
    panels?: unknown;
    quality?: unknown;
    line_token?: unknown;
  };
  const quality = text(parsed.quality);
  return {
    title: text(parsed.title),
    controls: Array.isArray(parsed.controls) ? parsed.controls.map(viewControl) : [],
    table: parsed.table ?? null,
    panels: Array.isArray(parsed.panels) ? parsed.panels : [],
    quality: quality.length > 0 ? quality : "final",
    lineToken: text(parsed.line_token),
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
