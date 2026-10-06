import { segmentChoices } from "@/CatalogField";

export type ControlSlot = {
  id: string;
  kind: "select" | "check" | "segments";
};

export type SegmentItem = {
  kind: string;
  state?: string;
  which?: string;
  column?: string;
  lo?: number;
  hi?: number;
  op?: string;
  threshold?: number;
};

export type ControlSection = {
  title: string;
  slots: ControlSlot[];
};

type SectionControl = {
  id: string;
  group?: string;
  kind?: string;
};

/** One fieldset per group, in the order the plot sent the controls. */
export function controlSections(controls: readonly SectionControl[]): ControlSection[] {
  const sections: ControlSection[] = [];
  for (const control of controls) {
    if (control.kind === "scrub") {
      continue;
    }
    const title = control.group != null && control.group.length > 0 ? control.group : "Options";
    const slot: ControlSlot = {
      id: control.id,
      kind:
        control.kind === "check" ? "check" : control.kind === "segments" ? "segments" : "select",
    };
    const last = sections[sections.length - 1];
    if (last != null && last.title === title) {
      last.slots.push(slot);
    } else {
      sections.push({ title, slots: [slot] });
    }
  }
  return sections;
}

export { segmentChoices };

const GRAIN_FIELDS = ["y", "x", "xy"] as const;

/**
 * Timeline playback filters the scatter in the plot. A settled playhead still
 * has to rebuild the picture when the spectrum, confidence, or coverage depends
 * on that window.
 */
export function playheadReplay(options: Readonly<Record<string, string>>): boolean {
  if (options.fft === "On") {
    return true;
  }
  if (options.scatter !== "On") {
    return false;
  }
  return options.scatter_xy === "Confidence" || options.scatter_xy === "Coverage (%)";
}

export type GrainMemory = {
  spot?: Record<string, string>;
  timeslice?: Record<string, string>;
};

function grainName(source: string): "spot" | "timeslice" | "" {
  const lower = source.trim().toLowerCase();
  if (lower.includes("time")) {
    return "timeslice";
  }
  if (lower.includes("spot")) {
    return "spot";
  }
  return "";
}

/**
 * Spot and Timeslice keep their own Y, X, and XY choices. The first visit to a
 * grain keeps the current choice so a shared quantity can carry over.
 */
export function applyOption(
  options: Record<string, string>,
  resolved: Readonly<Record<string, string>>,
  memory: GrainMemory,
  id: string,
  value: string,
): Record<string, string> {
  if (id !== "source") {
    return { ...options, [id]: value };
  }
  const from = grainName(resolved.source ?? options.source ?? "");
  if (from !== "") {
    const snapshot: Record<string, string> = {};
    for (const key of GRAIN_FIELDS) {
      const shown = resolved[key];
      if (shown != null && shown.length > 0) {
        snapshot[key] = shown;
      }
    }
    memory[from] = snapshot;
  }
  const next: Record<string, string> = { ...options, source: value };
  const to = grainName(value);
  const saved = to === "" ? undefined : memory[to];
  if (saved != null) {
    for (const key of GRAIN_FIELDS) {
      const remembered = saved[key];
      if (remembered != null) {
        next[key] = remembered;
      }
    }
  }
  return next;
}

const FRESH_SEGMENT: Record<string, SegmentItem> = {
  beam: { kind: "beam", state: "both" },
  rank: { kind: "rank", which: "all" },
  range: { kind: "range", column: "energy", lo: 0, hi: 0 },
  compare: { kind: "compare", column: "", op: "abs_gt", threshold: 1 },
};

export function parseSegments(value: string): SegmentItem[] | null {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      return null;
    }
    const items: SegmentItem[] = [];
    for (const item of parsed) {
      if (item == null || typeof item !== "object" || !("kind" in item)) {
        return null;
      }
      const row = item as { kind?: unknown };
      if (typeof row.kind !== "string") {
        return null;
      }
      items.push(item as SegmentItem);
    }
    return items;
  } catch {
    return null;
  }
}

export function segmentsText(items: readonly SegmentItem[]): string {
  return JSON.stringify(items);
}

export function addSegment(items: readonly SegmentItem[], kind: string): SegmentItem[] {
  if (items.some((item) => item.kind === kind)) {
    return [...items];
  }
  return [...items, FRESH_SEGMENT[kind] ?? { kind }];
}

export function removeSegment(items: readonly SegmentItem[], index: number): SegmentItem[] {
  return items.filter((_, item) => item !== index);
}

export function replaceSegment(
  items: readonly SegmentItem[],
  index: number,
  next: SegmentItem,
): SegmentItem[] {
  return items.map((item, itemIndex) => (itemIndex === index ? next : item));
}

/** Histogram bins follow the show-panel checkbox. X bins apply to every axis. */
export function controlDisabled(id: string, values: Readonly<Record<string, string>>): boolean {
  return (
    (id === "hist_bins" || id === "share" || id === "shared") &&
    values.hist != null &&
    values.hist !== "On"
  );
}
