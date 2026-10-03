import { segmentChoices } from "@/CatalogField";

export type ControlSlot = {
  id: string;
  kind: "select" | "check";
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
    const title = control.group != null && control.group.length > 0 ? control.group : "Options";
    const slot: ControlSlot = {
      id: control.id,
      kind: control.kind === "check" ? "check" : "select",
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

/** Histogram bins follow the show-panel checkbox. X bins apply to every axis. */
export function controlDisabled(id: string, values: Readonly<Record<string, string>>): boolean {
  return (
    (id === "hist_bins" || id === "share" || id === "shared") &&
    values.hist != null &&
    values.hist !== "On"
  );
}
