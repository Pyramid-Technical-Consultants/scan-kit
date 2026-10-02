type ControlKind = "select" | "bare" | "check";

export type ControlSlot = {
  id: string;
  kind: ControlKind;
  /** Caption when the payload label is not the one the Python sidebar showed. */
  label?: string;
};

export type ControlSection = {
  title: string;
  slots: ControlSlot[];
  /** Checks share one row, in the same order as the plot. */
  inline?: boolean;
};

const SECTIONS: Record<string, ControlSection[]> = {
  binned_summary: [
    {
      title: "Data Source",
      slots: [
        { id: "source", kind: "select" },
        { id: "metric", kind: "select" },
        { id: "x", kind: "select" },
        { id: "bins", kind: "select", label: "Bins" },
      ],
    },
    {
      title: "Plot Style",
      slots: [
        { id: "glyph", kind: "select" },
        { id: "trend", kind: "select", label: "Trend" },
        { id: "cutoff", kind: "select", label: "Contour Cutoff" },
        { id: "interlock", kind: "select", label: "Interlock Thresholds" },
      ],
    },
    {
      title: "Histogram",
      slots: [
        { id: "hist", kind: "check", label: "Show Panel" },
        { id: "hist_bins", kind: "select", label: "Bins" },
        { id: "shared", kind: "check", label: "Share Bin Edges Across Rows" },
      ],
    },
    {
      title: "Correlation",
      slots: [{ id: "corr", kind: "check", label: "Show Panel" }],
    },
    {
      title: "Filter Data",
      slots: [
        { id: "domain", kind: "select" },
        { id: "beam", kind: "select" },
      ],
    },
  ],
  distribution: [
    {
      title: "Data Source",
      slots: [
        { id: "grain", kind: "select" },
        { id: "mode", kind: "select" },
      ],
    },
    {
      title: "Columns",
      inline: true,
      slots: [
        { id: "plan", kind: "check" },
        { id: "ic1", kind: "check" },
        { id: "ic2", kind: "check" },
      ],
    },
    {
      title: "Plot Style",
      slots: [
        { id: "draw", kind: "select" },
        { id: "ramp", kind: "select" },
        { id: "cutoff", kind: "select", label: "Contour Cutoff" },
      ],
    },
    {
      title: "Histogram",
      slots: [{ id: "hist_bins", kind: "select", label: "Bins" }],
    },
    { title: "Filter Data", slots: [{ id: "beam", kind: "select" }] },
  ],
  timeslice_replay: [{ title: "Data Source", slots: [{ id: "channel", kind: "bare" }] }],
  ic_fft_analysis: [{ title: "Data Source", slots: [{ id: "channel", kind: "bare" }] }],
  ic_audio_player: [{ title: "Data Source", slots: [{ id: "channel", kind: "bare" }] }],
  trajectory: [{ title: "Display", slots: [{ id: "azimuth", kind: "select" }] }],
  dose_accumulation: [{ title: "Options", slots: [{ id: "calibrate", kind: "select" }] }],
  dose_volume: [
    {
      title: "Data Source",
      slots: [
        { id: "grain", kind: "select" },
        { id: "xy", kind: "select", label: "Position" },
        { id: "quantity", kind: "select" },
        { id: "plan_sigma", kind: "select", label: "Plan Sigma" },
        { id: "sigma_ref", kind: "select", label: "Reference" },
      ],
    },
    {
      title: "Model",
      slots: [
        { id: "model", kind: "select" },
        { id: "scatter", kind: "select" },
        { id: "histories", kind: "select" },
        { id: "spread", kind: "select", label: "Energy Spread" },
      ],
    },
    {
      title: "Phantom",
      slots: [
        { id: "medium", kind: "select" },
        { id: "phantom", kind: "select", label: "Thickness" },
        { id: "wet", kind: "select", label: "Entrance" },
      ],
    },
    {
      title: "Compare",
      slots: [
        { id: "compare", kind: "select" },
        { id: "edge", kind: "select", label: "Field Edge" },
      ],
    },
    { title: "Color", slots: [{ id: "scale", kind: "select" }] },
    { title: "Patient", slots: [{ id: "fraction", kind: "select" }] },
  ],
};

/** Sidebar groups for one view. Unknown controls land in Options. */
export function controlSections(viewId: string, ids: readonly string[]): ControlSection[] {
  const specs = SECTIONS[viewId];
  if (specs == null) {
    const slots = ids.map((id) => ({ id, kind: "select" as const }));
    return slots.length === 0 ? [] : [{ title: "Options", slots }];
  }
  const present = new Set(ids);
  const used = new Set<string>();
  const sections: ControlSection[] = [];
  for (const spec of specs) {
    const slots = spec.slots.filter((slot) => present.has(slot.id));
    for (const slot of slots) {
      used.add(slot.id);
    }
    if (slots.length > 0) {
      sections.push({ ...spec, slots });
    }
  }
  const extra = ids.filter((id) => !used.has(id));
  if (extra.length > 0) {
    sections.push({
      title: "Options",
      slots: extra.map((id) => ({ id, kind: "select" })),
    });
  }
  return sections;
}

export { segmentChoices } from "@/CatalogField";

/** Histogram bins follow the show-panel checkbox. X bins apply to every axis. */
export function controlDisabled(id: string, values: Readonly<Record<string, string>>): boolean {
  return (id === "hist_bins" || id === "shared") && values.hist != null && values.hist !== "On";
}
