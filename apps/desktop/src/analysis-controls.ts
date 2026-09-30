export type ControlKind = "select" | "bare" | "check";

export type ControlSlot = {
  id: string;
  kind: ControlKind;
  /** Caption when the payload label is not the one the Python sidebar showed. */
  label?: string;
};

export type ControlSection = {
  title: string;
  slots: ControlSlot[];
};

const SECTIONS: Record<string, ControlSection[]> = {
  binned_summary: [
    {
      title: "Y Metric",
      slots: [
        { id: "source", kind: "select" },
        { id: "metric", kind: "bare" },
      ],
    },
    {
      title: "X Parameter",
      slots: [
        { id: "x", kind: "bare" },
        { id: "bins", kind: "select", label: "Bins" },
      ],
    },
    {
      title: "Plot Style",
      slots: [
        { id: "glyph", kind: "bare" },
        { id: "trend", kind: "check", label: "Trend Line" },
        { id: "cutoff", kind: "select" },
        { id: "fliers", kind: "check", label: "Show Box Outliers" },
        { id: "interlock", kind: "check", label: "Interlock thresholds" },
      ],
    },
    {
      title: "Histogram",
      slots: [
        { id: "hist", kind: "check", label: "Show panel" },
        { id: "hist_bins", kind: "select", label: "Bins" },
        { id: "shared", kind: "check", label: "Share bin edges across rows" },
      ],
    },
    {
      title: "Correlation",
      slots: [{ id: "corr", kind: "check", label: "Show panel" }],
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
      title: "Distribution",
      slots: [
        { id: "mode", kind: "bare" },
        { id: "grain", kind: "select" },
      ],
    },
    { title: "Plot Style", slots: [{ id: "draw", kind: "bare" }] },
    { title: "Filter Data", slots: [{ id: "beam", kind: "select" }] },
  ],
  timeslice_replay: [
    { title: "Signal Source", slots: [{ id: "channel", kind: "bare" }] },
    { title: "Options", slots: [{ id: "scrub", kind: "select" }] },
  ],
  ic_fft_analysis: [{ title: "Signal Source", slots: [{ id: "channel", kind: "bare" }] }],
  ic_audio_player: [{ title: "Signal Source", slots: [{ id: "channel", kind: "bare" }] }],
  trajectory: [{ title: "Display", slots: [{ id: "azimuth", kind: "select" }] }],
  dose_accumulation: [{ title: "Options", slots: [{ id: "calibrate", kind: "select" }] }],
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
      sections.push({ title: spec.title, slots });
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

/** Histogram bin controls follow the show-panel checkbox, same as the Python panel. */
export function controlDisabled(id: string, values: Readonly<Record<string, string>>): boolean {
  return (id === "hist_bins" || id === "shared") && values.hist !== "On";
}
