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

/** Histogram bins follow the show-panel checkbox. X bins apply to every axis. */
export function controlDisabled(id: string, values: Readonly<Record<string, string>>): boolean {
  return (id === "hist_bins" || id === "shared") && values.hist != null && values.hist !== "On";
}
