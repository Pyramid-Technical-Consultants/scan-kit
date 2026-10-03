import { ChartColumn, Columns2, LayoutGrid } from "lucide-react";
import { expect, it } from "vitest";

import {
  applyOption,
  controlDisabled,
  controlSections,
  segmentChoices,
  type GrainMemory,
} from "./analysis-controls";
import { optionIcon } from "./option-icons";

const BINNED = [
  { id: "source", group: "Data Source" },
  { id: "y", group: "Data Source" },
  { id: "x", group: "Data Source" },
  { id: "bins", group: "Data Source" },
  { id: "glyph", group: "Plot Style" },
  { id: "trend", group: "Plot Style" },
  { id: "interlock", group: "Plot Style" },
  { id: "cutoff", group: "Plot Style" },
  { id: "hist", group: "Histogram", kind: "check" },
  { id: "hist_bins", group: "Histogram" },
  { id: "share", group: "Histogram" },
  { id: "corr", group: "Correlation", kind: "check" },
  { id: "domain", group: "Filter Data" },
  { id: "beam", group: "Filter Data" },
];

it("groups binned summary controls in plot order", () => {
  const sections = controlSections(BINNED);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Plot Style",
    "Histogram",
    "Correlation",
    "Filter Data",
  ]);
  expect(sections.flatMap((section) => section.slots.map((slot) => slot.id))).toEqual(
    BINNED.map((control) => control.id),
  );
  expect(sections[2]?.slots.find((slot) => slot.id === "hist")?.kind).toBe("check");
  expect(sections[1]?.slots.find((slot) => slot.id === "interlock")?.kind).toBe("select");
});

it("keeps a joined button row for two or three short names", () => {
  expect(segmentChoices(["Off", "Linear", "Polynomial"])).toBe(true);
  expect(segmentChoices(["Off", "On"])).toBe(true);
  expect(segmentChoices(["Beam on", "Beam off", "Both"])).toBe(true);
  expect(segmentChoices(["on", "off"])).toBe(true);
  expect(segmentChoices(["spot", "timeslice"])).toBe(true);
  expect(segmentChoices(["Energy", "Target MU", "Spot time", "Radius"])).toBe(false);
  expect(segmentChoices(["All", "Lower 95%", "Upper 95%", "MAD outliers"])).toBe(false);
  expect(segmentChoices(["Violin", "Box", "Mean", "Scatter", "Contour"])).toBe(false);
  expect(segmentChoices(["Spot — Isocenter", "Spot — Chamber"])).toBe(false);
  expect(segmentChoices(["Energy"])).toBe(false);
  expect(segmentChoices(["Auto", "8", "16", "32", "64"])).toBe(true);
  expect(segmentChoices(["Own", "Plot", "Page"])).toBe(true);
  expect(optionIcon("Own")).toBe(ChartColumn);
  expect(optionIcon("Plot")).toBe(Columns2);
  expect(optionIcon("Page")).toBe(LayoutGrid);
  expect(
    segmentChoices([
      { label: "Spot", detail: "One row per spot" },
      { label: "Timeslice", detail: "One row per millisecond" },
    ]),
  ).toBe(true);
});

it("parks a control with no group in Options", () => {
  const sections = controlSections([
    { id: "y", group: "Data Source" },
    { id: "x", group: "Data Source" },
    { id: "glyph", group: "Plot Style" },
    { id: "azimuth" },
  ]);
  expect(sections.map((section) => [section.title, section.slots.map((slot) => slot.id)])).toEqual([
    ["Data Source", ["y", "x"]],
    ["Plot Style", ["glyph"]],
    ["Options", ["azimuth"]],
  ]);
});

it("keeps distribution checks in Data Source after the axes", () => {
  const sections = controlSections([
    { id: "source", group: "Data Source" },
    { id: "xy", group: "Data Source" },
    { id: "plan", group: "Data Source", kind: "check" },
    { id: "ic1", group: "Data Source", kind: "check" },
    { id: "ic2", group: "Data Source", kind: "check" },
    { id: "draw", group: "Plot Style" },
    { id: "ramp", group: "Plot Style" },
    { id: "cutoff", group: "Plot Style" },
    { id: "hist_bins", group: "Histogram" },
    { id: "beam", group: "Filter Data" },
  ]);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Plot Style",
    "Histogram",
    "Filter Data",
  ]);
  expect(sections[0]?.slots.map((slot) => slot.id)).toEqual(["source", "xy", "plan", "ic1", "ic2"]);
  expect(sections[0]?.slots.filter((slot) => slot.kind === "check").map((slot) => slot.id)).toEqual([
    "plan",
    "ic1",
    "ic2",
  ]);
  expect(sections[1]?.slots.map((slot) => slot.id)).toEqual(["draw", "ramp", "cutoff"]);
  expect(sections[2]?.slots.map((slot) => slot.id)).toEqual(["hist_bins"]);
  expect(controlSections([{ id: "y", group: "Data Source" }])[0]?.title).toBe("Data Source");
});

it("uses one Options group when the control names no fieldset", () => {
  expect(controlSections([{ id: "calibrate", kind: "select" }])).toEqual([
    { title: "Options", slots: [{ id: "calibrate", kind: "select" }] },
  ]);
});

it("groups dose volume into source, model, phantom, compare, and color", () => {
  const sections = controlSections([
    { id: "source", group: "Data Source" },
    { id: "xy", group: "Data Source" },
    { id: "quantity", group: "Data Source" },
    { id: "plan_sigma", group: "Data Source" },
    { id: "model", group: "Model" },
    { id: "scatter", group: "Model" },
    { id: "spread", group: "Model" },
    { id: "medium", group: "Phantom" },
    { id: "phantom", group: "Phantom" },
    { id: "wet", group: "Phantom" },
    { id: "compare", group: "Compare" },
    { id: "edge", group: "Compare" },
    { id: "scale", group: "Color" },
  ]);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Model",
    "Phantom",
    "Compare",
    "Color",
  ]);
  expect(sections[0]?.slots.map((slot) => slot.id)).toEqual([
    "source",
    "xy",
    "quantity",
    "plan_sigma",
  ]);
  expect(sections[4]?.slots.map((slot) => slot.id)).toEqual(["scale"]);
  const withCt = controlSections([{ id: "fraction", group: "Patient" }]);
  expect(withCt.map((section) => section.title)).toEqual(["Patient"]);
});

it("remembers the axes picked on spot and on timeslice", () => {
  const memory: GrainMemory = {};
  const spot = { y: "Dose Error (%)", x: "Radius (mm)", glyph: "Violin", ic1: "On" };
  let options = applyOption(spot, { ...spot, source: "Spot" }, memory, "source", "Timeslice");
  expect(options).toMatchObject({
    source: "Timeslice",
    y: "Dose Error (%)",
    x: "Radius (mm)",
    glyph: "Violin",
  });

  options = applyOption(
    options,
    { source: "Timeslice", y: "Current Ratios (%)", x: "Energy (MeV)", glyph: "Violin" },
    memory,
    "source",
    "Spot",
  );
  expect(options.y).toBe("Dose Error (%)");
  expect(options.x).toBe("Radius (mm)");

  options = applyOption(
    options,
    { source: "Spot", y: "Dose Error (%)", x: "Radius (mm)" },
    memory,
    "source",
    "Timeslice",
  );
  expect(options.y).toBe("Current Ratios (%)");
  expect(options.x).toBe("Energy (MeV)");

  const distribution: GrainMemory = {};
  options = applyOption(
    { xy: "Position (mm)", ic1: "On" },
    { source: "Spot", xy: "Position (mm)", ic1: "On" },
    distribution,
    "source",
    "Timeslice",
  );
  options = applyOption(
    options,
    { source: "Timeslice", xy: "Position (mm)", ic1: "On" },
    distribution,
    "xy",
    "Amplifier (V)",
  );
  options = applyOption(
    options,
    { source: "Timeslice", xy: "Amplifier (V)", ic1: "On" },
    distribution,
    "source",
    "Spot",
  );
  expect(options.xy).toBe("Position (mm)");
  expect(options.ic1).toBe("On");
  options = applyOption(
    options,
    { source: "Spot", xy: "Position (mm)", ic1: "On" },
    distribution,
    "source",
    "Timeslice",
  );
  expect(options.xy).toBe("Amplifier (V)");
});

it("disables histogram bin controls until the panel is on", () => {
  expect(controlDisabled("hist_bins", { hist: "Off" })).toBe(true);
  expect(controlDisabled("hist_bins", {})).toBe(false);
  expect(controlDisabled("share", { hist: "On" })).toBe(false);
  expect(controlDisabled("share", { hist: "Off" })).toBe(true);
  expect(controlDisabled("domain", { hist: "Off" })).toBe(false);
});

it("leaves X bins available for every axis and glyph", () => {
  expect(controlDisabled("bins", { x: "Energy", glyph: "Violin" })).toBe(false);
  expect(controlDisabled("bins", { x: "Target MU", glyph: "Scatter" })).toBe(false);
  expect(controlDisabled("bins", { x: "Radius", glyph: "Contour" })).toBe(false);
});
