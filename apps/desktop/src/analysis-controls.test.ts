import { ChartColumn, Columns2, LayoutGrid, Box, RectangleHorizontal, RectangleVertical, Square } from "lucide-react";
import { expect, it } from "vitest";

import {
  addSegment,
  applyOption,
  controlDisabled,
  controlSections,
  parseSegments,
  playheadReplay,
  removeSegment,
  segmentChoices,
  segmentsText,
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
  { id: "segments", group: "Filter Data", kind: "segments" },
];

it("rebuilds a settled playhead for the spectrum and the reduced distributions", () => {
  expect(playheadReplay({})).toBe(false);
  expect(playheadReplay({ scatter: "On" })).toBe(false);
  expect(playheadReplay({ scatter: "On", scatter_xy: "Probe (G)" })).toBe(false);
  expect(playheadReplay({ scatter: "On", scatter_xy: "Probe (G)", draw: "Scatter" })).toBe(false);
  expect(playheadReplay({ fft: "On" })).toBe(true);
  expect(playheadReplay({ scatter: "On", scatter_xy: "Confidence" })).toBe(true);
  expect(playheadReplay({ scatter: "On", scatter_xy: "Coverage (%)" })).toBe(true);
  expect(playheadReplay({ scatter: "On", draw: "Contour" })).toBe(false);
  expect(playheadReplay({ scatter: "On", draw: "Density" })).toBe(false);
  expect(playheadReplay({ draw: "Density" })).toBe(false);
});

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
  expect(sections[sections.length - 1]?.slots[0]?.kind).toBe("segments");
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
  for (const scale of [
    "Turbo",
    "Viridis",
    "Magma",
    "Inferno",
    "Plasma",
    "Cividis",
    "Deep",
    "Cubehelix",
    "Heat",
    "Gray",
    "Managua",
    "Berlin",
    "Coolwarm",
    "RdYlBu",
    "Spectral",
    "PuOr",
  ]) {
    expect(optionIcon(scale), scale).toBeTruthy();
  }
  expect(optionIcon("Own")).toBe(ChartColumn);
  expect(optionIcon("Plot")).toBe(Columns2);
  expect(optionIcon("Page")).toBe(LayoutGrid);
  expect(optionIcon("axial")).toBe(Square);
  expect(optionIcon("coronal")).toBe(RectangleHorizontal);
  expect(optionIcon("sagittal")).toBe(RectangleVertical);
  expect(optionIcon("volume")).toBe(Box);
  expect(optionIcon("3D")).toBe(Box);
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
    { id: "segments", group: "Filter Data", kind: "segments" },
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

it("keeps a goals note as a text field", () => {
  expect(controlSections([{ id: "goals", kind: "text", group: "Goals" }])).toEqual([
    { title: "Goals", slots: [{ id: "goals", kind: "text" }] },
  ]);
});

it("uses one Options group when the control names no fieldset", () => {
  expect(controlSections([{ id: "calibrate", kind: "select" }])).toEqual([
    { title: "Options", slots: [{ id: "calibrate", kind: "select" }] },
  ]);
});

it("groups dose volume by the order a user sets it up", () => {
  const sections = controlSections([
    { id: "source", group: "Dose" },
    { id: "xy", group: "Dose" },
    { id: "quantity", group: "Dose" },
    { id: "compare", group: "Dose" },
    { id: "model", group: "Calculation", kind: "radio" },
    { id: "histories", group: "Calculation", kind: "radio" },
    { id: "spread", group: "Calculation" },
    { id: "medium", group: "Phantom" },
    { id: "phantom", group: "Phantom" },
    { id: "wet", group: "Phantom" },
    { id: "edge", group: "Field" },
    { id: "field", group: "Field", kind: "check" },
    { id: "ray", group: "Picture" },
    { id: "scale", group: "Picture" },
    { id: "level", group: "Picture", kind: "range" },
  ]);
  expect(sections.map((section) => section.title)).toEqual([
    "Dose",
    "Calculation",
    "Phantom",
    "Field",
    "Picture",
  ]);
  expect(sections[0]?.slots.map((slot) => slot.id)).toEqual(["source", "xy", "quantity", "compare"]);
  expect(sections[1]?.slots.map((slot) => slot.id)).toEqual(["model", "histories", "spread"]);
  expect(sections[1]?.slots.filter((slot) => slot.kind === "radio").map((slot) => slot.id)).toEqual([
    "model",
    "histories",
  ]);
  expect(sections[4]?.slots.map((slot) => slot.id)).toEqual(["ray", "scale", "level"]);
  expect(sections[4]?.slots.find((slot) => slot.id === "level")?.kind).toBe("range");
  const withCt = controlSections([{ id: "fraction", group: "Study" }]);
  expect(withCt.map((section) => section.title)).toEqual(["Study"]);
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

it("adds a missing segment and drops one by index", () => {
  expect(addSegment([], "beam")).toEqual([{ kind: "beam", state: "both" }]);
  const beam = [{ kind: "beam", state: "on" }];
  const both = addSegment(beam, "rank");
  expect(both).toEqual([
    { kind: "beam", state: "on" },
    { kind: "rank", which: "all" },
  ]);
  expect(addSegment(both, "rank")).toEqual(both);
  const text = segmentsText(removeSegment(both, 0));
  expect(parseSegments(text)).toEqual([{ kind: "rank", which: "all" }]);
  expect(parseSegments("nope")).toBeNull();
});

it("leaves X bins available for every axis and glyph", () => {
  expect(controlDisabled("bins", { x: "Energy", glyph: "Violin" })).toBe(false);
  expect(controlDisabled("bins", { x: "Target MU", glyph: "Scatter" })).toBe(false);
  expect(controlDisabled("bins", { x: "Radius", glyph: "Contour" })).toBe(false);
});
