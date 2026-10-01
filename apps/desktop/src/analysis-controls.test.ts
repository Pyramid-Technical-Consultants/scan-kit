import { expect, it } from "vitest";

import { controlDisabled, controlSections, segmentChoices } from "./analysis-controls";

const BINNED_IDS = [
  "metric",
  "source",
  "x",
  "glyph",
  "beam",
  "domain",
  "trend",
  "hist",
  "corr",
  "fliers",
  "interlock",
  "bins",
  "hist_bins",
  "shared",
  "cutoff",
];

it("groups binned summary controls", () => {
  const sections = controlSections("binned_summary", BINNED_IDS);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Plot Style",
    "Histogram",
    "Correlation",
    "Filter Data",
  ]);
  expect(sections.flatMap((section) => section.slots.map((slot) => slot.id))).toEqual([
    "source",
    "metric",
    "x",
    "bins",
    "glyph",
    "trend",
    "cutoff",
    "fliers",
    "interlock",
    "hist",
    "hist_bins",
    "shared",
    "corr",
    "domain",
    "beam",
  ]);
  expect(sections[1]?.slots.find((slot) => slot.id === "trend")?.label).toBe("Trend Line");
});

it("keeps a joined button row for two or three short names", () => {
  expect(segmentChoices(["Beam on", "Beam off", "Both"])).toBe(true);
  expect(segmentChoices(["on", "off"])).toBe(true);
  expect(segmentChoices(["spot", "timeslice"])).toBe(true);
  expect(segmentChoices(["Energy", "Target MU", "Spot time", "Radius"])).toBe(false);
  expect(segmentChoices(["All", "Lower 95%", "Upper 95%", "MAD outliers"])).toBe(false);
  expect(segmentChoices(["Violin", "Box", "Mean", "Scatter", "Contour"])).toBe(false);
  expect(segmentChoices(["Spot — Isocenter", "Spot — Chamber"])).toBe(false);
  expect(segmentChoices(["Energy"])).toBe(false);
});

it("drops controls the view did not send and parks unknown ones in Options", () => {
  const sections = controlSections("binned_summary", ["metric", "x", "glyph", "azimuth"]);
  expect(sections.map((section) => [section.title, section.slots.map((slot) => slot.id)])).toEqual([
    ["Data Source", ["metric", "x"]],
    ["Plot Style", ["glyph"]],
    ["Options", ["azimuth"]],
  ]);
});

it("uses one Options group when the view has no sidebar map", () => {
  expect(controlSections("dose_volume", ["calibrate"])).toEqual([
    { title: "Options", slots: [{ id: "calibrate", kind: "select" }] },
  ]);
});

it("disables histogram bin controls until the panel is on", () => {
  expect(controlDisabled("hist_bins", { hist: "Off" })).toBe(true);
  expect(controlDisabled("shared", { hist: "On" })).toBe(false);
  expect(controlDisabled("domain", { hist: "Off" })).toBe(false);
});
