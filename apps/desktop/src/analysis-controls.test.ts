import { expect, it } from "vitest";

import { controlDisabled, controlSections } from "./analysis-controls";

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

it("groups binned summary like the Python sidebar", () => {
  const sections = controlSections("binned_summary", BINNED_IDS);
  expect(sections.map((section) => section.title)).toEqual([
    "Y Metric",
    "X Parameter",
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
  expect(sections[2]?.slots.find((slot) => slot.id === "trend")?.label).toBe("Trend Line");
});

it("drops controls the view did not send and parks unknown ones in Options", () => {
  const sections = controlSections("binned_summary", ["metric", "x", "glyph", "azimuth"]);
  expect(sections.map((section) => [section.title, section.slots.map((slot) => slot.id)])).toEqual([
    ["Y Metric", ["metric"]],
    ["X Parameter", ["x"]],
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
