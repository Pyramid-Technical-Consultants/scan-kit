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
    "interlock",
    "hist",
    "hist_bins",
    "shared",
    "corr",
    "domain",
    "beam",
  ]);
  expect(sections[1]?.slots.find((slot) => slot.id === "trend")?.label).toBe("Trend");
  expect(sections[1]?.slots.find((slot) => slot.id === "interlock")?.label).toBe(
    "Interlock Thresholds",
  );
  expect(sections[1]?.slots.some((slot) => slot.id === "fliers")).toBe(false);
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
});

it("drops controls the view did not send and parks unknown ones in Options", () => {
  const sections = controlSections("binned_summary", ["metric", "x", "glyph", "azimuth"]);
  expect(sections.map((section) => [section.title, section.slots.map((slot) => slot.id)])).toEqual([
    ["Data Source", ["metric", "x"]],
    ["Plot Style", ["glyph"]],
    ["Options", ["azimuth"]],
  ]);
});

it("puts the spot source ahead of the distribution signal", () => {
  const sections = controlSections("distribution", [
    "mode",
    "grain",
    "draw",
    "beam",
    "ramp",
    "cutoff",
    "ic1",
    "ic2",
    "plan",
    "hist_bins",
  ]);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Columns",
    "Plot Style",
    "Histogram",
    "Filter Data",
  ]);
  expect(sections[0]?.slots.map((slot) => slot.id)).toEqual(["grain", "mode"]);
  expect(sections[1]?.slots.map((slot) => slot.id)).toEqual(["ic1", "ic2", "plan"]);
  expect(sections[1]?.slots.every((slot) => slot.kind === "check")).toBe(true);
  expect(sections[2]?.slots.map((slot) => slot.id)).toEqual(["draw", "ramp", "cutoff"]);
  expect(sections[3]?.slots.map((slot) => slot.id)).toEqual(["hist_bins"]);
  expect(controlSections("timeslice_replay", ["channel"])[0]?.title).toBe("Data Source");
  expect(controlSections("ic_fft_analysis", ["channel"])[0]?.title).toBe("Data Source");
  expect(controlSections("ic_audio_player", ["channel"])[0]?.title).toBe("Data Source");
});

it("uses one Options group when the view has no sidebar map", () => {
  expect(controlSections("session_log_compare", ["calibrate"])).toEqual([
    { title: "Options", slots: [{ id: "calibrate", kind: "select" }] },
  ]);
});

it("groups dose volume into source, model, phantom, compare, and color", () => {
  const sections = controlSections("dose_volume", [
    "grain",
    "xy",
    "quantity",
    "model",
    "scatter",
    "spread",
    "medium",
    "phantom",
    "wet",
    "compare",
    "edge",
    "scale",
  ]);
  expect(sections.map((section) => section.title)).toEqual([
    "Data Source",
    "Model",
    "Phantom",
    "Compare",
    "Color",
  ]);
  expect(sections[0]?.slots.map((slot) => slot.id)).toEqual(["grain", "xy", "quantity"]);
  expect(sections[4]?.slots.map((slot) => slot.id)).toEqual(["scale"]);
  const withCt = controlSections("dose_volume", ["fraction"]);
  expect(withCt.map((section) => section.title)).toEqual(["Patient"]);
});

it("disables histogram bin controls until the panel is on", () => {
  expect(controlDisabled("hist_bins", { hist: "Off" })).toBe(true);
  expect(controlDisabled("hist_bins", {})).toBe(false);
  expect(controlDisabled("shared", { hist: "On" })).toBe(false);
  expect(controlDisabled("domain", { hist: "Off" })).toBe(false);
});

it("leaves X bins available for every axis and glyph", () => {
  expect(controlDisabled("bins", { x: "Energy", glyph: "Violin" })).toBe(false);
  expect(controlDisabled("bins", { x: "Target MU", glyph: "Scatter" })).toBe(false);
  expect(controlDisabled("bins", { x: "Radius", glyph: "Contour" })).toBe(false);
});
