# Software requirements

| | |
|---|---|
| Document | SK-SRS |
| Standard | IEC 62304:2006+AMD1:2015, clause 5.2 |
| Safety class | A |
| Product | Scan Kit 2 |

Git history is the revision record. New behavior adds the next `SK-REQ-###` id. Ids are not reused.

Phase 1 requirements are the seams of the replacement. They do not restate the Python application’s behavior. Ported behavior is specified here when it is ported.

## SK-REQ-001

The software reports a single version string.

The Rust workspace version is that string. The desktop shell and the `scan_kit_version` tool both return it. The Python package version is a different product identity and is not this string.

## SK-REQ-002

An MCP client can list the tool catalog and call `scan_kit_version` and `scan_kit_health`.

`scan_kit_version` is a granular tool and returns the version string from SK-REQ-001. `scan_kit_health` is a workflow tool. It calls the version function and reports that the tool catalog is non-empty. Both tools run inside `scan-kit-core`. The MCP server does not implement them again.

## SK-REQ-003

The About dialog text includes the workspace version from SK-REQ-001.

`scan_kit_about` is a granular tool. It returns that version and the same description, project link, maintainer, support address, and copyright as the Python About dialog. The desktop Help menu shows this text. The MCP server does not write it again.

## SK-REQ-004

Opening a data folder discovers local sessions and reuses cached metadata when the file fingerprint matches.

A session is an unpacked folder that contains `input_map.csv`, or a `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, or `.tar.xz` archive. An unpacked folder wins over an archive with the same id. Metadata comes from `termination_summary.txt`, including the date, primary dose, treatment time, room, configuration name, the larger spot extent, and the planned layer count. The summary is read from an archive without unpacking it. The sqlite row stores size and mtime. A later open keeps the cached metadata when both still match. Map extent and layer count are filled from spot positions only when the cached row does not already have them.

`scan_kit_open_library` is the workflow tool for this. It discovers, syncs the index, and returns the rows.

## SK-REQ-005

Session notes and the selected session ids are stored in the existing sqlite file.

The file is `~/.scan-kit/scan-kit.sqlite` with `user_version` 3 and the `prefs`, `libraries`, and `sessions` tables. A file written at an older `user_version` is migrated forward. `scan_kit_set_note` and `scan_kit_select_sessions` are granular tools. At most five sessions are selected. The last data folder and the window geometry are prefs in that same file.

## SK-REQ-006

Loading named columns resolves aliases and applies the G2 current scale.

`scan_kit_load_columns` returns one typed column per requested field. Alias resolution uses the concept table (a G2 `r_ic1_current_dose` column satisfies `ic1_current`). G2 current columns store coulombs over a 1 ms timeslice and are scaled to nA by `1e12`. G3 current columns are already nA and are not scaled. Quoted CSV fields are preserved. The result is columns, not a table of rows.

## SK-REQ-007

A compute kernel can update a storage buffer and read it back when a GPU adapter is present.

The kernel lives in `scan-kit-compute`. `wgpu` is the only compute library. The check compiles the shader even when the machine has no adapter, and it skips the dispatch in that case. No scientific kernel is part of this requirement.

## SK-REQ-008

Opening an analysis view returns one RGBA frame and the control list for the selected sessions.

`scan_kit_run_view` builds the scene, paints it, and returns the frame with the controls. At most five sessions are accepted. The shell renders those controls. It loads the same scene as one payload through `scan_kit_open_plot` and draws later frames in the webview with the same renderer.

## SK-REQ-009

A plot scene of polylines, points, bars, and one heatmap renders to an RGBA frame.

Colors for the frame background and series come from the caller. When no GPU adapter is present, the same marks are projected with the plot matrix and filled on the CPU.

## SK-REQ-010

Spot position remap, dose ratio, dose error, calibration, and the beam-state filter have one implementation.

## SK-REQ-011

A timeslice load returns device-unit columns and the `input_map.csv` energy lookup length. The channel catalog lists the concepts present for Replay.

## SK-REQ-012

Withdrawn. Dose Accumulation is not an analysis view.

## SK-REQ-013

Withdrawn. IC Peak Amplitude — Beam-Off is not an analysis view. Peak amplitude remains a timeslice source for Distribution Explorer.

## SK-REQ-014

Withdrawn. Beam Error Motion vs Energy is not an analysis view.

## SK-REQ-015

Distribution Explorer draws position, position error, or sigma as scatter or a density grid. Confidence is beam-on confidence against peak amplitude. Coverage is the percent of spots whose confidence stays above each threshold.

## SK-REQ-016

Binned summaries use quantile bin edges and a histogram of the values in each bin.

## SK-REQ-017

Binned Summary draws the 1.8 metric groups (dose error, dose ratio, dose rate, current ratio, IC current, position error, sigma, sigma error, IC2 minus IC1, and spot time) against energy, target MU, spot time, or radius. Energy is one bin per value and the other axes use quantile bins. Glyphs are violin, box, mean, scatter, and contour, with the 1.8 trend, histogram, correlation, fliers, the segment list, and interlock lines. Timeslice Replay stacks every channel of one timeslice source from the shared Y menu, and filters those rows with the segment list. The playhead is one more segment on that list: the previous one second, or every row at or before the playhead. An optional FFT column sits beside each channel, and an optional side scatter uses the Distribution Explorer source menu. Zoom reads those traces; the scene is not a fixed overview or a strided detail.

## SK-REQ-018

Timeslice Replay can draw a Welch spectrum beside each channel, from 1 Hz to 500 Hz, using 4096-sample segments and 50% overlap.

## SK-REQ-019

Welch PSD of a pure tone peaks at that tone.

## SK-REQ-021

Withdrawn. The ramp-down decay fit is not part of the product.

## SK-REQ-022

IC HV Transient draws the current trace and the measured capacitance next to the firmware grade.

## SK-REQ-023

A line fit returns slope and intercept.

## SK-REQ-024

Session Log Compare returns overview counts, timeline rows, error issues, watchdog mismatches, and a template diff when two sessions are selected.

## SK-REQ-025

IC1 sits 100 mm downstream of IC2. Dose Volume uses that separation.

## SK-REQ-026

Withdrawn. IC Beam Trajectory is not an analysis view.

## SK-REQ-027

An analytic Gaussian splat peaks on the spot. The splat, ray march, gamma, DVH, resample, and Monte Carlo transport shaders compile.

## SK-REQ-028

Dose Volume draws a maximum-intensity projection, sagittal and coronal slices, depth and lateral profiles, a DVH, a gamma comparison with the requested charge, and a resampled projection.

## SK-REQ-030

Gamma, DVH, and nearest-neighbor resample match their reference cases.

## SK-REQ-032

A DICOM folder of explicit little-endian files indexes by modality. A clinical goal reports the volume fraction at or above a dose, and the study report includes both.

## SK-REQ-033

Spot current sums, HV step capacitance, coverage percent, and session-log timeline comparison match their reference cases.

## Trace

Verification links are in [traceability.md](traceability.md).
