# Scan Kit 2 engineering guide

This is a project guide for people and agents changing the repository. It is not an IEC 62304 architectural design deliverable. Class A does not require one. The development plan is [software-development-plan.md](software-development-plan.md).

## Product shape

Scan Kit 2.0 replaces the Python, Qt, and VisPy application with Rust libraries and a Tauri 2 desktop shell. Domain behavior lives in Rust. The desktop UI, the MCP server, and tests call the same functions.

```text
Cargo.toml                      workspace
rust-toolchain.toml             stable pin
crates/scan-kit-core/           pure behavior: version, tools, summary parse, column aliases
crates/scan-kit-io/             session discovery, the sqlite store, columnar loads
crates/scan-kit-plot/           wgpu plot renderer, native and wasm32
crates/scan-kit-compute/        wgpu kernels, view scenes, and MCP plot frames
crates/scan-kit-dicom/          DICOM study index, clinical goals, and report
crates/scan-kit-mcp/            stdio MCP server
apps/desktop/                   React, Tailwind, shadcn, Glide Data Grid
apps/desktop/src-tauri/         Tauri 2 shell
```

`scan-kit-core` does not depend on Tauri, the MCP SDK, sqlite, or `wgpu`. `scan-kit-io` and `scan-kit-compute` depend on core. `scan-kit-plot` depends on core. `scan-kit-compute` also depends on `scan-kit-io` so one function can load a view, and on `scan-kit-plot` to paint it. Only `scan-kit-plot` and `scan-kit-compute` link `wgpu`. `scan-kit-mcp` registers the concatenation of each crate's tool list and dispatches to that crate. It does not reimplement an operation. The Tauri shell calls the same functions through commands.

`scan-kit-dicom` reads an explicit little-endian DICOM folder. The Monte Carlo transport shader lives in `scan-kit-compute`. The material tables stay under `scan_kit/assets/`.

## MCP first

A behavior is a function on the crate that owns it. Each public operation is also a tool: a stable name, a one-line summary, JSON input and output, and a kind of `granular` or `workflow`.

`scan-kit-mcp` registers whatever each crate's `tools()` returns. Workflow tools call granular functions in process. They do not call themselves through MCP.

The shell gets data through a Tauri command that calls the same function.

Tools:

- `scan_kit_version`, granular, returns the workspace version `2.0.0-dev`
- `scan_kit_health`, workflow, calls the version function and reports that the catalog is non-empty
- `scan_kit_about`, granular, returns the About dialog text and the workspace version
- `scan_kit_open_library`, workflow, discovers a local data folder, syncs the sqlite index, and returns the session rows
- `scan_kit_set_note` and `scan_kit_select_sessions`, granular
- `scan_kit_load_columns`, granular, returns requested columns after alias resolution and the G2 current scale
- `scan_kit_load_timeslice` and `scan_kit_channel_catalog`, granular
- `scan_kit_analysis_scene`, workflow, builds one view scene from the selected sessions
- `scan_kit_run_view`, workflow, paints that scene to one RGBA frame for MCP and tests
- `scan_kit_calibrate`, `scan_kit_dose_error`, `scan_kit_beam_mask`, `scan_kit_bin_edges`, `scan_kit_histogram`, `scan_kit_welch`, and `scan_kit_fit_decay`, granular
- `scan_kit_open_study` and `scan_kit_clinical_goal`, the DICOM study index and a dose-volume goal
- `scan_kit_plan_catalog`, granular, lists the four plan templates and their parameter specs
- `scan_kit_synthesize_plan`, workflow, builds an input map CSV from one of those templates
- `scan_kit_config_catalog`, granular, lists the four configuration tuners and the remembered config folder
- `scan_kit_config_open`, `scan_kit_config_form`, and `scan_kit_config_apply`, granular, browse a config folder and edit its XML through a form
- `scan_kit_config_save`, workflow, writes the folder and refreshes Pyramid `.md5` sidecars
- `scan_kit_config_tune`, workflow, updates devices.xml from the selected sessions
- `scan_kit_config_integrity` and `scan_kit_config_hide`, granular
- `scan_kit_phantom_catalog` and `scan_kit_phantom_preview`, granular, the phantom form and its CT summary
- `scan_kit_write_phantom`, workflow, writes a synthetic CT, RTSTRUCT, and RT Ion plan
- `scan_kit_runner_catalog` and `scan_kit_runner_status`, granular, the Plan Runner operator view
- `scan_kit_runner_connect`, `scan_kit_runner_upload`, `scan_kit_runner_control`, and `scan_kit_runner_download`, workflow, the live RCI session
- `scan_kit_runner_disconnect` and `scan_kit_runner_remember`, granular

Agents in this repository launch the server with `.cursor/mcp.json` (`cargo run -p scan-kit-mcp --quiet` on stdio).

## One implementation

There is no PyO3 bridge and no second copy of a ported behavior.

The Python tree is the unported backlog. A port change implements the capability in the owning crate, adds the tool, adds the Tauri surface when the capability had a user interface, and deletes the Python modules and tests that exist only for it. A module that unported code still imports stays until its last caller is gone. If a file is still under `scan_kit/` or `tests/`, it is not ported.

The Qt app still imports the session store, discovery, and schema modules, so this phase does not delete them. The Rust copies are the ones the new shell calls.

## Versions and branches

`scan_kit/__init__.py` is the released Python version until 2.0 is promoted. The Rust workspace version is `2.0.0-dev`. Do not bump the Python version to 2.0 to match the workspace.

Day-to-day pull requests target `develop`. `main` is the release branch. After the first deletion of ported Python, `develop` is no longer a full 1.x application. User-facing releases wait for the promotion of `develop` to `main`.

## User interface

The desktop chrome is React, Tailwind, and shadcn. Dark mode is always on. The palette, radius, and type are the stock shadcn theme written by `shadcn` init. Do not edit those tokens to restyle a screen.

Chrome uses shadcn semantic tokens (`bg-background`, `text-foreground`, `bg-card`, `border-border`, and the rest of that theme) and components added with `shadcn add`. The menu bar is a shadcn menu, not a native menu. There is no theme menu.

Tabular data, including the session list and Session Log Compare, is drawn by [Glide Data Grid](https://grid.glideapps.com/). The grid theme is filled from the stock tokens (background, card, foreground, muted foreground, border, accent, and the Geist font). Those tokens are not edited. A DOM table, including a shadcn Table, is not used for data.

The Analysis menu opens a view when one to five sessions are selected. Controls are shadcn components added with `shadcn add` (Select and Field for choices), not native form elements. A control change calls `scan_kit_open_plot`, which returns one binary payload: a JSON header (controls, table, samples, panel frames) and the encoded marks. The webview loads that payload into `scan-kit-plot` built for wasm32 and draws on the canvas with WebGPU, or WebGL2 where WebGPU is missing. Wheel, drag, hover, and resize stay in the webview. No frame crosses the Tauri bridge. `scan_kit_run_view` renders the same `Plot` offscreen and returns one base64 frame for MCP and tests, colored from the stock tokens. Audio Explorer plays and exports the open plot's samples with Web Audio. Dose Volume can open a DICOM folder through `scan_kit_open_study`.

A plotted series contains every sample that passed the view's filters. Nothing drops rows before the camera exists: no fixed stride, bucket count, or point cap on a line or a scatter. Histograms, contours, and spectra still reduce the samples, and they count every sample that passed the filter. The renderer may later simplify a stroke for the current camera when several samples fall in one pixel. That simplification keeps the extrema in the pixel, and a camera that gives a sample its own pixel draws the sample. Zoom and pan stay in the webview, so a thinned payload can never grow back.

Row filters are one segment list, combined with AND, and the mask is computed once per table. Beam, rank, a column range, and a threshold compare are kinds of that list. A later cut, including energy, time, and a dose threshold, is another kind. Playback writes the time range; it does not grow a second filter.

No custom CSS for color, radius, type, or spacing. A unique visual style, when it exists, is a deliberate change to the shadcn theme, not one-off overrides in a feature change.

## Numeric work

`wgpu` is the only GPU library. Drawing lives in `scan-kit-plot`, which builds natively and for `wasm32-unknown-unknown` from one source and one WGSL shader. The analytic splat, ray march, gamma, DVH, resample, and Monte Carlo transport shaders live in `scan-kit-compute`. Nothing else links `wgpu`. Marks are stored as `vec3` in data space. One `clip_from_data` matrix places them, and a later orbit or volume writes that same matrix. The shader stays inside the WebGL2 downlevel limits: no nonzero base instance and one color target. Series hover is a CPU hit test on the same marks. The native path reads a frame back as RGBA for MCP. The readback test skips the dispatch when the machine has no adapter and still compiles the shader. The no-adapter picture projects the same buffers.

No dataframe crate and no ORM. Session columns are `Vec<f32>` or `Vec<i32>`, parsed in one pass.

A view that takes long enough to block the window runs as a task: one bounded slice per poll, one progress report, and cancel by generation. Plot payloads stay the binary scene the webview already draws. The slice rules, the Monte Carlo preview contract, and the view faces are in [progressive-loading.md](progressive-loading.md).

## Session data

The database path, table names, and `user_version` 3 match the Python store: `~/.scan-kit/scan-kit.sqlite`, tables `prefs`, `libraries`, and `sessions`, WAL, foreign keys. An existing file opens as-is. This window uses the last data folder and the window geometry prefs.

Discovery reads a local directory of session folders and `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.tar.xz`. It reads `termination_summary.txt` out of a folder or archive and does not unpack the archive. Unpacked folders win over an archive with the same id. The sqlite index skips re-parsing when size and mtime match. Map extent is filled from spot positions only when the cached row does not already have it.

SFTP and SMB are not in this phase. A normal path, including a Windows UNC path, needs no extra crate.

## Where not to put logic

- Not in the React components, except rendering and calls to Tauri commands.
- Not in `scan-kit-mcp`, except protocol registration and transport.
- Not in a new Python module.
- Not in a second GPU stack, a dataframe crate, or an ORM.
