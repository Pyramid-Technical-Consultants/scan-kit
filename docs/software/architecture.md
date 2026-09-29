# Scan Kit 2 engineering guide

This is a project guide for people and agents changing the repository. It is not an IEC 62304 architectural design deliverable. Class A does not require one. The development plan is [software-development-plan.md](software-development-plan.md).

## Product shape

Scan Kit 2.0 replaces the Python, Qt, and VisPy application with Rust libraries and a Tauri 2 desktop shell. Domain behavior lives in Rust. The desktop UI, the MCP server, and tests call the same functions.

```text
Cargo.toml                      workspace
rust-toolchain.toml             stable pin
crates/scan-kit-core/           pure behavior: version, tools, summary parse, column aliases
crates/scan-kit-io/             session discovery, the sqlite store, columnar loads
crates/scan-kit-compute/        wgpu device and kernels
crates/scan-kit-mcp/            stdio MCP server
apps/desktop/                   React, Tailwind, shadcn, Glide Data Grid
apps/desktop/src-tauri/         Tauri 2 shell
```

`scan-kit-core` does not depend on Tauri, the MCP SDK, sqlite, or `wgpu`. `scan-kit-io` and `scan-kit-compute` depend on core. Only `scan-kit-compute` links `wgpu`. `scan-kit-mcp` registers the concatenation of each crate's tool list and dispatches to that crate. It does not reimplement an operation. The Tauri shell calls the same functions through commands.

A DICOM crate waits until DICOM is ported. WGSL that still lives under `scan_kit/assets/` stays there until a kernel moves into `scan-kit-compute`.

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

Tabular data, including the session list and later column views, is drawn by [Glide Data Grid](https://grid.glideapps.com/). The grid theme is filled from the stock tokens (background, card, foreground, muted foreground, border, accent, and the Geist font). Those tokens are not edited. A DOM table, including a shadcn Table, is not used for data. Later views that show loaded columns use this same grid, fed by `load_columns`.

No custom CSS for color, radius, type, or spacing. A unique visual style, when it exists, is a deliberate change to the shadcn theme, not one-off overrides in a feature change.

## Numeric work

`wgpu` is the only compute library. It lives in `scan-kit-compute`. Nothing else links it. This phase stands up the device, one storage-buffer dispatch, and a readback so later kernels use that path. No scientific kernel moves yet. The readback test skips when the machine has no adapter and still compiles the shader.

No dataframe crate and no ORM. Session columns are `Vec<f32>` or `Vec<i32>`, parsed in one pass.

## Session data

The database path, table names, and `user_version` 3 match the Python store: `~/.scan-kit/scan-kit.sqlite`, tables `prefs`, `libraries`, and `sessions`, WAL, foreign keys. An existing file opens as-is. This window uses the last data folder and the window geometry prefs.

Discovery reads a local directory of session folders and `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.tar.xz`. It reads `termination_summary.txt` out of a folder or archive and does not unpack the archive. Unpacked folders win over an archive with the same id. The sqlite index skips re-parsing when size and mtime match. Map extent is filled from spot positions only when the cached row does not already have it.

SFTP and SMB are not in this phase. A normal path, including a Windows UNC path, needs no extra crate.

## Where not to put logic

- Not in the React components, except rendering and calls to Tauri commands.
- Not in `scan-kit-mcp`, except protocol registration and transport.
- Not in a new Python module.
- Not in a second GPU stack, a dataframe crate, or an ORM.
