# Scan Kit 2 engineering guide

This is a project guide for people and agents changing the repository. It is not an IEC 62304 architectural design deliverable. Class A does not require one. The development plan is [software-development-plan.md](software-development-plan.md).

## Product shape

Scan Kit 2.0 replaces the Python, Qt, and VisPy application with Rust libraries and a Tauri 2 desktop shell. Domain behavior lives in Rust. The desktop UI, the MCP server, and tests call the same functions. Heavy numeric work uses `wgpu` when those features are ported, in a crate that does not exist yet.

```text
Cargo.toml                      workspace
rust-toolchain.toml             stable pin
crates/scan-kit-core/           operations and the tool catalog
crates/scan-kit-mcp/            stdio MCP server
apps/desktop/                   React, Tailwind, shadcn
apps/desktop/src-tauri/         Tauri 2 shell
```

`scan-kit-core` does not depend on Tauri or the MCP SDK. `scan-kit-mcp` registers the catalog and transports it. The Tauri shell calls core through a command.

Future crates, not created until a port needs them:

- `scan-kit-io` for session files, settings, and remote reads
- `scan-kit-compute` for `wgpu` kernels
- a DICOM crate when DICOM is ported

WGSL that still lives under `scan_kit/assets/` stays there until the compute crate exists.

## MCP first

A behavior is a function on `scan-kit-core`. Each public operation is also a tool: a stable name, a one-line summary, JSON input and output, and a kind of `granular` or `workflow`.

`scan-kit-mcp` registers whatever `tools()` returns. It does not reimplement the operation. Workflow tools call granular functions in process. They do not call themselves through MCP.

The shell gets data through a Tauri command that calls the same core function.

Phase 1 tools:

- `scan_kit_version`, granular, returns the workspace version `2.0.0-dev`
- `scan_kit_health`, workflow, calls the version function and reports that the catalog is non-empty

Agents in this repository launch the server with `.cursor/mcp.json` (`cargo run -p scan-kit-mcp --quiet` on stdio).

## One implementation

There is no PyO3 bridge and no second copy of a ported behavior.

The Python tree is the unported backlog. A port change implements the capability in core, adds the tool, adds the Tauri surface when the capability had a user interface, and deletes the Python modules and tests that exist only for it. A module that unported code still imports stays until its last caller is gone. If a file is still under `scan_kit/` or `tests/`, it is not ported.

Phase 1 does not delete product Python. Nothing has been ported yet.

## Versions and branches

`scan_kit/__init__.py` is the released Python version until 2.0 is promoted. The Rust workspace version is `2.0.0-dev`. Do not bump the Python version to 2.0 to match the workspace.

Day-to-day pull requests target `develop`. `main` is the release branch. After the first deletion of ported Python, `develop` is no longer a full 1.x application. User-facing releases wait for the promotion of `develop` to `main`.

## User interface

The desktop UI is React, Tailwind, and shadcn. Dark mode is always on. The palette, radius, and type are the stock shadcn theme written by `shadcn` init. Do not edit those tokens to restyle a screen.

Screens use shadcn semantic tokens (`bg-background`, `text-foreground`, `bg-card`, `border-border`, and the rest of that theme) and components added with `shadcn add`. No second component library. No custom CSS for color, radius, type, or spacing.

A unique visual style, when it exists, is a deliberate change to that theme, not one-off overrides in a feature change.

## Where not to put logic

- Not in the React components, except rendering and calls to Tauri commands.
- Not in `scan-kit-mcp`, except protocol registration and transport.
- Not in a new Python module.
- Not in a crate created ahead of the port that needs it.
