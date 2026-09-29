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

## Trace

Verification links are in [traceability.md](traceability.md).
