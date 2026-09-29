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

## Trace

Verification links are in [traceability.md](traceability.md).
