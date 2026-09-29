# Traceability

| | |
|---|---|
| Document | SK-TRACE |
| Product | Scan Kit 2 |

Git history is the revision record. Each requirement maps to the test that fails if the requirement breaks.

| Requirement | Test |
|---|---|
| SK-REQ-001 | `crates/scan-kit-core` `sk_req_001_version_is_a_single_string` |
| SK-REQ-002 | `crates/scan-kit-core` `sk_req_002_health_uses_version_and_catalog`; `crates/scan-kit-mcp` `sk_req_002_mcp_lists_and_calls_version_and_health` |

The desktop shell shows the same version string. Its command calls `scan_kit_core::version` and does not have a separate requirement.
