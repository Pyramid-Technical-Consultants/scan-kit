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
| SK-REQ-003 | `crates/scan-kit-core` `sk_req_003_about_includes_version`; `crates/scan-kit-mcp` `sk_req_003_mcp_calls_about` |
| SK-REQ-004 | `crates/scan-kit-core` `sk_req_004_termination_summary_parses`; `crates/scan-kit-io` `sk_req_004_open_library_reuses_fingerprint`; `crates/scan-kit-mcp` `sk_req_004_mcp_opens_library` |
| SK-REQ-005 | `crates/scan-kit-io` `sk_req_005_notes_and_selection_round_trip`; `crates/scan-kit-io` `sk_req_005_migrates_user_version_1` |
| SK-REQ-006 | `crates/scan-kit-core` `sk_req_006_g2_alias_and_scale`; `crates/scan-kit-io` `sk_req_006_quoted_csv_projects_two_columns_and_scales_g2` |
| SK-REQ-007 | `crates/scan-kit-compute` `sk_req_007_storage_buffer_round_trip` |

The desktop shell shows the same version string. Its command calls `scan_kit_core::version` and does not have a separate requirement.
