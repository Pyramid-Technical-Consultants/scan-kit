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
| SK-REQ-004 | `crates/scan-kit-core` `sk_req_004_termination_summary_parses`; `crates/scan-kit-io` `sk_req_004_open_library_reuses_fingerprint`; `crates/scan-kit-io` `urls_classify_and_drop_the_password`; `crates/scan-kit-io` `a_nested_session_is_found_and_a_duplicate_is_named`; `crates/scan-kit-io` `a_walk_stops_at_eight_levels`; `crates/scan-kit-io` `a_remote_listing_is_cached_when_the_view_opens`; `crates/scan-kit-io` `a_remote_open_asks_again_when_the_password_is_missing`; `crates/scan-kit-io` `saved_locations_share_one_session_list_and_exams_stay_separate`; `crates/scan-kit-dicom` `peek_stops_at_pixel_data_and_reads_the_study_header`; `crates/scan-kit-mcp` `sk_req_004_mcp_opens_library` |
| SK-REQ-005 | `crates/scan-kit-io` `sk_req_005_notes_and_selection_round_trip`; `crates/scan-kit-io` `sk_req_005_migrates_user_version_1`; `crates/scan-kit-io` `saved_locations_share_one_session_list_and_exams_stay_separate` |
| SK-REQ-006 | `crates/scan-kit-core` `sk_req_006_g2_alias_and_scale`; `crates/scan-kit-io` `sk_req_006_quoted_csv_projects_two_columns_and_scales_g2` |
| SK-REQ-007 | `crates/scan-kit-compute` `sk_req_007_storage_buffer_round_trip` |
| SK-REQ-008 | `crates/scan-kit-compute` `sk_req_008_run_view_returns_a_frame_and_controls` |
| SK-REQ-009 | `crates/scan-kit-plot` `sk_req_009_plot_shader_compiles_and_draws_a_line` |
| SK-REQ-010 | `crates/scan-kit-core` `sk_req_010_remap_calibration_and_beam_mask` |
| SK-REQ-011 | `crates/scan-kit-io` `sk_req_011_timeslice_load_reports_columns` |
| SK-REQ-012 | Withdrawn |
| SK-REQ-013 | Withdrawn |
| SK-REQ-014 | Withdrawn |
| SK-REQ-015 | `crates/scan-kit-io` `sk_req_015_distribution_exposes_spot_modes` |
| SK-REQ-016 | `crates/scan-kit-core` `sk_req_016_quantile_edges_and_histogram` |
| SK-REQ-017 | `crates/scan-kit-io` `sk_req_017_bins_and_timeline_share_the_session`; `crates/scan-kit-io` `timeline_scatter_keeps_rows_past_the_playhead`; `crates/scan-kit-plot` `follow_time_hides_scatter_points_outside_the_playhead`; `crates/scan-kit-plot` `follow_time_draws_the_time_trace_inside_the_playhead` |
| SK-REQ-018 | `crates/scan-kit-io` `sk_req_018_fft_draws_a_spectrum` |
| SK-REQ-019 | `crates/scan-kit-core` `sk_req_019_welch_sees_a_tone` |
| SK-REQ-021 | Withdrawn |
| SK-REQ-022 | `crates/scan-kit-io` `sk_req_022_hv_draws_the_current` |
| SK-REQ-023 | `crates/scan-kit-core` `sk_req_023_line_fit_returns_slope_and_intercept` |
| SK-REQ-024 | `crates/scan-kit-io` `sk_req_024_session_log_table` |
| SK-REQ-025 | `crates/scan-kit-core` `sk_req_025_chambers_are_100_mm_apart` |
| SK-REQ-026 | Withdrawn |
| SK-REQ-027 | `crates/scan-kit-core` `sk_req_027_splat_peaks_on_the_spot`; `crates/scan-kit-compute` `sk_req_027_through_031_scientific_shaders_compile` |
| SK-REQ-028 | `crates/scan-kit-io` `sk_req_028_volumetric_has_slices_dvh_and_gamma` |
| SK-REQ-030 | `crates/scan-kit-core` `sk_req_030_gamma_dvh_and_resample` |
| SK-REQ-032 | `crates/scan-kit-dicom` `sk_req_032_study_index_goal_and_report` |
| SK-REQ-033 | `crates/scan-kit-core` `sk_req_033_spot_sums_hv_and_coverage`; `crates/scan-kit-core` `sk_req_033_session_log_timeline_issue_and_diff` |
| SK-REQ-034 | `crates/scan-kit-io` `sk_req_034_amplifier_voltage_is_not_the_error` |
| SK-REQ-035 | `crates/scan-kit-core` `sk_req_035_relative_position_radius_sigma_percent_and_dose_per_mu`; `crates/scan-kit-io` `sk_req_035_spot_table_removes_one_nozzle_offset` |

The desktop shell shows the same version string. Its command calls `scan_kit_core::version` and does not have a separate requirement.
