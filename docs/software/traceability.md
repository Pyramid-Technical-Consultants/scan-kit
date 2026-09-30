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
| SK-REQ-008 | `crates/scan-kit-compute` `sk_req_008_run_view_returns_a_frame_and_controls` |
| SK-REQ-009 | `crates/scan-kit-plot` `sk_req_009_plot_shader_compiles_and_draws_a_line` |
| SK-REQ-010 | `crates/scan-kit-core` `sk_req_010_remap_calibration_and_beam_mask` |
| SK-REQ-011 | `crates/scan-kit-io` `sk_req_011_timeslice_load_reports_columns` |
| SK-REQ-012 | `crates/scan-kit-io` `sk_req_012_dose_accumulation_scene_has_cumulative_lines` |
| SK-REQ-013 | `crates/scan-kit-io` `sk_req_013_peak_amplitude_is_beam_off_bars` |
| SK-REQ-014 | `crates/scan-kit-io` `sk_req_014_beam_motion_draws_spill_paths` |
| SK-REQ-015 | `crates/scan-kit-io` `sk_req_015_distribution_exposes_spot_modes` |
| SK-REQ-016 | `crates/scan-kit-core` `sk_req_016_quantile_edges_and_histogram` |
| SK-REQ-017 | `crates/scan-kit-io` `sk_req_017_binned_summary_and_replay_share_the_session` |
| SK-REQ-018 | `crates/scan-kit-io` `sk_req_018_fft_and_audio_share_the_spectrum` |
| SK-REQ-019 | `crates/scan-kit-core` `sk_req_019_welch_sees_a_tone` |
| SK-REQ-021 | `crates/scan-kit-core` `sk_req_021_decay_fit_recovers_tau` |
| SK-REQ-022 | `crates/scan-kit-io` `sk_req_022_rampdown_amplifier_and_hv` |
| SK-REQ-023 | `crates/scan-kit-core` `sk_req_023_settled_mask_waits_out_a_command_step` |
| SK-REQ-024 | `crates/scan-kit-io` `sk_req_024_session_log_table` |
| SK-REQ-025 | `crates/scan-kit-core` `sk_req_025_geometry_parses_and_fits` |
| SK-REQ-026 | `crates/scan-kit-io` `sk_req_026_trajectory_projects_a_path` |
| SK-REQ-027 | `crates/scan-kit-core` `sk_req_027_splat_peaks_on_the_spot`; `crates/scan-kit-compute` `sk_req_027_through_031_scientific_shaders_compile` |
| SK-REQ-028 | `crates/scan-kit-io` `sk_req_028_dose_volume_has_slices_dvh_and_gamma` |
| SK-REQ-030 | `crates/scan-kit-core` `sk_req_030_gamma_dvh_and_resample` |
| SK-REQ-032 | `crates/scan-kit-dicom` `sk_req_032_study_index_goal_and_report` |
| SK-REQ-033 | `crates/scan-kit-core` `sk_req_033_spot_sums_arc_hv_and_ramp_edges`; `crates/scan-kit-core` `sk_req_033_session_log_timeline_issue_and_diff` |

The desktop shell shows the same version string. Its command calls `scan_kit_core::version` and does not have a separate requirement.
