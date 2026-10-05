//! Domain operations and the tool catalog.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! implement the operations themselves. Session files live in `scan-kit-io`.
//! GPU work lives in `scan-kit-compute`.

mod config;
mod dose;
mod geometry;
mod plan;
mod plot;
mod ramp;
mod runner;
mod schema;
mod segment;
mod session;
mod session_log;
mod signal;
mod task;
mod tune;
mod xml_dom;

pub use config::{apply_form, config_form};
pub use dose::{
    analytic_on, analytic_volume, bragg_idd, csda_range_mm, dose_frame, field_bounds, medium,
    protons_from_mu, robust_high, through_wet, water, DoseFrame, McJob, McResult, Medium,
    PatientRequest, Pencil, Quantity, SlabRequest, Volume,
};
pub use geometry::{IC1_Z_MM, IC2_Z_MM, IC_SEP_MM};
pub use plan::{
    build_plan, dicom_beam_size, parse_pld, plan_catalog, standard_energies, validate_plan,
    ImportSpot, PlanDocument, PlanSource,
};
pub use plot::{
    apply_clip, format_tick, map_span, project, ticks, Camera, Choice, Control, DataTable, Panel,
    PlotRect, PlotScene, Series,
};
pub use ramp::{choices, index, is_session, resolve, sample, session, Family, SESSION, VIRIDIS};
pub use runner::{
    control_button, default_session_zip_path, device_file_url, field_subscribe_key, io_bool,
    io_number, io_url, normalize_host, parse_host, resolve_session_download, runner_view,
    session_download_hint, DEFAULT_CONTROL_POINTS, DEFAULT_SESSION_ROOT, LOAD_STATE_SUCCESS,
    POINTS_LOAD_STATE, POINTS_UPLOAD, POINTS_UPLOAD_TARGET, POINTS_VALID, SESSION_DIRECTORY,
    STATUS_PATHS,
};
pub use session_log::{compare_templates, parse_session_log, LayerEvent, SessionLog};
pub use signal::{
    assign_bin_centers, beam_on_mask, box_stats, calibration_factor, coverage_percent,
    dose_error_pct, dose_ratio_pct, dvh, g2_ic2_mm, gamma_index, histogram, hv_capacitance_pf,
    hv_delta_v, hv_expected_pf, hv_firmware_flags, hv_step_window, linear_fit, median_finite,
    mip_xy, quantile_edges, remap, remap_g2_raw, remap_g2_raw_reversed, remap_g3_raw,
    remap_g3_raw_reversed, resample_nearest, scale_column, splat_gaussians, sums_by_spot_id,
    sums_by_spot_run, trapz, welch_psd, BoxStats, G2_MM_PER_STRIP, G2_STRIP_CENTER,
    G3_STRIP_CENTER, G3_STRIP_PITCH_MM,
};
pub use tune::{run_tune, tune_catalog, TuneSpots};
pub use xml_dom::{parse_xml, write_xml, Elem};

pub use schema::{
    column_is_integer, column_scale_factor, concept_column_candidates, normalize_column_name,
    resolve_column_name, resolve_concept_column, resolve_requested_column,
    timeslice_amplifier_field_columns, ConceptAlias, DerivedSum, CONCEPT_ALIASES, DERIVED_SUMS,
    IC3_QUAD_ZERO_FILL, POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW,
};
pub use segment::{
    apply_mask, parse_segments, row_mask, scrub_control, segments_control, segments_from,
    segments_json, time_end, BeamGate, CompareOp, Rank, Segment,
};
pub use session::{merge_session_geom, parse_termination_summary_text, SessionMeta, SummaryDate};
pub use task::{
    adapt_chunk, decode_poll, encode_poll, generation_matches, settle, Cancel, Phase, Poll, Report,
};

use serde::Serialize;
use serde_json::{json, Value};

/// Workspace version. The released Python package version is a different identity.
pub const VERSION: &str = "2.0.0-dev";

/// Whether a tool is a single operation or a composition of those operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Granular,
    Workflow,
}

/// One callable operation. The MCP server registers this list as-is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: &'static str,
    pub summary: &'static str,
    pub kind: ToolKind,
}

const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        name: "scan_kit_version",
        summary: "Return the Scan Kit workspace version.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_health",
        summary: "Report the workspace version and that the tool catalog is non-empty.",
        kind: ToolKind::Workflow,
    },
    ToolSpec {
        name: "scan_kit_about",
        summary: "Return the About dialog text and the workspace version.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_calibrate",
        summary: "Scale factor that zeros the dose-weighted error of one delivered column.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_dose_error",
        summary: "Percent dose error of delivered values against a target column.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_beam_mask",
        summary: "Beam-on mask from a gate column.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_bin_edges",
        summary: "Quantile bin edges for a numeric column.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_histogram",
        summary: "Histogram counts for a numeric column.",
        kind: ToolKind::Granular,
    },
    ToolSpec {
        name: "scan_kit_welch",
        summary: "Welch power spectrum of a 1 kHz timeslice signal.",
        kind: ToolKind::Granular,
    },
];

/// SK-REQ-001. The single Rust version string.
pub fn version() -> &'static str {
    VERSION
}

/// Tools an MCP client can list and call.
pub fn tools() -> &'static [ToolSpec] {
    TOOLS
}

/// JSON Schema for one core tool.
pub fn tool_input_schema(name: &str) -> Value {
    match name {
        "scan_kit_version" | "scan_kit_health" | "scan_kit_about" => json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        }),
        "scan_kit_calibrate" | "scan_kit_dose_error" => json!({
            "type": "object",
            "properties": {
                "target": { "type": "array", "items": { "type": "number" } },
                "delivered": { "type": "array", "items": { "type": "number" } }
            },
            "required": ["target", "delivered"],
            "additionalProperties": false
        }),
        "scan_kit_beam_mask" => json!({
            "type": "object",
            "properties": {
                "gate": { "type": "array", "items": { "type": "number" } }
            },
            "required": ["gate"],
            "additionalProperties": false
        }),
        "scan_kit_bin_edges" | "scan_kit_histogram" | "scan_kit_welch" => json!({
            "type": "object",
            "properties": {
                "values": { "type": "array", "items": { "type": "number" } },
                "bins": { "type": "integer" }
            },
            "required": ["values"],
            "additionalProperties": false
        }),
        _ => json!({ "type": "object", "additionalProperties": false }),
    }
}

fn f32_array(input: &Value, key: &str) -> Result<Vec<f32>, InvokeError> {
    input
        .get(key)
        .and_then(Value::as_array)
        .ok_or(InvokeError::UnexpectedInput)?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|number| number as f32)
                .ok_or(InvokeError::UnexpectedInput)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VersionReport {
    pub version: &'static str,
}

/// Granular tool `scan_kit_version`.
pub fn scan_kit_version() -> VersionReport {
    VersionReport { version: version() }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HealthReport {
    pub version: &'static str,
    pub catalog_non_empty: bool,
}

/// Workflow tool `scan_kit_health`. Calls [`scan_kit_version`] in process.
pub fn scan_kit_health() -> HealthReport {
    HealthReport {
        version: scan_kit_version().version,
        catalog_non_empty: !tools().is_empty(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AboutReport {
    pub version: &'static str,
    pub title: &'static str,
    pub product_line: String,
    pub description: &'static str,
    pub source_lead: &'static str,
    pub github_label: &'static str,
    pub github_url: &'static str,
    pub maintainer: &'static str,
    pub website_label: &'static str,
    pub website_url: &'static str,
    pub support_lead: &'static str,
    pub support_email: &'static str,
    pub copyright: &'static str,
    pub license_lead: &'static str,
    pub license_label: &'static str,
    pub license_url: &'static str,
}

/// Granular tool `scan_kit_about`. Same sentences as the Python About dialog.
pub fn scan_kit_about() -> AboutReport {
    AboutReport {
        version: version(),
        title: "About Scan Kit",
        product_line: format!("Scan Kit v{}", version()),
        description: "Analysis, plan synthesis, and configuration tuning for pencil-beam scanning session data.",
        source_lead: "Free and open-source software —",
        github_label: "view it on GitHub",
        github_url: "https://github.com/Pyramid-Technical-Consultants/scan-kit",
        maintainer: "Maintained by Pyramid Technical Consultants, Inc.",
        website_label: "www.pyramid.tech",
        website_url: "https://www.pyramid.tech",
        support_lead: "Questions or support:",
        support_email: "support@ptcusa.com",
        copyright: "Copyright © 2026 Pyramid Technical Consultants.",
        license_lead: "Released under the",
        license_label: "MIT License",
        license_url: "https://github.com/Pyramid-Technical-Consultants/scan-kit/blob/main/LICENSE",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvokeError {
    UnknownTool { name: String },
    UnexpectedInput,
}

impl std::fmt::Display for InvokeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool { name } => write!(f, "unknown tool {name}"),
            Self::UnexpectedInput => write!(f, "this tool takes no arguments"),
        }
    }
}

impl std::error::Error for InvokeError {}

/// Run a catalog tool. Version, health, and about take no arguments.
pub fn invoke(name: &str, input: &Value) -> Result<Value, InvokeError> {
    if matches!(
        name,
        "scan_kit_version" | "scan_kit_health" | "scan_kit_about"
    ) && !input.is_null()
        && input.as_object().is_none_or(|object| !object.is_empty())
    {
        return Err(InvokeError::UnexpectedInput);
    }
    let value = match name {
        "scan_kit_version" => serde_json::to_value(scan_kit_version()).expect("version report"),
        "scan_kit_health" => serde_json::to_value(scan_kit_health()).expect("health report"),
        "scan_kit_about" => serde_json::to_value(scan_kit_about()).expect("about report"),
        "scan_kit_calibrate" => {
            let target = f32_array(input, "target")?;
            let delivered = f32_array(input, "delivered")?;
            json!({ "factor": calibration_factor(&target, &delivered) })
        }
        "scan_kit_dose_error" => {
            let target = f32_array(input, "target")?;
            let delivered = f32_array(input, "delivered")?;
            json!({ "error_pct": dose_error_pct(&delivered, &target) })
        }
        "scan_kit_beam_mask" => {
            let gate = f32_array(input, "gate")?;
            json!({ "beam_on": beam_on_mask(&gate) })
        }
        "scan_kit_bin_edges" => {
            let values = f32_array(input, "values")?;
            let bins = input.get("bins").and_then(Value::as_u64).unwrap_or(8) as usize;
            json!({ "edges": quantile_edges(&values, bins) })
        }
        "scan_kit_histogram" => {
            let values = f32_array(input, "values")?;
            let bins = input.get("bins").and_then(Value::as_u64).unwrap_or(16) as usize;
            let (edges, counts) = histogram(&values, bins);
            json!({ "edges": edges, "counts": counts })
        }
        "scan_kit_welch" => {
            let values = f32_array(input, "values")?;
            let (freqs, psd) = welch_psd(&values, 1000.0, 4096, 0.5);
            json!({ "freqs": freqs, "psd": psd })
        }
        _ => {
            return Err(InvokeError::UnknownTool {
                name: name.to_owned(),
            })
        }
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_001_version_is_a_single_string() {
        assert_eq!(version(), VERSION);
        assert_eq!(scan_kit_version().version, version());
        assert!(!version().is_empty());
        assert!(!version().contains(char::is_whitespace));
        assert_eq!(
            invoke("scan_kit_version", &Value::Null).unwrap(),
            serde_json::json!({ "version": VERSION })
        );
    }

    #[test]
    fn sk_req_002_health_uses_version_and_catalog() {
        let health = scan_kit_health();
        assert_eq!(health.version, scan_kit_version().version);
        assert!(health.catalog_non_empty);
        assert!(tools()
            .iter()
            .any(|tool| { tool.name == "scan_kit_version" && tool.kind == ToolKind::Granular }));
        assert!(tools()
            .iter()
            .any(|tool| { tool.name == "scan_kit_health" && tool.kind == ToolKind::Workflow }));
        assert_eq!(
            invoke("scan_kit_health", &serde_json::json!({})).unwrap(),
            serde_json::json!({
                "version": VERSION,
                "catalog_non_empty": true
            })
        );
    }

    #[test]
    fn sk_req_003_about_includes_version() {
        let about = scan_kit_about();
        assert_eq!(about.version, version());
        assert_eq!(about.product_line, format!("Scan Kit v{}", version()));
        assert!(about.description.contains("pencil-beam scanning"));
        assert_eq!(about.support_email, "support@ptcusa.com");
        assert!(about.copyright.contains("2026"));
        let value = invoke("scan_kit_about", &Value::Null).unwrap();
        assert_eq!(value["version"], version());
        assert_eq!(value["title"], "About Scan Kit");
        assert!(invoke("scan_kit_about", &json!({"extra": true})).is_err());
    }
}
