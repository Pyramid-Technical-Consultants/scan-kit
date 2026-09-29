//! Domain operations and the tool catalog.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! implement the operations themselves. Session files live in `scan-kit-io`.
//! GPU work lives in `scan-kit-compute`.

mod schema;
mod session;

pub use schema::{
    column_is_integer, column_scale_factor, concept_column_candidates, normalize_column_name,
    resolve_column_name, resolve_concept_column, resolve_requested_column,
    timeslice_amplifier_field_columns, ConceptAlias, DerivedSum, CONCEPT_ALIASES, DERIVED_SUMS,
    IC3_QUAD_ZERO_FILL, POSITION_KEY_G2_RAW, POSITION_KEY_G3_RAW,
};
pub use session::{merge_session_geom, parse_termination_summary_text, SessionMeta, SummaryDate};

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
];

/// SK-REQ-001. The single Rust version string.
pub fn version() -> &'static str {
    VERSION
}

/// Tools an MCP client can list and call.
pub fn tools() -> &'static [ToolSpec] {
    TOOLS
}

/// JSON Schema for one core tool. Core tools take no arguments.
pub fn tool_input_schema(name: &str) -> Value {
    let _ = name;
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
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

/// Run a catalog tool. Both phase 1 tools take no arguments (`null` or `{}`).
pub fn invoke(name: &str, input: &Value) -> Result<Value, InvokeError> {
    if !input.is_null() && input.as_object().is_none_or(|object| !object.is_empty()) {
        return Err(InvokeError::UnexpectedInput);
    }
    let value = match name {
        "scan_kit_version" => serde_json::to_value(scan_kit_version()).expect("version report"),
        "scan_kit_health" => serde_json::to_value(scan_kit_health()).expect("health report"),
        "scan_kit_about" => serde_json::to_value(scan_kit_about()).expect("about report"),
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
