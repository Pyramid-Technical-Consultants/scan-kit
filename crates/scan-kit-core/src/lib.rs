//! Domain operations and the tool catalog.
//!
//! The desktop shell and the MCP server call these functions. They do not
//! implement the operations themselves.

use serde::Serialize;
use serde_json::Value;

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
];

/// SK-REQ-001. The single Rust version string.
pub fn version() -> &'static str {
    VERSION
}

/// Tools an MCP client can list and call.
pub fn tools() -> &'static [ToolSpec] {
    TOOLS
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
}
