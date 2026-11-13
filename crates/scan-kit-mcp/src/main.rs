//! MCP server. Registers each crate's tool list and dispatches to that crate.
//! It does not implement the operations.

use std::sync::Arc;

use rmcp::{
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
        ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
    },
    service::RequestContext,
    transport::stdio,
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt,
};
use serde_json::{Map, Value};

/// Tools an agent gets without `--allow-effects`. Every other tool writes outside the
/// session index, drives a device or the network, or changes reference configuration,
/// so a prompt injected through session data cannot reach it. A new tool stays hidden
/// until it is listed here.
const READ_ONLY: &[&str] = &[
    "scan_kit_version",
    "scan_kit_health",
    "scan_kit_about",
    "scan_kit_calibrate",
    "scan_kit_dose_error",
    "scan_kit_beam_mask",
    "scan_kit_bin_edges",
    "scan_kit_histogram",
    "scan_kit_welch",
    "scan_kit_open_library",
    "scan_kit_set_note",
    "scan_kit_select_sessions",
    "scan_kit_data_dirs",
    "scan_kit_read_catalog",
    "scan_kit_load_columns",
    "scan_kit_load_timeslice",
    "scan_kit_channel_catalog",
    "scan_kit_analysis_scene",
    "scan_kit_plan_catalog",
    "scan_kit_synthesize_plan",
    "scan_kit_config_catalog",
    "scan_kit_config_open",
    "scan_kit_config_form",
    "scan_kit_config_apply",
    "scan_kit_config_tune",
    "scan_kit_config_diff",
    "scan_kit_config_integrity",
    "scan_kit_config_hide",
    "scan_kit_phantom_catalog",
    "scan_kit_phantom_preview",
    "scan_kit_runner_catalog",
    "scan_kit_run_view",
    "scan_kit_open_study",
    "scan_kit_clinical_goal",
];

#[derive(Clone, Default)]
struct ScanKit {
    effects: bool,
}

impl ScanKit {
    /// `None` when the call may run. Plan synthesis stays listed either way: building
    /// the CSV is read-only, and only saving it to `path` is an effect.
    fn blocked<'a>(&self, name: &'a str, input: &Value) -> Option<&'a str> {
        if self.effects {
            return None;
        }
        if !READ_ONLY.contains(&name) {
            return Some(name);
        }
        let saves = name == "scan_kit_synthesize_plan"
            && input
                .get("path")
                .and_then(Value::as_str)
                .is_some_and(|path| !path.trim().is_empty());
        saves.then_some(name)
    }

    fn allows(&self, name: &str) -> bool {
        self.blocked(name, &Value::Null).is_none()
    }
}

fn schema_object(value: Value) -> Arc<Map<String, Value>> {
    let Value::Object(schema) = value else {
        unreachable!("tool schema is an object");
    };
    Arc::new(schema)
}

fn catalog_specs() -> Vec<scan_kit_core::ToolSpec> {
    let mut specs = scan_kit_core::tools().to_vec();
    specs.extend_from_slice(scan_kit_io::tools());
    specs.extend_from_slice(scan_kit_compute::tools());
    specs.extend_from_slice(scan_kit_dicom::tools());
    specs
}

fn schema_for(name: &str) -> Value {
    if scan_kit_core::tools().iter().any(|tool| tool.name == name) {
        scan_kit_core::tool_input_schema(name)
    } else if scan_kit_io::tools().iter().any(|tool| tool.name == name) {
        scan_kit_io::tool_input_schema(name)
    } else if scan_kit_compute::tools()
        .iter()
        .any(|tool| tool.name == name)
    {
        scan_kit_compute::tool_input_schema(name)
    } else {
        scan_kit_dicom::tool_input_schema(name)
    }
}

fn dispatch(name: &str, input: &Value) -> Result<Value, String> {
    if scan_kit_core::tools().iter().any(|tool| tool.name == name) {
        scan_kit_core::invoke(name, input).map_err(|err| err.to_string())
    } else if scan_kit_io::tools().iter().any(|tool| tool.name == name) {
        scan_kit_io::invoke(name, input).map_err(|err| err.to_string())
    } else if scan_kit_compute::tools()
        .iter()
        .any(|tool| tool.name == name)
    {
        scan_kit_compute::invoke(name, input)
    } else if scan_kit_dicom::tools().iter().any(|tool| tool.name == name) {
        scan_kit_dicom::invoke(name, input)
    } else {
        Err(format!("unknown tool {name}"))
    }
}

fn catalog_tools(server: &ScanKit) -> Vec<Tool> {
    catalog_specs()
        .into_iter()
        .filter(|spec| server.allows(spec.name))
        .map(|spec| {
            Tool::new_with_raw(
                spec.name,
                Some(spec.summary.into()),
                schema_object(schema_for(spec.name)),
            )
        })
        .collect()
}

impl ServerHandler for ScanKit {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("scan-kit", scan_kit_core::version()))
            .with_instructions(
                "Scan Kit tools. Granular tools are single operations. Workflow tools compose them in process.",
            )
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        catalog_tools(self)
            .into_iter()
            .find(|tool| tool.name == name)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(catalog_tools(self)))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let input = match request.arguments {
            None => Value::Null,
            Some(arguments) => Value::Object(arguments),
        };
        let result = match self.blocked(&request.name, &input) {
            Some(name) => Err(format!(
                "{name} changes files, devices, or reference configuration; start the server with --allow-effects to use it"
            )),
            None => dispatch(&request.name, &input),
        };
        match result {
            Ok(value) => Ok(CallToolResult::structured(value).into()),
            Err(error) => Ok(CallToolResult::error(vec![ContentBlock::text(error)]).into()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let effects = std::env::args().any(|arg| arg == "--allow-effects");
    let server = ScanKit { effects }.serve(stdio()).await?;
    server.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rmcp::model::CallToolRequestParams;
    use rmcp::ServiceExt;
    use serde_json::json;

    use super::ScanKit;

    #[tokio::test]
    async fn sk_req_002_mcp_lists_and_calls_version_and_health() {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let server = tokio::spawn(async move {
            let running = ScanKit::default().serve(server_io).await.expect("server");
            running.waiting().await.expect("server wait");
        });

        let client = ().serve(client_io).await.expect("client");
        let listed = client.list_tools(None).await.expect("list tools");
        let names: Vec<&str> = listed.tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert!(names.contains(&"scan_kit_version"));
        assert!(names.contains(&"scan_kit_health"));

        let version = client
            .call_tool(CallToolRequestParams::new("scan_kit_version"))
            .await
            .expect("version");
        assert_eq!(
            version.structured_content,
            Some(json!({ "version": scan_kit_core::version() }))
        );

        let health = client
            .call_tool(CallToolRequestParams::new("scan_kit_health"))
            .await
            .expect("health");
        assert_eq!(
            health.structured_content,
            Some(json!({
                "version": scan_kit_core::version(),
                "catalog_non_empty": true
            }))
        );

        client.cancel().await.expect("cancel");
        server.await.expect("server task");
    }

    fn arguments(value: serde_json::Value) -> Option<serde_json::Map<String, serde_json::Value>> {
        match value {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        }
    }

    #[tokio::test]
    async fn sk_req_003_mcp_calls_about() {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let server = tokio::spawn(async move {
            let running = ScanKit::default().serve(server_io).await.expect("server");
            running.waiting().await.expect("server wait");
        });
        let client = ().serve(client_io).await.expect("client");
        let listed = client.list_tools(None).await.expect("list tools");
        let names: Vec<&str> = listed.tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert!(names.contains(&"scan_kit_about"));
        assert!(names.contains(&"scan_kit_open_library"));
        assert!(names.contains(&"scan_kit_set_note"));
        assert!(names.contains(&"scan_kit_select_sessions"));
        assert!(names.contains(&"scan_kit_load_columns"));

        let mut request = CallToolRequestParams::new("scan_kit_about");
        request.arguments = arguments(json!({}));
        let about = client.call_tool(request).await.expect("about");
        assert_eq!(
            about
                .structured_content
                .as_ref()
                .and_then(|value| value.get("version")),
            Some(&json!(scan_kit_core::version()))
        );
        client.cancel().await.expect("cancel");
        server.await.expect("server task");
    }

    #[tokio::test]
    async fn sk_req_004_mcp_opens_library() {
        let root = std::env::temp_dir().join(format!(
            "scan-kit-mcp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = root.join("sess");
        std::fs::create_dir_all(&session).unwrap();
        std::fs::write(
            session.join("input_map.csv"),
            "energy,X_POSITION,Y_POSITION\n1,0,0\n",
        )
        .unwrap();
        std::fs::write(
            session.join("termination_summary.txt"),
            "Date: Thu Sep 10 21:07:41 2026\nConfiguration name: mcp\n",
        )
        .unwrap();
        let db = root.join("test.sqlite");

        let (client_io, server_io) = tokio::io::duplex(256 * 1024);
        let server = tokio::spawn(async move {
            let running = ScanKit::default().serve(server_io).await.expect("server");
            running.waiting().await.expect("server wait");
        });
        let client = ().serve(client_io).await.expect("client");
        let mut request = CallToolRequestParams::new("scan_kit_open_library");
        request.arguments = arguments(json!({
            "path": root.to_string_lossy(),
            "db_path": db.to_string_lossy(),
        }));
        let opened = client.call_tool(request).await.expect("open library");
        let rows = opened
            .structured_content
            .as_ref()
            .and_then(|value| value.get("rows"))
            .and_then(|value| value.as_array())
            .expect("rows");
        assert!(rows
            .iter()
            .any(|row| row["session_id"] == "sess" && row["config"] == "mcp"));
        client.cancel().await.expect("cancel");
        server.await.expect("server task");
        let _ = std::fs::remove_dir_all(root);
    }

    async fn listed_and_control(effects: bool) -> (Vec<String>, bool, bool) {
        let (client_io, server_io) = tokio::io::duplex(64 * 1024);
        let server = tokio::spawn(async move {
            let running = ScanKit { effects }.serve(server_io).await.expect("server");
            running.waiting().await.expect("server wait");
        });
        let client = ().serve(client_io).await.expect("client");
        let listed = client.list_tools(None).await.expect("list tools");
        let names = listed
            .tools
            .iter()
            .map(|tool| tool.name.to_string())
            .collect();
        let mut request = CallToolRequestParams::new("scan_kit_runner_control");
        request.arguments = arguments(json!({ "action": "start" }));
        let refused = client
            .call_tool(request)
            .await
            .expect("call")
            .content
            .iter()
            .any(|block| format!("{block:?}").contains("--allow-effects"));
        let plan = std::env::temp_dir().join(format!(
            "scan-kit-mcp-plan-{}-{effects}.csv",
            std::process::id()
        ));
        let mut save = CallToolRequestParams::new("scan_kit_synthesize_plan");
        save.arguments = arguments(json!({
            "template": "zero_field",
            "params": {},
            "path": plan,
        }));
        let save_refused = client
            .call_tool(save)
            .await
            .expect("save")
            .content
            .iter()
            .any(|block| format!("{block:?}").contains("--allow-effects"));
        client.cancel().await.expect("cancel");
        server.await.expect("server task");
        let _ = std::fs::remove_file(plan);
        (names, refused, save_refused)
    }

    #[tokio::test]
    async fn device_and_write_tools_need_allow_effects() {
        let (names, refused, save_refused) = listed_and_control(false).await;
        assert!(names.iter().any(|name| name == "scan_kit_run_view"));
        assert!(names.iter().any(|name| name == "scan_kit_synthesize_plan"));
        assert!(save_refused);
        for gated in [
            "scan_kit_runner_control",
            "scan_kit_runner_upload",
            "scan_kit_config_save",
            "scan_kit_write_report",
            "scan_kit_discover_subnet",
        ] {
            assert!(!names.iter().any(|name| name == gated), "{gated} is listed");
        }
        assert!(refused);

        let (names, refused, save_refused) = listed_and_control(true).await;
        assert!(names.iter().any(|name| name == "scan_kit_runner_control"));
        assert!(!refused);
        assert!(!save_refused);
    }
}
