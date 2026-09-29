//! MCP server. Registers [`scan_kit_core::tools`] and dispatches to
//! [`scan_kit_core::invoke`]. It does not implement the operations.

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

#[derive(Clone)]
struct ScanKit;

fn empty_input_schema() -> Arc<Map<String, Value>> {
    let Value::Object(schema) = serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    }) else {
        unreachable!("schema is an object");
    };
    Arc::new(schema)
}

fn catalog_tools() -> Vec<Tool> {
    let schema = empty_input_schema();
    scan_kit_core::tools()
        .iter()
        .map(|spec| Tool::new_with_raw(spec.name, Some(spec.summary.into()), Arc::clone(&schema)))
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
        catalog_tools().into_iter().find(|tool| tool.name == name)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(catalog_tools()))
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
        match scan_kit_core::invoke(&request.name, &input) {
            Ok(value) => Ok(CallToolResult::structured(value).into()),
            Err(error) => {
                Ok(CallToolResult::error(vec![ContentBlock::text(error.to_string())]).into())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = ScanKit.serve(stdio()).await?;
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
            let running = ScanKit.serve(server_io).await.expect("server");
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
}
