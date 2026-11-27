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
        // Unknown names fall through to `dispatch`, which reports `unknown tool`
        // — don't claim an unlisted typo needs --allow-effects.
        if !READ_ONLY.contains(&name) {
            return catalog_specs()
                .iter()
                .any(|tool| tool.name == name)
                .then_some(name);
        }
        let saves = name == "scan_kit_synthesize_plan"
            && input
                .get("path")
                .and_then(Value::as_str)
                .is_some_and(|path| !path.trim().is_empty());
        // A caller-chosen store is outside the index the sandbox covers —
        // it creates directories, sqlite files, and migrates foreign schemas.
        let foreign_db = input
            .get("db_path")
            .and_then(Value::as_str)
            .is_some_and(|path| !path.trim().is_empty());
        // A remote location spec reaches out to the network: SFTP TOFU
        // writes known_hosts, archives download into the remote cache,
        // SMB opens sessions, and FTP/HTTP touch any host named.
        // is_remote_location only sees scheme-prefixed specs — a UNC share
        // or file://host reaches the same wire through plain filesystem
        // calls, so the gate checks for those spellings itself.
        // Nested containers (params.dicom_path, options.study/scanners…) also
        // carry caller-chosen paths, so every string anywhere in the input is
        // checked — enumerating nested keys would just rot again.
        let remote = Self::remote_in(input, name);
        (saves || foreign_db || remote).then_some(name)
    }

    /// A UNC share is any two leading separators — `//`, `\\`, `\/`,
    /// `/\` all parse as UNC on Windows.
    fn unc(spec: &str) -> bool {
        let mut chars = spec.chars();
        matches!(chars.next(), Some('\\' | '/')) && matches!(chars.next(), Some('\\' | '/'))
    }

    /// True when a string looks like a network-reachable location: a remote
    /// scheme, a UNC share, or a file://host URI (file:/// is local).
    fn remote_spec(spec: &str) -> bool {
        let spec = spec.trim();
        // `\\?\` verbatim paths are local only in the ordinary drive form
        // (`\\?\C:\...`). Every other object-manager spelling —
        // `\\?\UNC\` shares, `GLOBALROOT` device paths reaching the
        // network redirectors, anything unrecognized — fails closed.
        if spec
            .get(..4)
            .is_some_and(|head| head.eq_ignore_ascii_case(r"\\?\"))
        {
            if spec
                .get(4..8)
                .is_some_and(|rest| rest.eq_ignore_ascii_case("UNC\\"))
            {
                return true;
            }
            let drive = spec.as_bytes().get(4).is_some_and(u8::is_ascii_alphabetic)
                && spec.as_bytes().get(5) == Some(&b':');
            return !drive;
        }
        // `\\.\` is the device namespace — pipes, serial ports, raw volumes.
        // None of it is an ordinary local file for the effects gate.
        if spec
            .get(..4)
            .is_some_and(|head| head.eq_ignore_ascii_case(r"\\.\"))
        {
            return true;
        }
        if scan_kit_io::is_remote_location(spec) || spec.starts_with('\\') || Self::unc(spec) {
            return true;
        }
        if spec
            .get(..7)
            .is_some_and(|head| head.eq_ignore_ascii_case("file://"))
        {
            // Decode like the open path does, then re-apply the UNC rules —
            // file:////host and %2f escapes still reach the network.
            let path = scan_kit_io::file_uri_path(&spec[7..]);
            return Self::unc(&path.to_string_lossy());
        }
        false
    }

    /// Walks every string in the input; a JSON-in-JSON value such as
    /// options.scanners (a string carrying an array) is walked too.
    fn remote_in(value: &Value, name: &str) -> bool {
        Self::remote_in_at(value, true, name, 0)
    }

    fn remote_in_at(value: &Value, top: bool, name: &str, depth: u8) -> bool {
        // Each string is re-parsed as JSON — serde caps one document's
        // nesting but not a doc-inside-string chain. An arg needing more
        // than a few re-parse levels is adversarial; gate it.
        if depth >= 4 {
            return true;
        }
        match value {
            Value::String(spec) => {
                Self::remote_spec(spec)
                    || serde_json::from_str::<Value>(spec)
                        .ok()
                        .is_some_and(|parsed| Self::remote_in_at(&parsed, false, name, depth + 1))
            }
            Value::Array(items) => items
                .iter()
                .any(|item| Self::remote_in_at(item, false, name, depth)),
            // `note` is the one free-text field in the tool surface — a note
            // that *is* a URL must not trip the path gate. The exemption is
            // top-level only and scoped to the one tool that takes a note:
            // a nested `note` key carrying a location would otherwise walk
            // straight past the gate. Keys are checked too — a remote spec
            // hiding in a field name must not slip through.
            Value::Object(map) => map.iter().any(|(key, value)| {
                Self::remote_spec(key)
                    || !(top && name == "scan_kit_set_note" && key == "note" && value.is_string())
                        && Self::remote_in_at(value, false, name, depth)
            }),
            _ => false,
        }
    }

    fn allows(&self, name: &str) -> bool {
        self.blocked(name, &Value::Null).is_none()
    }
}

fn schema_object(value: Value) -> Arc<Map<String, Value>> {
    match value {
        Value::Object(schema) => Arc::new(schema),
        // A non-object catalog schema must not panic the handler task —
        // rmcp never awaits the spawned task, so a panic hangs the client.
        _ => Arc::new(Map::new()),
    }
}

fn catalog_specs() -> &'static [scan_kit_core::ToolSpec] {
    static SPECS: std::sync::OnceLock<Vec<scan_kit_core::ToolSpec>> = std::sync::OnceLock::new();
    SPECS.get_or_init(|| {
        let mut specs = scan_kit_core::tools().to_vec();
        specs.extend_from_slice(scan_kit_io::tools());
        specs.extend_from_slice(scan_kit_compute::tools());
        specs.extend_from_slice(scan_kit_dicom::tools());
        // A name collision across catalogs would shadow the later tool.
        // Panics at startup — inside a spawned handler task a panic hangs
        // the client, since rmcp never joins the task.
        assert_eq!(
            specs
                .iter()
                .map(|tool| tool.name)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            specs.len(),
            "duplicate tool names across catalogs"
        );
        specs
    })
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

fn dispatch(
    name: &str,
    input: &Value,
    cancel: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<Value, String> {
    if scan_kit_core::tools().iter().any(|tool| tool.name == name) {
        scan_kit_core::invoke(name, input).map_err(|err| err.to_string())
    } else if scan_kit_io::tools().iter().any(|tool| tool.name == name) {
        scan_kit_io::invoke_flag(name, input, cancel.as_ref()).map_err(|err| err.to_string())
    } else if scan_kit_compute::tools()
        .iter()
        .any(|tool| tool.name == name)
    {
        scan_kit_compute::invoke_flag(
            name,
            input,
            &scan_kit_core::Cancel::from_arc(cancel.clone()),
        )
    } else if scan_kit_dicom::tools().iter().any(|tool| tool.name == name) {
        scan_kit_dicom::invoke(name, input)
    } else {
        Err(format!("unknown tool {name}"))
    }
}

fn catalog_tools(server: &ScanKit) -> Vec<Tool> {
    catalog_specs()
        .iter()
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
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let input = match request.arguments {
            None => Value::Null,
            Some(arguments) => Value::Object(arguments),
        };
        let name: String = request.name.chars().take(256).collect();
        let result = match self.blocked(&name, &input) {
            Some(name) => Err(format!(
                "{name} writes files/devices/configuration or reaches the network; start the server with --allow-effects to use it"
            )),
            None => {
                if !catalog_specs().iter().any(|tool| tool.name == name) {
                    return Err(McpError::invalid_params(
                        format!("unknown tool {name}"),
                        None,
                    ));
                }
                let ct = context.ct;
                // rmcp spawns one task per request; cap in-flight
                // dispatches so a flood can't queue unbounded blocking work.
                // The acquire itself races the ct — a request cancelled while
                // queued frees immediately instead of parking its (up to
                // 16 MiB) parsed args behind the 8 live dispatches.
                static DISPATCH_LIMIT: tokio::sync::Semaphore =
                    tokio::sync::Semaphore::const_new(8);
                // The permit must live as long as the work it admits — held
                // here in the async scope it would drop on cancellation
                // while the detached blocking task keeps running, leaking
                // the in-flight cap. It moves into the closure below.
                let permit = tokio::select! {
                    permit = DISPATCH_LIMIT.acquire() => permit.map_err(|_| {
                        McpError::internal_error("dispatch limiter closed", None)
                    })?,
                    _ = ct.cancelled() => {
                        return Err(McpError::internal_error("cancelled", None));
                    }
                };
                // The advertised schema is the contract — a typo'd arg
                // must not silently default (e.g. `db_pth` writes the
                // default index, `dest` drops a save). db_path is exempt:
                // dispatch reads it for every store tool regardless of
                // what the schema lists.
                if let Some(schema) = schema_for(&name).as_object() {
                    let args = input.as_object();
                    if let Some(missing) = schema
                        .get("required")
                        .and_then(Value::as_array)
                        .and_then(|required| {
                            required
                                .iter()
                                .filter_map(Value::as_str)
                                .find(|key| args.is_none_or(|args| !args.contains_key(*key)))
                        })
                    {
                        return Err(McpError::invalid_params(
                            format!("{name} requires {missing}"),
                            None,
                        ));
                    }
                    if let Some(extra) = args.and_then(|args| {
                        args.keys().find(|key| {
                            *key != "db_path"
                                && schema
                                    .get("properties")
                                    .and_then(Value::as_object)
                                    .is_none_or(|props| !props.contains_key(*key))
                        })
                    }) {
                        return Err(McpError::invalid_params(
                            format!("{name} does not take {extra}"),
                            None,
                        ));
                    }
                }
                // dispatch blocks on file IO, remote connect timeouts, and GPU
                // waits — pinning a tokio worker starves every other request.
                // The client's ct cancels on CancelledNotification: it frees
                // this request immediately, and the flag makes cooperative
                // tools (index scans) roll back instead of running to
                // completion in a detached task.
                // Cancelled mid-validation —
                // do not spawn work the client already abandoned.
                if ct.is_cancelled() {
                    return Err(McpError::internal_error("cancelled", None));
                }
                let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let flag = cancel.clone();
                let tool = name.clone();
                let joined = tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    dispatch(&name, &input, &cancel)
                });
                // A blocking task cannot be aborted mid-run, but selecting
                // frees this request now. The flag is set HERE, in the arm
                // — a watcher task needs a separate scheduling slot and
                // aborting it before it polls would leave flag false, so
                // the detached task would commit writes after 'cancelled'.
                let result = tokio::select! {
                    out = joined => match out {
                        Ok(inner) => inner,
                        Err(err) => {
                            // JoinError's Display embeds the panic payload —
                            // internal details stay server-side.
                            eprintln!("scan-kit-mcp: {tool} task failed: {err}");
                            Err(format!("{tool} failed"))
                        }
                    },
                    _ = ct.cancelled() => {
                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        // ponytail: the detached spawn_blocking still runs to
                        // completion — only tools that poll the flag (open_library,
                        // compute stages) actually stop. A cancelled write tool
                        // reports "cancelled" while its effect lands, and a slow
                        // core/dicom read (index_study, load_study) keeps burning
                        // IO; threading the flag into those invokes is the fix.
                        Err("cancelled".to_owned())
                    }
                };
                result
            }
        };
        match result {
            Ok(value) => {
                // structuredContent must be an object on the wire; the one
                // bare-array tool result (data_dirs) gets wrapped here rather
                // than changing the shape desktop invoke relies on.
                let value = if value.is_object() {
                    value
                } else {
                    serde_json::json!({ "result": value })
                };
                // rmcp silently discards response lines over 16 MiB — a
                // full-dim run_view frame would hang the client, not fail it.
                // CallToolResult::structured duplicates the payload — once
                // as structuredContent, once escaped inside content[].text
                // — so the wire line runs up to ~3x the measured size
                // against the 16 MiB transport cap.
                const MAX_RESULT: usize = 5 << 20;
                let oversized = serde_json::to_vec(&value)
                    .map(|bytes| bytes.len() > MAX_RESULT)
                    .unwrap_or(true);
                if oversized {
                    return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                        "{} produced a response over the transport's message limit",
                        request.name
                    ))])
                    .into());
                }
                Ok(CallToolResult::structured(value).into())
            }
            Err(error) => {
                // Error text is outside the success-path size guard — a
                // reflected oversized input must not cross the transport's
                // frame cap and turn a deliverable error into a dropped one.
                let capped: String = error.chars().take(4096).collect();
                Ok(CallToolResult::error(vec![ContentBlock::text(capped)]).into())
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut effects = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--allow-effects" => effects = true,
            _ => {
                eprintln!("unknown argument {arg}; usage: scan-kit-mcp [--allow-effects]");
                std::process::exit(2);
            }
        }
    }
    // Init the memoized catalog now — a duplicate tool name must fail at
    // startup, not inside a spawned handler task rmcp never joins.
    catalog_specs();
    let server = ScanKit { effects }.serve(stdio()).await?;
    server.waiting().await?;
    // A detached spawn_blocking task (e.g. a cancelled core/dicom read)
    // would otherwise keep the runtime — and this process — alive past
    // stdin EOF, a zombie holding the index and GPU.
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use rmcp::model::CallToolRequestParams;
    use rmcp::ServiceExt;
    use serde_json::json;

    use super::ScanKit;

    #[test]
    fn unc_like_paths_trip_the_remote_gate() {
        for spec in [
            "//nas/share/dose.dcm",
            r"\\nas\share",
            r"/\nas\share",
            r"\/nas/share",
            "file:////nas/share/dose.dcm",
            "file:///%2f%2fnas%2fshare",
            r"\\?\UNC\nas\share",
            r"\\?\GLOBALROOT\Device\Mup\nas\share",
            r"\\.\GLOBALROOT\Device\WebDavRedirector\nas\share",
            r"\\?\GLOBALROOT\??\UNC\nas\share",
            r"\\.\pipe\somepipe",
            r"\\.\COM1",
            r"\\.\PhysicalDrive0",
        ] {
            assert!(ScanKit::remote_spec(spec), "{spec}");
        }
        for local in [
            "C:/data/dose.dcm",
            "file:///C:/data/dose.dcm",
            "./rel/path",
            r"\\?\C:\data\dose.dcm",
        ] {
            assert!(!ScanKit::remote_spec(local), "{local}");
        }
    }

    #[test]
    fn a_nested_note_does_not_bypass_the_remote_gate() {
        // Only a top-level string `note` is exempt — a nested note key, or a
        // top-level note carrying a structured value, still trips the gate.
        assert!(!ScanKit::remote_in_at(
            &json!({"note": "sftp://device.invalid/data"}),
            true,
            "scan_kit_set_note",
            0
        ));
        assert!(ScanKit::remote_in_at(
            &json!({"args": {"note": "sftp://device.invalid/data"}}),
            true,
            "scan_kit_set_note",
            0
        ));
        assert!(ScanKit::remote_in_at(
            &json!({"note": {"path": "sftp://device.invalid/data"}}),
            true,
            "scan_kit_set_note",
            0
        ));
    }

    #[test]
    fn the_note_exemption_is_scoped_to_set_note() {
        // Another tool's top-level `note` carrying a location still trips the
        // gate - only scan_kit_set_note earns the exemption.
        assert!(ScanKit::remote_in_at(
            &json!({"note": "sftp://device.invalid/data"}),
            true,
            "scan_kit_run_view",
            0
        ));
    }

    #[test]
    fn a_remote_spec_in_a_key_trips_the_gate() {
        assert!(ScanKit::remote_in_at(
            &json!({"sftp://host.invalid/share": "x"}),
            true,
            "scan_kit_run_view",
            0
        ));
    }

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
            let running = ScanKit { effects: true }
                .serve(server_io)
                .await
                .expect("server");
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

    #[test]
    fn remote_location_args_count_as_effects() {
        let kit = ScanKit { effects: false };
        let remote = serde_json::json!({ "path": "sftp://host.example/lib" });
        assert_eq!(
            kit.blocked("scan_kit_open_library", &remote),
            Some("scan_kit_open_library")
        );
        let ftp = serde_json::json!({ "storage_path": "ftp://host.example/x.zip" });
        assert_eq!(
            kit.blocked("scan_kit_load_timeslice", &ftp),
            Some("scan_kit_load_timeslice")
        );
        let local = serde_json::json!({ "path": "C:\\lib" });
        assert_eq!(kit.blocked("scan_kit_open_library", &local), None);
        // A drive-letter scheme is local, not remote — is_remote_location
        // treats a one-letter scheme as a Windows drive.
        let drive = serde_json::json!({ "path": "C://lib/data" });
        assert_eq!(kit.blocked("scan_kit_open_library", &drive), None);
        let file = serde_json::json!({ "path": "file:///lib" });
        assert_eq!(kit.blocked("scan_kit_open_library", &file), None);
        // UNC and file://host reach the wire without a scheme the driver
        // classifier sees — the gate has to catch its own spellings.
        let unc = serde_json::json!({ "path": "\\\\filer\\share\\lib" });
        assert_eq!(
            kit.blocked("scan_kit_open_library", &unc),
            Some("scan_kit_open_library")
        );
        let hosted = serde_json::json!({ "path": "file://filer/share/lib" });
        assert_eq!(
            kit.blocked("scan_kit_open_library", &hosted),
            Some("scan_kit_open_library")
        );
    }

    #[test]
    fn nested_remote_locations_count_as_effects() {
        let kit = ScanKit { effects: false };
        // params.dicom_path / params.pld_path feed plain filesystem reads in
        // synthesize_plan — a UNC share reaches the wire through them.
        let params = serde_json::json!({
            "template": "iba_pld_plan",
            "params": { "pld_path": "\\\\evil\\share\\x.pld" }
        });
        assert_eq!(
            kit.blocked("scan_kit_synthesize_plan", &params),
            Some("scan_kit_synthesize_plan")
        );
        // options.study / options.scanner short-circuit volumetric into a
        // folder walk and a calibration lookup before sessions are touched.
        let study = serde_json::json!({ "options": { "study": "\\\\evil\\share\\dicom" } });
        assert_eq!(
            kit.blocked("scan_kit_run_view", &study),
            Some("scan_kit_run_view")
        );
        // options.scanners is a JSON string carrying an array — the nested
        // document is walked too.
        let scanners = serde_json::json!({
            "options": { "scanners": "[{\"label\":\"x\",\"folder\":\"\\\\\\\\evil\\\\share\"}]" }
        });
        assert_eq!(
            kit.blocked("scan_kit_run_view", &scanners),
            Some("scan_kit_run_view")
        );
        // Honest nested local paths still pass.
        let local = serde_json::json!({
            "params": { "dicom_path": "C:\\dicom\\study" },
            "options": { "study": "D:\\data", "scanner": "default" }
        });
        assert_eq!(kit.blocked("scan_kit_run_view", &local), None);
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

    #[test]
    fn a_nested_note_key_does_not_skip_the_gate() {
        let kit = ScanKit { effects: false };
        // Top-level `note` is the free-text exemption — scoped to the one
        // tool that takes a note.
        let top = serde_json::json!({ "note": "\\\\filer\\share\\x" });
        assert_eq!(kit.blocked("scan_kit_set_note", &top), None);
        assert_eq!(
            kit.blocked("scan_kit_run_view", &top),
            Some("scan_kit_run_view")
        );
        // A `note` key at any deeper level is just a string — and must not
        // smuggle a remote spec past the gate.
        let nested = serde_json::json!({ "options": { "note": "\\\\filer\\share\\x" } });
        assert_eq!(
            kit.blocked("scan_kit_run_view", &nested),
            Some("scan_kit_run_view")
        );
    }
}
