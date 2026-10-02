use serde_json::{json, Value};

#[tauri::command]
fn version() -> &'static str {
    scan_kit_core::version()
}

#[tauri::command]
fn scan_kit_about() -> Result<Value, String> {
    scan_kit_core::invoke("scan_kit_about", &Value::Null).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_open_library(path: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_open_library", &json!({ "path": path }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_set_note(path: String, session_id: String, note: String) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_set_note",
        &json!({ "path": path, "session_id": session_id, "note": note }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_select_sessions(path: String, session_ids: Vec<String>) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_select_sessions",
        &json!({ "path": path, "session_ids": session_ids }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_last_data_dir() -> Result<Option<String>, String> {
    scan_kit_io::read_last_data_dir()
}

#[tauri::command]
fn scan_kit_window_geometry() -> Result<Value, String> {
    scan_kit_io::read_window_geometry()
}

#[tauri::command]
fn scan_kit_set_window_geometry(width: i32, height: i32, x: i32, y: i32) -> Result<(), String> {
    scan_kit_io::write_window_geometry(width, height, x, y)
}

#[tauri::command]
fn scan_kit_last_main_tab() -> Result<Option<String>, String> {
    scan_kit_io::read_last_main_tab()
}

#[tauri::command]
fn scan_kit_set_last_main_tab(tab: String) -> Result<(), String> {
    scan_kit_io::write_last_main_tab(&tab)
}

#[tauri::command]
fn scan_kit_run_view(
    view: String,
    path: String,
    session_ids: Vec<String>,
    options: Value,
    width: u32,
    height: u32,
    background: Vec<f32>,
    foreground: Vec<f32>,
    palette: Vec<Vec<f32>>,
) -> Result<Value, String> {
    let background = color4(&background, [0.11, 0.11, 0.12, 1.0]);
    let foreground = color4(&foreground, [0.92, 0.92, 0.93, 1.0]);
    let palette: Vec<[f32; 4]> = palette
        .iter()
        .map(|row| color4(row, [0.9, 0.9, 0.9, 1.0]))
        .collect();
    scan_kit_compute::run_view(
        &view,
        std::path::Path::new(&path),
        &session_ids,
        &options,
        width,
        height,
        background,
        foreground,
        &palette,
    )
}

/// The packed scene for the webview's wasm plot. Pan, zoom, and hover stay in the webview.
#[tauri::command]
async fn scan_kit_open_plot(
    view: String,
    path: String,
    session_ids: Vec<String>,
    options: Value,
    background: Vec<f32>,
    foreground: Vec<f32>,
    palette: Vec<Vec<f32>>,
) -> Result<tauri::ipc::Response, String> {
    let background = color4(&background, [0.11, 0.11, 0.12, 1.0]);
    let foreground = color4(&foreground, [0.92, 0.92, 0.93, 1.0]);
    let palette: Vec<[f32; 4]> = palette
        .iter()
        .map(|row| color4(row, [0.9, 0.9, 0.9, 1.0]))
        .collect();
    let bytes = tauri::async_runtime::spawn_blocking(move || {
        scan_kit_compute::open_plot(
            &view,
            std::path::Path::new(&path),
            &session_ids,
            &options,
            background,
            foreground,
            &palette,
        )
    })
    .await
    .map_err(|err| err.to_string())??;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
fn scan_kit_open_study(path: String) -> Result<Value, String> {
    scan_kit_dicom::invoke("scan_kit_open_study", &json!({ "path": path }))
}

#[tauri::command]
fn scan_kit_plan_catalog() -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_plan_catalog", &json!({})).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_synthesize_plan(
    template: String,
    params: Value,
    path: Option<String>,
    csv: Option<String>,
) -> Result<Value, String> {
    let mut body = json!({ "template": template, "params": params });
    if let Some(path) = path {
        body["path"] = json!(path);
    }
    if let Some(csv) = csv {
        body["csv"] = json!(csv);
    }
    scan_kit_io::invoke("scan_kit_synthesize_plan", &body).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_catalog() -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_config_catalog", &json!({})).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_open(path: Option<String>) -> Result<Value, String> {
    let mut body = json!({});
    if let Some(path) = path {
        body["path"] = json!(path);
    }
    scan_kit_io::invoke("scan_kit_config_open", &body).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_form(path: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_config_form", &json!({ "path": path }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_apply(xml: String, form: Value) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_config_apply",
        &json!({ "xml": xml, "form": form }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_save(
    source: String,
    dest: String,
    files: Value,
    hide_unused: Option<bool>,
) -> Result<Value, String> {
    let mut body = json!({ "source": source, "dest": dest, "files": files });
    if let Some(hide_unused) = hide_unused {
        body["hide_unused"] = json!(hide_unused);
    }
    scan_kit_io::invoke("scan_kit_config_save", &body).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_tune(
    workflow: String,
    xml: String,
    form: Option<Value>,
    data_dir: String,
    session_ids: Vec<String>,
    params: Value,
) -> Result<Value, String> {
    let mut body = json!({
        "workflow": workflow,
        "xml": xml,
        "data_dir": data_dir,
        "session_ids": session_ids,
        "params": params,
    });
    if let Some(form) = form {
        body["form"] = form;
    }
    scan_kit_io::invoke("scan_kit_config_tune", &body).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_phantom_catalog() -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_phantom_catalog", &json!({})).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_phantom_preview(params: Value) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_phantom_preview", &json!({ "params": params }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_write_phantom(parent: String, params: Value) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_write_phantom",
        &json!({ "parent": parent, "params": params }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_config_hide(hide_unused: bool) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_config_hide",
        &json!({ "hide_unused": hide_unused }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_catalog() -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_catalog", &json!({})).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_connect(host: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_connect", &json!({ "host": host }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_disconnect() -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_disconnect", &json!({})).map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_status(has_plan: Option<bool>, dest: Option<String>) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_runner_status",
        &json!({ "has_plan": has_plan.unwrap_or(false), "dest": dest.unwrap_or_default() }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_upload(path: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_upload", &json!({ "path": path }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_control(action: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_control", &json!({ "action": action }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_download(dest: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_download", &json!({ "dest": dest }))
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_runner_remember(file_dir: String) -> Result<Value, String> {
    scan_kit_io::invoke("scan_kit_runner_remember", &json!({ "file_dir": file_dir }))
        .map_err(|err| err.to_string())
}

fn color4(values: &[f32], fallback: [f32; 4]) -> [f32; 4] {
    let mut out = fallback;
    for (index, value) in values.iter().take(4).enumerate() {
        out[index] = *value;
    }
    out
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            version,
            scan_kit_about,
            scan_kit_open_library,
            scan_kit_set_note,
            scan_kit_select_sessions,
            scan_kit_last_data_dir,
            scan_kit_window_geometry,
            scan_kit_set_window_geometry,
            scan_kit_last_main_tab,
            scan_kit_set_last_main_tab,
            scan_kit_run_view,
            scan_kit_open_plot,
            scan_kit_open_study,
            scan_kit_plan_catalog,
            scan_kit_synthesize_plan,
            scan_kit_config_catalog,
            scan_kit_config_open,
            scan_kit_config_form,
            scan_kit_config_apply,
            scan_kit_config_save,
            scan_kit_config_tune,
            scan_kit_config_hide,
            scan_kit_phantom_catalog,
            scan_kit_phantom_preview,
            scan_kit_write_phantom,
            scan_kit_runner_catalog,
            scan_kit_runner_connect,
            scan_kit_runner_disconnect,
            scan_kit_runner_status,
            scan_kit_runner_upload,
            scan_kit_runner_control,
            scan_kit_runner_download,
            scan_kit_runner_remember
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
