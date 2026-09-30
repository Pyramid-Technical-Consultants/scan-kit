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

#[tauri::command]
fn scan_kit_open_study(path: String) -> Result<Value, String> {
    scan_kit_dicom::invoke("scan_kit_open_study", &json!({ "path": path }))
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
            scan_kit_open_study
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
