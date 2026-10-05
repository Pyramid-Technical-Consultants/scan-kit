use serde_json::{json, Value};

/// A command whose JSON object uses each argument name as the key.
macro_rules! forward {
    ($name:ident) => {
        #[tauri::command]
        fn $name() -> Result<Value, String> {
            scan_kit_io::invoke(stringify!($name), &json!({})).map_err(|err| err.to_string())
        }
    };
    ($name:ident ( $($arg:ident : $ty:ty),+ $(,)? )) => {
        #[tauri::command]
        fn $name($($arg: $ty),+) -> Result<Value, String> {
            scan_kit_io::invoke(stringify!($name), &json!({ $(stringify!($arg): $arg),+ }))
                .map_err(|err| err.to_string())
        }
    };
}

forward!(scan_kit_open_library(path: String));
forward!(scan_kit_set_note(path: String, session_id: String, note: String));
forward!(scan_kit_select_sessions(path: String, session_ids: Vec<String>));
forward!(scan_kit_plan_catalog);
forward!(scan_kit_config_catalog);
forward!(scan_kit_config_form(path: String));
forward!(scan_kit_config_apply(xml: String, form: Value));
forward!(scan_kit_config_hide(hide_unused: bool));
forward!(scan_kit_phantom_catalog);
forward!(scan_kit_phantom_preview(params: Value));
forward!(scan_kit_write_phantom(parent: String, params: Value));
forward!(scan_kit_runner_catalog);
forward!(scan_kit_runner_connect(host: String));
forward!(scan_kit_runner_disconnect);
forward!(scan_kit_runner_upload(path: String));
forward!(scan_kit_runner_control(action: String));
forward!(scan_kit_runner_download(dest: String));
forward!(scan_kit_runner_remember(file_dir: String));

#[tauri::command]
fn version() -> &'static str {
    scan_kit_core::version()
}

#[tauri::command]
fn scan_kit_about() -> Result<Value, String> {
    scan_kit_core::invoke("scan_kit_about", &Value::Null).map_err(|err| err.to_string())
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

/// Start a progressive task. The webview polls it and draws each payload itself.
#[tauri::command]
async fn scan_kit_start(
    view: String,
    path: String,
    session_ids: Vec<String>,
    options: Value,
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
    tauri::async_runtime::spawn_blocking(move || {
        scan_kit_compute::start_task(
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
    .map_err(|err| err.to_string())?
}

/// One slice of a task: a progress report and, when this slice drew something, its payload.
#[tauri::command]
async fn scan_kit_poll(task: u64) -> Result<tauri::ipc::Response, String> {
    let bytes = tauri::async_runtime::spawn_blocking(move || scan_kit_compute::poll_task(task))
        .await
        .map_err(|err| err.to_string())??;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
async fn scan_kit_cancel(task: u64) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || scan_kit_compute::cancel_task(task))
        .await
        .map_err(|err| err.to_string())
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
fn scan_kit_config_open(
    path: Option<String>,
    data_dir: Option<String>,
    session_id: Option<String>,
) -> Result<Value, String> {
    let mut body = json!({});
    if let Some(path) = path.filter(|text| !text.is_empty()) {
        body["path"] = json!(path);
    }
    if let Some(data_dir) = data_dir.filter(|text| !text.is_empty()) {
        body["data_dir"] = json!(data_dir);
    }
    if let Some(session_id) = session_id.filter(|text| !text.is_empty()) {
        body["session_id"] = json!(session_id);
    }
    scan_kit_io::invoke("scan_kit_config_open", &body).map_err(|err| err.to_string())
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
fn scan_kit_runner_status(has_plan: Option<bool>, dest: Option<String>) -> Result<Value, String> {
    scan_kit_io::invoke(
        "scan_kit_runner_status",
        &json!({ "has_plan": has_plan.unwrap_or(false), "dest": dest.unwrap_or_default() }),
    )
    .map_err(|err| err.to_string())
}

#[tauri::command]
fn scan_kit_reveal(path: String) -> Result<(), String> {
    let target = std::path::PathBuf::from(&path);
    if !target.exists() {
        return Err(format!("Nothing at {path}"));
    }
    // explorer.exe often exits nonzero after the window is already open.
    std::process::Command::new(reveal_program())
        .args(reveal_args(&target))
        .spawn()
        .map(|_| ())
        .map_err(|err| err.to_string())
}

fn reveal_program() -> &'static str {
    #[cfg(windows)]
    {
        "explorer"
    }
    #[cfg(target_os = "macos")]
    {
        "open"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "xdg-open"
    }
}

fn reveal_args(path: &std::path::Path) -> Vec<std::ffi::OsString> {
    #[cfg(windows)]
    {
        if path.is_dir() {
            vec![path.as_os_str().to_os_string()]
        } else {
            vec![std::ffi::OsString::from(format!(
                "/select,{}",
                path.display()
            ))]
        }
    }
    #[cfg(target_os = "macos")]
    {
        vec![
            std::ffi::OsString::from("-R"),
            path.as_os_str().to_os_string(),
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let folder = path.parent().unwrap_or(path);
        vec![folder.as_os_str().to_os_string()]
    }
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
            scan_kit_start,
            scan_kit_poll,
            scan_kit_cancel,
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
            scan_kit_runner_remember,
            scan_kit_reveal
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod reveal_tests {
    use super::reveal_args;

    #[test]
    fn a_file_is_selected_and_a_folder_is_opened() {
        let file = std::env::temp_dir().join("scan-kit-not-a-dir.csv");
        assert!(!reveal_args(&file).is_empty());
        assert!(!reveal_args(&std::env::temp_dir()).is_empty());
        #[cfg(windows)]
        {
            let file_arg = reveal_args(&file)[0].to_string_lossy().into_owned();
            assert!(file_arg.starts_with("/select,"));
            assert!(file_arg.contains("scan-kit-not-a-dir.csv"));
            assert!(!reveal_args(&std::env::temp_dir())[0]
                .to_string_lossy()
                .starts_with("/select,"));
        }
    }
}
