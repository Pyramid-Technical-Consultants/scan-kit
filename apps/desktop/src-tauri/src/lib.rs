#[tauri::command]
fn version() -> &'static str {
    scan_kit_core::version()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![version])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
