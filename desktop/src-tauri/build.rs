fn main() {
    // App commands are listed here so they go through the capability ACL: each window gets only
    // the commands its capability file allows, and the engine origin (http://127.0.0.1) gets none.
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "setup_state",
            "retry_setup",
            "use_system_tools",
            "use_download_tools",
            "add_camera",
        ]),
    ))
    .expect("tauri build script");
}
