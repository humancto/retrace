//! The app menu: File > Open Video…, Add Camera…; Help > Open Data Folder, GitHub, third-party
//! software.

use crate::ctx::Ctx;
use tauri::menu::{Menu, MenuEvent, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};
use tauri::{AppHandle, Manager, Runtime, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;

pub const GITHUB: &str = "https://github.com/humancto/retrace";
const VIDEO_EXT: &[&str] = &[
    "mp4", "mov", "m4v", "avi", "mkv", "webm", "mpg", "mpeg", "ts", "flv", "wmv",
];

pub fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let open = MenuItemBuilder::with_id("open_video", "Open Video…")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let camera = MenuItemBuilder::with_id("add_camera", "Add Camera…")
        .accelerator("CmdOrCtrl+Shift+N")
        .build(app)?;
    let file = SubmenuBuilder::new(app, "File")
        .item(&open)
        .item(&camera)
        .separator()
        .close_window()
        .build()?;
    let edit = SubmenuBuilder::new(app, "Edit")
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;
    let view = SubmenuBuilder::new(app, "Window")
        .minimize()
        .item(&PredefinedMenuItem::fullscreen(app, None)?)
        .build()?;
    let help = SubmenuBuilder::new(app, "Help")
        .item(&MenuItemBuilder::with_id("open_data", "Open Data Folder").build(app)?)
        .item(&MenuItemBuilder::with_id("github", "Retrace on GitHub").build(app)?)
        .item(&MenuItemBuilder::with_id("third_party", "Third-Party Software").build(app)?)
        .build()?;
    let menu = Menu::new(app)?;
    #[cfg(target_os = "macos")]
    {
        let app_menu = SubmenuBuilder::new(app, "Retrace")
            .about(None)
            .separator()
            .services()
            .separator()
            .hide()
            .hide_others()
            .show_all()
            .separator()
            .quit()
            .build()?;
        menu.append(&app_menu)?;
    }
    menu.append_items(&[&file, &edit, &view, &help])?;
    Ok(menu)
}

pub fn handle(app: &AppHandle, event: MenuEvent) {
    match event.id().as_ref() {
        "open_video" => open_video(app),
        "add_camera" => open_camera_window(app),
        "open_data" => {
            let root = app.state::<Ctx>().paths.root.to_string_lossy().into_owned();
            if let Err(e) = app.opener().open_path(root, None::<&str>) {
                tracing::warn!("open data folder: {e}");
            }
        }
        "github" => {
            let _ = app.opener().open_url(GITHUB, None::<&str>);
        }
        "third_party" => {
            let p = app.state::<Ctx>().paths.root.join("THIRD-PARTY.txt");
            if !p.exists() {
                let _ = std::fs::write(&p, crate::tools::THIRD_PARTY);
            }
            if let Err(e) = app.opener().open_path(p.to_string_lossy(), None::<&str>) {
                tracing::warn!("open third-party notices: {e}");
            }
        }
        _ => {}
    }
}

fn open_video(app: &AppHandle) {
    let ctx = app.state::<Ctx>();
    let last = ctx.settings.lock().unwrap().last_dir.clone();
    let mut dlg = app.dialog().file().add_filter("Video", VIDEO_EXT);
    if let Some(d) = last.filter(|d| d.is_dir()) {
        dlg = dlg.set_directory(d);
    }
    let app = app.clone();
    // The callback form: a blocking dialog on the main thread deadlocks on macOS.
    dlg.pick_file(move |picked| {
        let Some(path) = picked.and_then(|f| f.into_path().ok()) else {
            return;
        };
        let ctx = app.state::<Ctx>();
        {
            let mut s = ctx.settings.lock().unwrap();
            s.last_dir = path.parent().map(|p| p.to_path_buf());
            let _ = s.save(&ctx.paths.settings);
        }
        ctx.open_file(path);
    });
}

pub fn open_camera_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("camera") {
        let _ = w.set_focus();
        return;
    }
    let r = WebviewWindowBuilder::new(app, "camera", WebviewUrl::App("camera.html".into()))
        .title("Add Camera")
        .inner_size(460.0, 330.0)
        .resizable(false)
        .build();
    if let Err(e) = r {
        tracing::warn!("open the Add Camera window: {e}");
    }
}
