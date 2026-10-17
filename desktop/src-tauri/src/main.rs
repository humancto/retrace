//! Retrace Desktop: first-run setup, then the Retrace engine UI in a native window.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ctx;
mod engine;
mod logging;
mod menu;
mod nav;
mod paths;
mod settings;
mod setup;
mod tools;
mod worker;

use ctx::Ctx;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::webview::DownloadEvent;
use tauri::{
    AppHandle, DragDropEvent, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_opener::OpenerExt;

/// `--open <file>` (repeatable) and `--open=<file>`; relative paths resolve against `cwd`.
/// Other arguments (macOS adds `-psn_…`) are ignored.
pub fn parse_open_args(args: &[String], cwd: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        let v = if a == "--open" {
            it.next().cloned()
        } else {
            a.strip_prefix("--open=").map(str::to_string)
        };
        if let Some(v) = v.filter(|v| !v.is_empty()) {
            let p = PathBuf::from(v);
            out.push(if p.is_absolute() { p } else { cwd.join(p) });
        }
    }
    out
}

#[tauri::command]
fn setup_state(ctx: tauri::State<'_, Ctx>) -> setup::View {
    ctx.setup.lock().unwrap().clone()
}

#[tauri::command]
fn retry_setup(app: AppHandle) {
    setup::start(&app);
}

#[tauri::command]
fn use_system_tools(app: AppHandle) {
    {
        let ctx = app.state::<Ctx>();
        let mut s = ctx.settings.lock().unwrap();
        s.tools = settings::ToolSource::System;
        let _ = s.save(&ctx.paths.settings);
    }
    setup::start(&app);
}

#[tauri::command]
fn add_camera(app: AppHandle, name: String, url: String) -> Result<(), String> {
    worker::add_camera(&app, &name, &url)?;
    if let Some(w) = app.get_webview_window("camera") {
        let _ = w.close();
    }
    Ok(())
}

fn main() {
    let paths = match paths::Paths::from_env() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("retrace: {e:#}");
            std::process::exit(1);
        }
    };
    if let Err(e) = paths.create() {
        eprintln!("retrace: cannot create {}: {e}", paths.root.display());
        std::process::exit(1);
    }
    let _log_guard = logging::init(&paths.logs);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        root = %paths.root.display(),
        "Retrace Desktop starting"
    );
    // Every ffmpeg/ffprobe/yt-dlp the engine runs comes from <root>/bin first.
    retrace::tools::set_tools_dir(&paths.bin);

    let settings = settings::Settings::load(&paths.settings);
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    let args: Vec<String> = std::env::args().collect();
    let opens = parse_open_args(&args, &cwd);
    let ctx = Ctx::new(paths, settings);
    for p in opens {
        ctx.open_file(p);
    }

    let app = tauri::Builder::default()
        // Must be the first plugin: a second launch hands its args here and exits.
        .plugin(tauri_plugin_single_instance::init(|app, argv, cwd| {
            let ctx = app.state::<Ctx>();
            for p in parse_open_args(&argv, Path::new(&cwd)) {
                ctx.open_file(p);
            }
            if let Some(w) = app.get_webview_window("setup") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(ctx)
        .menu(menu::build)
        .on_menu_event(menu::handle)
        .invoke_handler(tauri::generate_handler![
            setup_state,
            retry_setup,
            use_system_tools,
            add_camera
        ])
        .on_window_event(|window, event| {
            if let WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) = event {
                let ctx = window.state::<Ctx>();
                for p in paths {
                    ctx.open_file(p.clone());
                }
            }
        })
        .setup(|app| {
            build_main_window(app.handle())?;
            watch_signals(app.handle().clone());
            setup::start(app.handle());
            Ok(())
        })
        .build(tauri::generate_context!());
    let app = match app {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("cannot start: {e}");
            eprintln!("retrace: {e}");
            std::process::exit(1);
        }
    };
    app.run(|app, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => cleanup(app),
        #[cfg(target_os = "macos")]
        RunEvent::Opened { urls } => {
            let ctx = app.state::<Ctx>();
            for u in urls {
                if let Ok(p) = u.to_file_path() {
                    ctx.open_file(p);
                }
            }
        }
        _ => {}
    });
}

fn build_main_window(app: &AppHandle) -> tauri::Result<()> {
    let port = app.state::<Ctx>().port.clone();
    let port2 = port.clone();
    let opener = app.clone();
    let root = app.state::<Ctx>().paths.root.clone();
    WebviewWindowBuilder::new(app, "setup", WebviewUrl::App("setup.html".into()))
        .title("Retrace")
        .inner_size(1280.0, 840.0)
        .min_inner_size(860.0, 560.0)
        .on_navigation(
            move |url| match nav::classify(url, port.load(Ordering::SeqCst)) {
                nav::Nav::Allow => true,
                nav::Nav::External => {
                    let _ = opener.opener().open_url(url.as_str(), None::<&str>);
                    false
                }
                nav::Nav::Deny => {
                    tracing::warn!("blocked navigation to {url}");
                    false
                }
            },
        )
        .on_download(move |_webview, event| match event {
            DownloadEvent::Requested { url, destination } => {
                if !nav::download_allowed(&url, port2.load(Ordering::SeqCst)) {
                    tracing::warn!("blocked download from {url}");
                    return false;
                }
                *destination = nav::download_target(destination, &url, &root);
                tracing::info!("download {url} -> {}", destination.display());
                true
            }
            DownloadEvent::Finished { url, path, success } => {
                tracing::info!(success, ?path, "download finished: {url}");
                true
            }
            _ => true,
        })
        .build()?;
    Ok(())
}

/// SIGTERM / SIGINT: clean up like a normal quit, then exit.
fn watch_signals(app: AppHandle) {
    #[cfg(unix)]
    tauri::async_runtime::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let (Ok(mut term), Ok(mut int)) = (
            signal(SignalKind::terminate()),
            signal(SignalKind::interrupt()),
        ) else {
            return;
        };
        tokio::select! {
            _ = term.recv() => {},
            _ = int.recv() => {},
        }
        tracing::info!("signal received, quitting");
        let a = app.clone();
        let _ = tauri::async_runtime::spawn_blocking(move || cleanup(&a)).await;
        app.exit(0);
    });
    #[cfg(not(unix))]
    let _ = app;
}

/// Quit without orphans: stop ingest, wait for the threads (≤3 s) so their own teardown runs,
/// kill registered helpers, stop the server, then end any child process that is still left.
fn cleanup(app: &AppHandle) {
    let ctx = app.state::<Ctx>();
    if ctx.cleaned_up.swap(true, Ordering::SeqCst) {
        return;
    }
    tracing::info!("quitting: stopping ingest");
    ctx.stop.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(3);
    let threads = std::mem::take(&mut *ctx.threads.lock().unwrap());
    for (name, done) in threads {
        let left = deadline.saturating_duration_since(Instant::now());
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = done.recv_timeout(left) {
            tracing::warn!("{name} did not stop in time");
        }
    }
    retrace::ingest::kill_children();
    if let Some(e) = ctx.engine.lock().unwrap().as_ref() {
        e.shutdown.trigger();
    }
    kill_child_processes();
    tracing::info!("quit complete");
}

/// Ends every direct child of this process (ffmpeg pipes, yt-dlp) that is still running:
/// SIGTERM, up to 1 s grace, then SIGKILL.
fn kill_child_processes() {
    #[cfg(unix)]
    {
        let pid = std::process::id().to_string();
        let has_children = || {
            std::process::Command::new("pgrep")
                .args(["-P", &pid])
                .stdout(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        if !has_children() {
            return;
        }
        let _ = std::process::Command::new("pkill")
            .args(["-TERM", "-P", &pid])
            .status();
        let until = Instant::now() + Duration::from_secs(1);
        while has_children() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(100));
        }
        if has_children() {
            tracing::warn!("child processes left after SIGTERM; sending SIGKILL");
            let _ = std::process::Command::new("pkill")
                .args(["-KILL", "-P", &pid])
                .status();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn open_args() {
        let cwd = Path::new("/work");
        assert_eq!(
            parse_open_args(
                &s(&["Retrace", "--open", "a.mp4", "--open=/v/b.avi", "-psn_0_1"]),
                cwd
            ),
            vec![PathBuf::from("/work/a.mp4"), PathBuf::from("/v/b.avi")]
        );
        assert!(parse_open_args(&s(&["Retrace", "--open"]), cwd).is_empty());
        assert!(parse_open_args(&s(&["Retrace"]), cwd).is_empty());
    }
}
