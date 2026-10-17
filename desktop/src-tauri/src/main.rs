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
use std::sync::PoisonError;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::webview::DownloadEvent;
use tauri::{
    AppHandle, DragDropEvent, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_opener::OpenerExt;

/// `--open <file>` (repeatable), `--open=<file>`, and bare arguments that name an existing file
/// (a Linux file association, `Retrace video.mp4`, or a second launch forwarded by the
/// single-instance plugin). Relative paths resolve against `cwd`. Other flags (macOS adds
/// `-psn_…`) and bare arguments that are not files are ignored.
pub fn parse_open_args(args: &[String], cwd: &Path) -> Vec<PathBuf> {
    let abs = |v: &str| {
        let p = PathBuf::from(v);
        if p.is_absolute() { p } else { cwd.join(p) }
    };
    let mut out = Vec::new();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == "--open" {
            if let Some(v) = it.next().filter(|v| !v.is_empty()) {
                out.push(abs(v));
            }
        } else if let Some(v) = a.strip_prefix("--open=") {
            if !v.is_empty() {
                out.push(abs(v));
            }
        } else if !a.is_empty() && !a.starts_with('-') {
            let p = abs(a);
            if p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

#[tauri::command]
fn setup_state(ctx: tauri::State<'_, Ctx>) -> setup::View {
    // The page calls this once when its script runs: proof that the page loaded under the CSP.
    tracing::info!("setup page connected");
    ctx.setup
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

#[tauri::command]
fn retry_setup(app: AppHandle) {
    setup::start(&app);
}

#[tauri::command]
fn use_system_tools(app: AppHandle) {
    set_tool_source(&app, settings::ToolSource::System);
}

/// The way back from "Use the installed tools", for when they were removed later.
#[tauri::command]
fn use_download_tools(app: AppHandle) {
    set_tool_source(&app, settings::ToolSource::Download);
}

/// Saves the choice, then runs setup with it: now, or as soon as the setup that runs now ends.
fn set_tool_source(app: &AppHandle, source: settings::ToolSource) {
    {
        let ctx = app.state::<Ctx>();
        let mut s = ctx.settings.lock().unwrap_or_else(PoisonError::into_inner);
        s.tools = source;
        if let Err(e) = s.save(&ctx.paths.settings) {
            tracing::warn!("cannot save settings: {e:#}");
        }
    }
    tracing::info!(?source, "tool source changed");
    setup::start(app);
}

/// Async, so Tauri runs it off the main thread: the stream check can run yt-dlp for seconds
/// (or download a whole video) and the window must not freeze.
#[tauri::command]
async fn add_camera(app: AppHandle, name: String, url: String) -> Result<(), String> {
    worker::add_camera(&app, &name, &url).await?;
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
    let log_guard = logging::init(&paths.logs);
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        root = %paths.root.display(),
        "Retrace Desktop starting"
    );
    // Every ffmpeg/ffprobe/yt-dlp the engine runs comes from <root>/bin first.
    retrace::tools::set_tools_dir(&paths.bin);
    // Each helper in its own process group, so a quit also ends the children yt-dlp starts.
    retrace::ingest::set_child_process_groups(true);

    let settings = settings::Settings::load(&paths.settings);
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    let args: Vec<String> = std::env::args().collect();
    let opens = parse_open_args(&args, &cwd);
    let ctx = Ctx::new(paths, settings);
    *ctx.log_guard.lock().unwrap_or_else(PoisonError::into_inner) = Some(log_guard);
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
            use_download_tools,
            add_camera
        ])
        .on_window_event(|window, event| match event {
            WindowEvent::DragDrop(DragDropEvent::Drop { paths, .. }) => {
                let ctx = window.state::<Ctx>();
                for p in paths {
                    ctx.open_file(p.clone());
                }
            }
            // Closing the main window quits the app, also while the Add Camera window is open:
            // otherwise the app keeps running with no way to show the main window again. The
            // window stays open until cleanup has moved it off the engine page.
            WindowEvent::CloseRequested { api, .. } if window.label() == "setup" => {
                api.prevent_close();
                request_quit(window.app_handle());
            }
            _ => {}
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
        // Hold the exit until cleanup has run on its own thread (it needs the main thread free to
        // move the window off the engine page), then `request_quit` exits for real.
        RunEvent::ExitRequested { api, .. } => {
            if !app.state::<Ctx>().quit_done.load(Ordering::SeqCst) {
                api.prevent_exit();
                request_quit(app);
            }
        }
        // Last resort if the exit could not be held; a no-op after a full quit.
        RunEvent::Exit => {
            if !cleanup(app) {
                exit_now();
            }
        }
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
        request_quit(&app);
    });
    #[cfg(not(unix))]
    let _ = app;
}

/// Runs `cleanup` on a new thread (once), then exits the app with code 0.
fn request_quit(app: &AppHandle) {
    if app.state::<Ctx>().quit_started.swap(true, Ordering::SeqCst) {
        return;
    }
    let a = app.clone();
    let spawned = std::thread::Builder::new()
        .name("quit".into())
        .spawn(move || {
            // Exit even if cleanup panics: otherwise quit_done stays false, every later
            // ExitRequested is held, and only a force kill ends the app.
            let clean = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cleanup(&a)))
            {
                Ok(clean) => clean,
                Err(_) => {
                    tracing::error!("cleanup panicked; exiting anyway");
                    false
                }
            };
            a.state::<Ctx>().quit_done.store(true, Ordering::SeqCst);
            if !clean {
                exit_now();
            }
            a.exit(0);
        });
    if let Err(e) = spawned {
        tracing::error!("cannot start the quit thread: {e}; cleaning up in place");
        if !cleanup(app) {
            exit_now();
        }
        app.exit(0);
    }
}

/// Ends the process without the C++ static destructors. Used when a thread that may own
/// llama.cpp Metal buffers (an engine load, a gemma ingest or live thread) did not stop in time:
/// a normal exit would run `ggml_metal_device_free` under it, which asserts and aborts (exit 134).
/// The log is flushed before this runs; child processes are already ended.
fn exit_now() -> ! {
    #[cfg(unix)]
    // SAFETY: `_exit` takes no pointers and does not return; skipping atexit handlers and static
    // destructors is the purpose.
    unsafe {
        libc::_exit(0)
    }
    #[cfg(not(unix))]
    std::process::exit(0)
}

/// How long a quit waits for a running setup (a model download or an engine load, which can
/// load EmbeddingGemma for several seconds) to notice the stop flag.
const SETUP_BUDGET: Duration = Duration::from_secs(10);
/// How long a quit waits for each worker or live thread, after the setup wait.
const THREAD_BUDGET: Duration = Duration::from_secs(3);

/// Quit without orphans: stop ingest, wait for setup and then for each thread (each with its own
/// budget) so their own teardown runs, kill registered helpers, stop the server, then end any
/// child process that is still left. Returns false when something did not stop in time; the
/// caller then exits with [`exit_now`]. A second call (the `Exit` event after a full quit) is a
/// no-op that returns true.
fn cleanup(app: &AppHandle) -> bool {
    let ctx = app.state::<Ctx>();
    if ctx.cleaned_up.swap(true, Ordering::SeqCst) {
        return true;
    }
    let mut clean = true;
    tracing::info!("quitting: stopping ingest");
    ctx.stop.store(true, Ordering::SeqCst);
    // The engine UI keeps an EventSource (/api/live) open, and a graceful server stop waits for
    // every connection. Leave the page so the server can stop and release its engine state.
    // Never close the window here: closing the last window requests exit again.
    if let Some(w) = app.get_webview_window("setup")
        && let Ok(blank) = tauri::Url::parse("about:blank")
    {
        let _ = w.navigate(blank);
    }
    // An engine start that is still loading the embedder checks `stop` when the load returns
    // and frees it itself; give it its own budget so it does not race the exit.
    let until = Instant::now() + SETUP_BUDGET;
    while ctx.setup_running.load(Ordering::SeqCst) && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(50));
    }
    if ctx.setup_running.load(Ordering::SeqCst) {
        tracing::warn!("setup did not stop in time");
        clean = false;
    }
    let threads = std::mem::take(&mut *ctx.threads.lock().unwrap_or_else(PoisonError::into_inner));
    for (name, done) in threads {
        if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) = done.recv_timeout(THREAD_BUDGET) {
            tracing::warn!("{name} did not stop in time");
            clean = false;
        }
    }
    retrace::ingest::kill_children();
    let engine = ctx
        .engine
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(e) = engine {
        e.shutdown.trigger();
        release_engine(e);
    }
    kill_child_processes();
    ctx.quit_done.store(true, Ordering::SeqCst);
    if clean {
        tracing::info!("quit complete");
    } else {
        tracing::warn!("quit complete; exiting without static destructors");
    }
    // `App::run` exits the process, so drop the log guard here to flush the last lines.
    drop(
        ctx.log_guard
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take(),
    );
    clean
}

/// Frees the engine (embedder, then index) before the process exits. The EmbeddingGemma embedder
/// owns llama.cpp Metal buffers; if they are still alive when the C++ static destructors run at
/// exit, `ggml_metal_device_free` asserts and the process aborts. The server's router and open
/// connections (the UI's EventSource) can still hold the state, so free the embedder through its
/// lock instead of waiting for the last holder.
fn release_engine(e: ctx::Engine) {
    e.state
        .emb
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .release();
    tracing::info!("embedder released");
    drop(e);
}

/// Process names of the helpers the engine starts. pgrep/pkill AND this pattern with `-P`, so
/// other children (WebKitGTK's web and network processes on Linux) are left alone.
#[cfg(unix)]
const HELPERS: &str = "^(ffmpeg|ffprobe|yt-dlp)";

/// Ends every direct child of this process that is an engine helper (ffmpeg pipes, yt-dlp) and
/// is still running: SIGTERM, up to 1 s grace, then SIGKILL.
fn kill_child_processes() {
    #[cfg(unix)]
    {
        let pid = std::process::id().to_string();
        let has_children = || {
            std::process::Command::new("pgrep")
                .args(["-P", &pid, HELPERS])
                .stdout(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        if !has_children() {
            return;
        }
        let _ = std::process::Command::new("pkill")
            .args(["-TERM", "-P", &pid, HELPERS])
            .status();
        let until = Instant::now() + Duration::from_secs(1);
        while has_children() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(100));
        }
        if has_children() {
            tracing::warn!("child processes left after SIGTERM; sending SIGKILL");
            let _ = std::process::Command::new("pkill")
                .args(["-KILL", "-P", &pid, HELPERS])
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

    #[test]
    fn bare_file_args() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join("v.mp4"), b"x").unwrap();
        let abs = t.path().join("v.mp4").to_string_lossy().into_owned();
        assert_eq!(
            parse_open_args(
                &s(&[
                    "Retrace",
                    "v.mp4",
                    &abs,
                    "missing.mp4",
                    "-psn_0_1",
                    "--flag"
                ]),
                t.path()
            ),
            vec![t.path().join("v.mp4"), t.path().join("v.mp4")]
        );
    }
}
