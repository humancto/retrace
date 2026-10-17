//! The ingest worker: one file at a time (each ingest loads its own detector and embedder, so
//! parallel ingests could run out of memory). Live cameras get a thread each.

use crate::ctx::{Ctx, Job};
use crate::paths::{Paths, sanitize};
use crate::settings::Settings;
use retrace::api::AppState;
use retrace::ingest::{self, IngestOpts};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};
use std::time::Duration;
use tauri::{AppHandle, Manager};

pub fn opts(
    paths: &Paths,
    s: &Settings,
    embedder: &str,
    source: String,
    cam: String,
    live: bool,
    state: &AppState,
) -> IngestOpts {
    let mut s = s.clone();
    s.embedder = embedder.to_string();
    IngestOpts {
        source,
        cam,
        fps: s.fps,
        width: s.width,
        start_ms: None,
        max_frames: None,
        crops_dir: paths.crops_for(embedder),
        db_path: paths.db_for(embedder),
        models: s.models(&paths.models),
        min_crop_h: s.min_crop_h,
        stale_s: 2.0,
        max_track_s: 120.0,
        record: live,
        rec_dir: paths.rec.clone(),
        retain_hours: 48.0,
        live_frames: Some(state.live.clone()),
        audio: true,
    }
}

/// The camera name for a file: its stem, made safe.
pub fn cam_name(p: &Path) -> String {
    sanitize(p.file_stem().and_then(|s| s.to_str()).unwrap_or("video"))
}

/// Starts the file worker (once). Files queued before this point are processed now.
pub fn start(app: &AppHandle, state: Arc<AppState>) {
    let ctx = app.state::<Ctx>();
    let Some(rx) = ctx.queue_rx.lock().unwrap().take() else {
        return;
    };
    let done = ctx.track_thread("ingest-worker");
    let app = app.clone();
    let spawned = std::thread::Builder::new()
        .name("ingest-worker".into())
        .spawn(move || {
            let _done = done;
            let ctx = app.state::<Ctx>();
            while !ctx.stop.load(Ordering::Relaxed) {
                let job = match rx.recv_timeout(Duration::from_millis(300)) {
                    Ok(j) => j,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => break,
                };
                index_file(&app, &ctx, &state, &job);
            }
            tracing::info!("ingest worker stopped");
        });
    if let Err(e) = spawned {
        tracing::error!("cannot start the ingest worker: {e}");
    }
}

/// The main window title: the only progress the window shows while a file is indexed.
fn set_title(app: &AppHandle, title: &str) {
    if let Some(w) = app.get_webview_window("setup") {
        let _ = w.set_title(title);
    }
}

fn index_file(app: &AppHandle, ctx: &Ctx, state: &AppState, job: &Job) {
    let path = job.path.as_path();
    let embedder = match ctx.engine.lock().unwrap().as_ref() {
        Some(e) => e.embedder.clone(),
        None => return,
    };
    let settings = ctx.settings.lock().unwrap().clone();
    let src = path.to_string_lossy().into_owned();
    let file = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| src.clone());
    set_title(app, &format!("Indexing {file}\u{2026}"));
    let source = match ingest::prepare_source(&src, &ctx.paths.cache) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot open {src}: {e:#}");
            set_title(app, &format!("Retrace \u{2014} cannot open {file}"));
            return;
        }
    };
    let o = opts(
        &ctx.paths,
        &settings,
        &embedder,
        source,
        job.cam.clone().unwrap_or_else(|| cam_name(path)),
        false,
        state,
    );
    tracing::info!(cam = %o.cam, "indexing {src}");
    let result = ingest::run_ingest(&o, Some(ctx.stop.clone()));
    // A quit stops the ingest early and it still returns Ok: only the flag tells the two apart.
    if ctx.stop.load(Ordering::SeqCst) {
        match result {
            Ok(stats) => tracing::warn!(
                cam = %o.cam,
                tracklets = stats.tracklets,
                frames = stats.frames,
                "indexing {src} stopped before the end (quit); the file is not done"
            ),
            Err(e) => tracing::warn!("indexing {src} stopped before the end (quit): {e:#}"),
        }
        return;
    }
    match result {
        Ok(stats) => {
            tracing::info!(
                cam = %o.cam,
                tracklets = stats.tracklets,
                frames = stats.frames,
                "indexed {src}"
            );
            set_title(
                app,
                &format!(
                    "Retrace \u{2014} {} tracklets indexed from {file}",
                    stats.tracklets
                ),
            );
        }
        Err(e) => {
            tracing::error!("indexing {src} failed: {e:#}");
            set_title(app, &format!("Retrace \u{2014} indexing {file} failed"));
        }
    }
}

/// Where a prepared camera source goes.
#[derive(Debug, PartialEq, Eq)]
pub enum Route {
    /// A stream: `run_live_forever` on its own thread.
    Live,
    /// A finished video (a downloaded YouTube video, a local file or a video file URL): the file
    /// queue indexes it once. `run_live_forever` would index it again and again.
    File,
}

const VIDEO_EXT: &[&str] = &[".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"];

/// Decides on the PREPARED source (after `prepare_source`), as the CLI does. A YouTube video that
/// is not live comes back as a local file; `is_live` is false for it.
pub fn route(prepared: &str) -> Route {
    if !ingest::is_live(prepared) {
        return Route::File;
    }
    let lower = prepared.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        let path = lower.split(['?', '#']).next().unwrap_or_default();
        if VIDEO_EXT.iter().any(|e| path.ends_with(e)) {
            return Route::File;
        }
    }
    Route::Live
}

fn thread_name(cam: &str) -> String {
    format!("live-{cam}")
}

/// Checks the stream address, prepares it on a blocking thread (for YouTube this runs yt-dlp,
/// which uses the network and, for a video that is not live, downloads it), then starts a live
/// camera on its own thread, or queues a finished video for indexing.
pub async fn add_camera(app: &AppHandle, name: &str, url: &str) -> Result<Route, String> {
    let name = sanitize(name.trim());
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err("Type the stream address.".into());
    }
    if !ingest::is_live(&url) {
        return Err(
            "This is not a live stream address. Use rtsp://, srt://, an HLS .m3u8 URL or a YouTube Live URL."
                .into(),
        );
    }
    let (state, embedder, settings, paths, stop) = {
        let ctx = app.state::<Ctx>();
        if ctx.has_thread(&thread_name(&name)) {
            return Err(format!(
                "A camera with the name {name} runs already. Type a different name."
            ));
        }
        let engine = ctx.engine.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(e) = engine.as_ref() else {
            return Err("Wait until setup is complete.".into());
        };
        let settings = ctx
            .settings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        (
            e.state.clone(),
            e.embedder.clone(),
            settings,
            ctx.paths.clone(),
            ctx.stop.clone(),
        )
    };
    let cache = paths.cache.clone();
    let source = tauri::async_runtime::spawn_blocking(move || ingest::prepare_source(&url, &cache))
        .await
        .map_err(|e| format!("the stream check stopped: {e}"))?
        .map_err(|e| format!("{e:#}"))?;
    if stop.load(Ordering::SeqCst) {
        return Err("Retrace is quitting.".into());
    }
    let ctx = app.state::<Ctx>();
    match route(&source) {
        Route::File => {
            tracing::info!(cam = %name, "{source} is a video, not a stream: indexing it once");
            ctx.queue(Job {
                path: PathBuf::from(source),
                cam: Some(name),
            });
            Ok(Route::File)
        }
        Route::Live => {
            let tname = thread_name(&name);
            let Some(done) = ctx.track_unique_thread(&tname) else {
                return Err(format!(
                    "A camera with the name {name} runs already. Type a different name."
                ));
            };
            let o = opts(
                &paths,
                &settings,
                &embedder,
                source,
                name.clone(),
                true,
                &state,
            );
            tracing::info!(cam = %name, "starting live camera");
            std::thread::Builder::new()
                .name(tname)
                .spawn(move || {
                    let _done = done;
                    ingest::run_live_forever(o, stop);
                })
                .map(|_| Route::Live)
                .map_err(|e| e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam_names_are_safe() {
        assert_eq!(cam_name(Path::new("/v/vtest.avi")), "vtest");
        assert_eq!(cam_name(Path::new("/v/front door:1.mp4")), "front-door-1");
    }

    #[test]
    fn only_streams_reach_the_live_loop() {
        // What prepare_source returns for a YouTube video that is not live: a cache file.
        assert_eq!(
            route("/Users/a/Library/Application Support/x/cache/abc123.mp4"),
            Route::File
        );
        assert_eq!(route("https://example.com/clips/a.mp4?sig=1"), Route::File);
        assert_eq!(route("http://example.com/a.MOV"), Route::File);
        assert_eq!(route("rtsp://192.168.1.20:554/stream1"), Route::Live);
        assert_eq!(route("srt://host:9000"), Route::Live);
        assert_eq!(route("https://example.com/live/index.m3u8"), Route::Live);
        // A YouTube URL is still a URL after prepare_source only when it is live.
        assert_eq!(
            route("https://www.youtube.com/watch?v=abcdefghijk"),
            Route::Live
        );
    }
}
