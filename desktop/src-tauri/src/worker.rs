//! The ingest worker: one file at a time (each ingest loads its own detector and embedder, so
//! parallel ingests could run out of memory). Live cameras get a thread each.

use crate::ctx::Ctx;
use crate::paths::{Paths, sanitize};
use crate::settings::Settings;
use retrace::api::AppState;
use retrace::ingest::{self, IngestOpts};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
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
                let path = match rx.recv_timeout(Duration::from_millis(300)) {
                    Ok(p) => p,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => break,
                };
                index_file(&ctx, &state, &path);
            }
            tracing::info!("ingest worker stopped");
        });
    if let Err(e) = spawned {
        tracing::error!("cannot start the ingest worker: {e}");
    }
}

fn index_file(ctx: &Ctx, state: &AppState, path: &Path) {
    let embedder = match ctx.engine.lock().unwrap().as_ref() {
        Some(e) => e.embedder.clone(),
        None => return,
    };
    let settings = ctx.settings.lock().unwrap().clone();
    let src = path.to_string_lossy().into_owned();
    let source = match ingest::prepare_source(&src, &ctx.paths.cache) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot open {src}: {e:#}");
            return;
        }
    };
    let o = opts(
        &ctx.paths,
        &settings,
        &embedder,
        source,
        cam_name(path),
        false,
        state,
    );
    tracing::info!(cam = %o.cam, "indexing {src}");
    match ingest::run_ingest(&o, Some(ctx.stop.clone())) {
        Ok(stats) => tracing::info!(
            cam = %o.cam,
            tracklets = stats.tracklets,
            frames = stats.frames,
            "indexed {src}"
        ),
        Err(e) => tracing::error!("indexing {src} failed: {e:#}"),
    }
}

/// Starts a live camera on its own thread (runs until quit).
pub fn add_camera(app: &AppHandle, name: &str, url: &str) -> Result<(), String> {
    let name = sanitize(name.trim());
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err("Type the stream address.".into());
    }
    if !ingest::is_live(&url) {
        return Err(
            "This is not a live stream address. Use rtsp://, rtmp://, srt://, an HLS .m3u8 URL or a YouTube Live URL."
                .into(),
        );
    }
    let ctx = app.state::<Ctx>();
    let (state, embedder) = match ctx.engine.lock().unwrap().as_ref() {
        Some(e) => (e.state.clone(), e.embedder.clone()),
        None => return Err("Wait until setup is complete.".into()),
    };
    let settings = ctx.settings.lock().unwrap().clone();
    let source = ingest::prepare_source(&url, &ctx.paths.cache).map_err(|e| format!("{e:#}"))?;
    let o = opts(
        &ctx.paths,
        &settings,
        &embedder,
        source,
        name.clone(),
        true,
        &state,
    );
    let stop = ctx.stop.clone();
    let done = ctx.track_thread(&format!("live-{name}"));
    tracing::info!(cam = %name, "starting live camera");
    std::thread::Builder::new()
        .name(format!("live-{name}"))
        .spawn(move || {
            let _done = done;
            ingest::run_live_forever(o, stop);
        })
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam_names_are_safe() {
        assert_eq!(cam_name(Path::new("/v/vtest.avi")), "vtest");
        assert_eq!(cam_name(Path::new("/v/front door:1.mp4")), "front-door-1");
    }
}
