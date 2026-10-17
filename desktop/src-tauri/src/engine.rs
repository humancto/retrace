//! Builds the engine `AppState`, binds it on 127.0.0.1:0, and moves the window to it.

use crate::ctx::{Ctx, Engine};
use anyhow::{Context, Result};
use retrace::api::AppState;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};

/// Loads the index and the embedder (blocking work on a blocking thread), serves, navigates.
pub async fn start(app: &AppHandle) -> Result<()> {
    let ctx = app.state::<Ctx>();
    if ctx.engine.lock().unwrap().is_some() {
        return Ok(());
    }
    let paths = ctx.paths.clone();
    let settings = ctx.settings.lock().unwrap().clone();
    let embedder = settings.embedder.clone();
    let state = tauri::async_runtime::spawn_blocking(move || -> Result<AppState> {
        let db_path = paths.db_for(&settings.embedder);
        let crops = paths.crops_for(&settings.embedder);
        std::fs::create_dir_all(&crops)?;
        let db = retrace::db::Db::open(&db_path)
            .with_context(|| format!("open the index {}", db_path.display()))?;
        let emb = retrace::models::Embedder::load(&settings.models(&paths.models))
            .context("load the embedder")?;
        db.ensure_vec_table(emb.dim, &emb.name)?;
        Ok(AppState {
            db: Mutex::new(db),
            emb: Mutex::new(emb),
            crops_dir: crops,
            rec_dir: paths.rec.clone(),
            mediamtx: None,
            started: std::time::Instant::now(),
            live: retrace::overlay::LiveFrames::default(),
            desktop: true,
        })
    })
    .await
    .map_err(|e| anyhow::anyhow!("engine start stopped: {e}"))??;
    let state = Arc::new(state);

    let (addr, handle) = retrace::api::bind(state.clone(), "127.0.0.1:0")?;
    let shutdown = handle.shutdown();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = handle.await {
            tracing::error!("engine server stopped: {e:#}");
        }
    });
    let url = format!("http://{addr}/");
    tracing::info!("engine listening on {url}");
    ctx.port.store(addr.port(), Ordering::SeqCst);
    *ctx.engine.lock().unwrap() = Some(Engine {
        state: state.clone(),
        shutdown,
        embedder,
    });
    crate::worker::start(app, state);
    crate::setup::update(app, |v| v.phase = "Ready.".into());
    if let Some(w) = app.get_webview_window("setup") {
        w.navigate(tauri::Url::parse(&url)?)?;
    }
    Ok(())
}
