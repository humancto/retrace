//! Builds the engine `AppState`, binds it on 127.0.0.1:0, and moves the window to it.

use crate::ctx::{Ctx, Engine};
use anyhow::{Context, Result};
use retrace::api::AppState;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{AppHandle, Manager};

/// Loads the index and the embedder (blocking work on a blocking thread), serves, navigates.
pub async fn start(app: &AppHandle) -> Result<()> {
    let ctx = app.state::<Ctx>();
    if ctx.stop.load(Ordering::SeqCst) {
        return Ok(());
    }
    if ctx
        .engine
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some()
    {
        // The engine runs already (a Retry after a failed navigate, or setup run again for a new
        // tool source): make sure the window shows it.
        crate::setup::update(app, |v| v.phase = "Ready.".into());
        return show_engine(app, ctx.port.load(Ordering::SeqCst));
    }
    let paths = ctx.paths.clone();
    let settings = ctx
        .settings
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
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
    if ctx.stop.load(Ordering::SeqCst) {
        // A quit came during the load. Free the embedder now (llama.cpp Metal buffers must not
        // outlive the C++ static destructors at exit) and do not serve.
        state
            .emb
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .release();
        tracing::info!("quit during engine start: embedder released");
        return Ok(());
    }
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
    *ctx.engine.lock().unwrap_or_else(PoisonError::into_inner) = Some(Engine {
        state: state.clone(),
        shutdown,
        embedder,
    });
    crate::worker::start(app, state);
    crate::setup::update(app, |v| v.phase = "Ready.".into());
    show_engine(app, addr.port())
}

/// Moves the main window to the engine UI on `port`, unless it shows it already.
fn show_engine(app: &AppHandle, port: u16) -> Result<()> {
    let url = tauri::Url::parse(&format!("http://127.0.0.1:{port}/"))?;
    if let Some(w) = app.get_webview_window("setup") {
        let on_engine = w
            .url()
            .map(|u| u.host_str() == Some("127.0.0.1") && u.port() == Some(port))
            .unwrap_or(false);
        if !on_engine {
            w.navigate(url)?;
        }
    }
    Ok(())
}
