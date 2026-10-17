//! First-run setup: tools, then model weights, with progress for setup.html, then the engine.
//!
//! The page asks for the whole state with the `setup_state` command when it loads, then gets the
//! same state in `setup-state` events, so it never misses an update that came before it listened.

use crate::ctx::Ctx;
use crate::settings::ToolSource;
use crate::tools;
use anyhow::{Context, Result};
use retrace::models::ModelFile;
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager};

/// Copies already-downloaded weights from this dir (same layout as `<root>/models`) after a size
/// and sha256 check, so repeated test runs do not download 1.5 GB again.
pub const ENV_MODELS_SEED: &str = "RETRACE_MODELS_SEED";
pub const EVENT: &str = "setup-state";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Item {
    pub label: String,
    pub kind: &'static str,
    pub size: u64,
    pub done: u64,
    /// waiting | active | done | failed
    pub state: &'static str,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct View {
    /// What runs now, in plain words.
    pub phase: String,
    pub items: Vec<Item>,
    pub failed: bool,
    pub error: Option<String>,
    pub warning: Option<String>,
    /// Installed tools exist, so the page can offer "Use the installed tools".
    pub system_tools_found: bool,
    pub tools: ToolSource,
}

/// Starts setup on a background task unless it already runs.
pub fn start(app: &AppHandle) {
    let ctx = app.state::<Ctx>();
    if ctx.setup_running.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let a = app.clone();
        let res = match tauri::async_runtime::spawn_blocking(move || prepare(&a)).await {
            Ok(r) => r,
            Err(e) => Err(anyhow::anyhow!("setup stopped: {e}")),
        };
        let res = match res {
            Ok(()) => crate::engine::start(&app).await,
            Err(e) => Err(e),
        };
        let ctx = app.state::<Ctx>();
        if let Err(e) = res {
            tracing::error!("setup failed: {e:#}");
            update(&app, |v| {
                v.failed = true;
                v.error = Some(format!("{e:#}"));
                v.phase = "Setup stopped.".into();
                for it in v.items.iter_mut().filter(|i| i.state == "active") {
                    it.state = "failed";
                }
            });
        }
        ctx.setup_running.store(false, Ordering::SeqCst);
    });
}

/// Changes the view and sends it to the page.
pub fn update(app: &AppHandle, f: impl FnOnce(&mut View)) {
    let ctx = app.state::<Ctx>();
    let snapshot = {
        let mut v = ctx.setup.lock().unwrap();
        f(&mut v);
        v.clone()
    };
    let _ = app.emit_to("setup", EVENT, snapshot);
}

fn phase(app: &AppHandle, text: &str) {
    tracing::info!("setup: {text}");
    update(app, |v| v.phase = text.into());
}

/// Tools and weights. Blocking: runs on a blocking thread.
fn prepare(app: &AppHandle) -> Result<()> {
    let ctx = app.state::<Ctx>();
    let paths = ctx.paths.clone();
    let mut settings = ctx.settings.lock().unwrap().clone();
    paths.create()?;
    tools::write_notices(&paths.root, &paths.bin)?;

    let mut mode = settings.effective_tools();
    // Empty on platforms without pinned builds (the const differs per target).
    #[allow(clippy::const_is_empty)]
    if tools::ASSETS.is_empty() {
        mode = ToolSource::System;
    }
    let path_env = std::env::var_os("PATH");
    let system = tools::find_system(path_env.as_deref());
    update(app, |v| {
        *v = View {
            phase: "Checking what is missing.".into(),
            system_tools_found: system.is_some(),
            tools: mode,
            ..View::default()
        }
    });

    let tool_plan = match mode {
        ToolSource::Download => tools::plan(&paths.bin, tools::ASSETS),
        ToolSource::System => {
            let st = system.clone().context(
                "Retrace did not find ffmpeg, ffprobe and yt-dlp on this computer. \
                 Install them, or select Download.",
            )?;
            tools::link_system(&paths.bin, &st)?;
            tracing::info!(?st, "using the installed tools");
            if settings.system_tools.as_ref() != Some(&st) {
                settings.system_tools = Some(st);
                settings.tools = ToolSource::System;
                let _ = settings.save(&paths.settings);
                *ctx.settings.lock().unwrap() = settings.clone();
            }
            vec![]
        }
    };

    let model_plan = retrace::models::fetch::plan(&settings.models(&paths.models))?;
    if let Some(seed) = std::env::var_os(ENV_MODELS_SEED).filter(|s| !s.is_empty()) {
        phase(app, "Copying model files from the seed folder.");
        seed_models(&model_plan, Path::new(&seed), &paths.models)?;
    }

    let tool_files: Vec<ModelFile> = tool_plan.iter().map(|(_, f)| f.clone()).collect();
    let mut items: Vec<Item> = tool_files.iter().map(|f| item(f, "tool")).collect();
    items.extend(model_plan.iter().map(|f| item(f, "model")));
    let total: u64 = items.iter().map(|i| i.size).sum();
    let missing: u64 = items.iter().map(|i| i.size - i.done).sum();
    tracing::info!(
        files = items.len(),
        total_bytes = total,
        missing_bytes = missing,
        "setup plan"
    );
    update(app, |v| v.items = items);

    // Tools first: they are small, so progress shows in seconds.
    let mut tool_err = None;
    if !tool_files.is_empty() {
        phase(app, "Downloading the video tools.");
        let r = retrace::models::fetch::prefetch_files(&tool_files, progress_cb(app, 0)).and_then(
            |_| {
                for (asset, file) in &tool_plan {
                    tools::install(&paths.bin, asset, &file.dest)?;
                }
                Ok(())
            },
        );
        if let Err(e) = r {
            tool_err = Some(e);
        }
    }
    match tool_err {
        None => check_tools(app, &paths.bin),
        Some(e) => {
            // Without tools, search still works on an existing index: go on if weights are present.
            if model_plan.iter().all(ModelFile::is_present) {
                tracing::warn!("tool download failed, going on without tools: {e:#}");
                update(app, |v| {
                    v.warning = Some(format!(
                        "The video tools did not download, so you cannot add video now. \
                         Search still works. Details: {e:#}"
                    ))
                });
            } else {
                return Err(e.context("download the video tools"));
            }
        }
    }

    if model_plan.iter().any(|f| !f.is_present()) {
        phase(app, "Downloading the AI models.");
    }
    retrace::models::fetch::prefetch_files(&model_plan, progress_cb(app, tool_files.len()))
        .context("download the AI models")?;
    phase(app, "Starting the search engine.");
    Ok(())
}

fn item(f: &ModelFile, kind: &'static str) -> Item {
    let present = f.is_present();
    Item {
        label: f.label.clone(),
        kind,
        size: f.size,
        done: if present { f.size } else { 0 },
        state: if present { "done" } else { "waiting" },
    }
}

/// Runs the tools once: proves they execute on this machine and that ffmpeg has libx264.
fn check_tools(app: &AppHandle, bin: &Path) {
    match tools::check_ffmpeg(&bin.join("ffmpeg")) {
        Ok(true) => tracing::info!("ffmpeg has libx264"),
        Ok(false) => update(app, |v| {
            v.warning = Some(
                "This ffmpeg has no libx264 encoder, so clip export does not work. \
                 Search and indexing work."
                    .into(),
            )
        }),
        Err(e) => update(app, |v| {
            v.warning = Some(format!("ffmpeg does not run: {e:#}"));
        }),
    }
    for (tool, arg) in [("ffprobe", "-version"), ("yt-dlp", "--version")] {
        match tools::version_line(&bin.join(tool), arg) {
            Ok(l) => tracing::info!("{tool}: {l}"),
            Err(e) => tracing::warn!("{tool} does not run: {e:#}"),
        }
    }
}

/// A progress callback for `prefetch_files` whose items start at `offset` in the view. It sends
/// at most ~8 updates a second and logs each file at every 10 %.
fn progress_cb(app: &AppHandle, offset: usize) -> impl FnMut(&retrace::models::Progress) + use<'_> {
    let mut last_emit = Instant::now() - Duration::from_secs(1);
    let mut last_logged: (usize, u64) = (usize::MAX, 0);
    move |p| {
        let i = offset + p.index;
        let complete = p.file_done >= p.file.size;
        let decile = if p.file.size == 0 {
            10
        } else {
            p.file_done * 10 / p.file.size
        };
        if last_logged.0 != i || decile > last_logged.1 {
            tracing::info!(
                file = %p.file.label,
                done = p.file_done,
                size = p.file.size,
                "download progress {}%",
                decile * 10
            );
            last_logged = (i, decile);
        }
        if complete || p.file_done == 0 || last_emit.elapsed() >= Duration::from_millis(120) {
            last_emit = Instant::now();
            update(app, |v| {
                if let Some(it) = v.items.get_mut(i) {
                    it.done = p.file_done;
                    it.state = if complete { "done" } else { "active" };
                }
            });
        }
    }
}

/// For each missing file of `plan`, takes `<seed>/<path relative to models_dir>` when it has the
/// right size and sha256: hard link, else copy. Other files are left for the download.
pub fn seed_models(plan: &[ModelFile], seed: &Path, models_dir: &Path) -> Result<usize> {
    let mut n = 0;
    for f in plan.iter().filter(|f| !f.is_present()) {
        let Ok(rel) = f.dest.strip_prefix(models_dir) else {
            continue;
        };
        let src = seed.join(rel);
        let ok_size = std::fs::metadata(&src)
            .map(|m| m.len() == f.size)
            .unwrap_or(false);
        if !ok_size {
            continue;
        }
        if let Some(want) = f.sha256 {
            let got = retrace::models::fetch::sha256_file(&src)?;
            if !got.eq_ignore_ascii_case(want) {
                tracing::warn!(file = %f.label, "seed file has the wrong sha256; skipping");
                continue;
            }
        }
        if let Some(d) = f.dest.parent() {
            std::fs::create_dir_all(d)?;
        }
        if std::fs::hard_link(&src, &f.dest).is_err() {
            let tmp = f.part_path();
            std::fs::copy(&src, &tmp)?;
            std::fs::rename(&tmp, &f.dest)?;
        }
        tracing::info!(file = %f.label, "seeded from {}", src.display());
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mf(models: &Path, rel: &str, body: &[u8], sha: Option<&'static str>) -> ModelFile {
        ModelFile {
            label: rel.into(),
            url: "https://example.invalid/x".into(),
            dest: models.join(rel),
            size: body.len() as u64,
            sha256: sha,
        }
    }

    #[test]
    fn seed_checks_size_and_digest() {
        let t = tempfile::tempdir().unwrap();
        let (seed, models) = (t.path().join("seed"), t.path().join("models"));
        std::fs::create_dir_all(seed.join("deim")).unwrap();
        std::fs::write(seed.join("deim/a.onnx"), b"abc").unwrap();
        std::fs::write(seed.join("deim/b.onnx"), b"xyz").unwrap();
        std::fs::write(seed.join("deim/c.onnx"), b"toolong").unwrap();
        // sha256("abc")
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let plan = vec![
            mf(&models, "deim/a.onnx", b"abc", Some(abc)),
            mf(&models, "deim/b.onnx", b"xyz", Some(abc)), // wrong digest
            mf(&models, "deim/c.onnx", b"abc", None),      // wrong size
            mf(&models, "deim/d.onnx", b"abc", None),      // not in the seed
        ];
        assert_eq!(seed_models(&plan, &seed, &models).unwrap(), 1);
        assert!(plan[0].is_present());
        assert!(!plan[1].is_present() && !plan[2].is_present() && !plan[3].is_present());
        // Present files are skipped on the next run.
        assert_eq!(seed_models(&plan, &seed, &models).unwrap(), 0);
    }
}
