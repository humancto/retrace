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
use std::sync::PoisonError;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Clears `setup_running` if the setup task unwinds (a panic in `engine::start`), so Retry,
/// the tool buttons and the quit wait are never blocked for good. The normal end disarms it and
/// clears the flag itself.
struct Running<'a> {
    flag: &'a AtomicBool,
    armed: bool,
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.flag.store(false, Ordering::SeqCst);
        }
    }
}

/// Starts setup on a background task. When setup runs already, it runs once more after the
/// current run, so a choice made during the run (a tool source) is not lost.
pub fn start(app: &AppHandle) {
    let ctx = app.state::<Ctx>();
    if ctx.setup_running.swap(true, Ordering::SeqCst) {
        ctx.setup_again.store(true, Ordering::SeqCst);
        tracing::info!("setup runs already; it runs again when this run ends");
        update(app, |v| {
            v.phase = "Setup runs again with your choice when this step ends.".into()
        });
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let ctx = app.state::<Ctx>();
        loop {
            let mut running = Running {
                flag: &ctx.setup_running,
                armed: true,
            };
            run_once(&app).await;
            running.armed = false;
            ctx.setup_running.store(false, Ordering::SeqCst);
            if !ctx.setup_again.swap(false, Ordering::SeqCst) || ctx.stop.load(Ordering::SeqCst) {
                break;
            }
            // Another start may have begun in the gap; then that one owns the run.
            if ctx.setup_running.swap(true, Ordering::SeqCst) {
                break;
            }
            tracing::info!("running setup again with the new choice");
        }
    });
}

async fn run_once(app: &AppHandle) {
    let a = app.clone();
    let res = match tauri::async_runtime::spawn_blocking(move || prepare(&a)).await {
        Ok(r) => r,
        Err(e) => Err(anyhow::anyhow!("setup stopped: {e}")),
    };
    let res = match res {
        Ok(()) => match crate::engine::start(app).await {
            Ok(()) => Ok(()),
            Err(e) => Err(drop_damaged_models(app, e).await),
        },
        Err(e) => Err(e),
    };
    if let Err(e) = res {
        tracing::error!("setup failed: {e:#}");
        update(app, |v| {
            v.failed = true;
            v.error = Some(format!("{e:#}"));
            v.phase = "Setup stopped.".into();
            for it in v.items.iter_mut().filter(|i| i.state == "active") {
                it.state = "failed";
            }
        });
    }
}

/// After the engine did not start: a model file with the right size but damaged bytes counts as
/// present, so Retry would fail the same way again. Check every pinned digest, delete the files
/// that do not match, and say so in the error.
async fn drop_damaged_models(app: &AppHandle, e: anyhow::Error) -> anyhow::Error {
    let ctx = app.state::<Ctx>();
    let settings = ctx
        .settings
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let models = settings.models(&ctx.paths.models);
    phase(app, "Checking the model files.");
    let checked = tauri::async_runtime::spawn_blocking(move || {
        let plan = retrace::models::fetch::plan(&models)?;
        retrace::models::fetch::remove_damaged(&plan)
    })
    .await;
    match checked {
        Ok(Ok(removed)) if !removed.is_empty() => {
            tracing::warn!(?removed, "damaged model files deleted");
            e.context(format!(
                "{} model file(s) were damaged and are deleted ({}). Select Try again to download them again",
                removed.len(),
                removed.join(", ")
            ))
        }
        Ok(Ok(_)) => e,
        Ok(Err(c)) => {
            tracing::warn!("cannot check the model files: {c:#}");
            e
        }
        Err(j) => {
            tracing::warn!("model check stopped: {j}");
            e
        }
    }
}

/// Changes the view and sends it to the page.
pub fn update(app: &AppHandle, f: impl FnOnce(&mut View)) {
    let ctx = app.state::<Ctx>();
    let snapshot = {
        let mut v = ctx.setup.lock().unwrap_or_else(PoisonError::into_inner);
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
    let mut settings = ctx
        .settings
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
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
                // Record the paths only. `settings.tools` keeps the stored choice: with
                // RETRACE_SETUP=system the mode is an override for this run, not a new default
                // ("Use the installed tools" saves the choice itself).
                settings.system_tools = Some(st);
                let _ = settings.save(&paths.settings);
                *ctx.settings.lock().unwrap_or_else(PoisonError::into_inner) = settings.clone();
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
        let decile = (p.file_done * 10).checked_div(p.file.size).unwrap_or(10);
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
