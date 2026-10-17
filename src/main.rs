//! The `retrace` command line: a thin wrapper over the `retrace` library.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use retrace::{annotate, api, db, ingest, models, overlay, search};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, atomic::AtomicBool};

#[derive(Parser)]
#[command(
    name = "retrace",
    version,
    about = "Retrace: ask your cameras what happened, then retrace their steps. Fully local."
)]
struct Cli {
    #[command(flatten)]
    common: Common,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// SQLite index file.
    #[arg(long, global = true, default_value = "data/index.db")]
    db: PathBuf,
    /// Directory for crops and context frames.
    #[arg(long, global = true, default_value = "data/crops")]
    crops: PathBuf,
    /// Directory holding pre-downloaded model files (optional; otherwise models download on first use).
    #[arg(long, global = true, env = "RETRACE_MODELS")]
    models_dir: Option<PathBuf>,
    /// Detector: deim-dfine-n | deim-dfine-s | deim-dfine-m | <path-or-URL>.onnx
    #[arg(long, global = true, default_value = "deim-dfine-s")]
    detector: String,
    /// Embedder: clip-b16 (GitHub-hosted, MIT) | siglip2-b16-256 | siglip2-b16-384 | siglip2-so400m-384 (Hugging Face, Apache-2.0) | gemma2 (EmbeddingGemma 2 via the sidecar: text+image+audio)
    #[arg(long, global = true, default_value = "clip-b16")]
    embedder: String,
    /// Execution provider: cpu | coreml (needs the `coreml` cargo feature)
    #[arg(long, global = true, default_value = "cpu")]
    device: String,
    /// onnxruntime intra-op threads (0 = library default)
    #[arg(long, global = true, default_value_t = 0)]
    threads: usize,
    /// Detection confidence threshold
    #[arg(long, global = true, default_value_t = 0.35)]
    conf: f32,
    /// Record live sources as 10 s MP4 segments for clip playback (codec copy, no re-encode).
    #[arg(long, global = true, default_value_t = true, action = clap::ArgAction::Set)]
    record: bool,
    /// Directory for recorded segments.
    #[arg(long, global = true, default_value = "data/rec")]
    rec: PathBuf,
    /// Hours of recordings to keep.
    #[arg(long, global = true, default_value_t = 48.0)]
    retain_hours: f64,
    /// Embedding sidecar URL for `--embedder gemma2` (sidecar/embed_server.py).
    #[arg(
        long,
        global = true,
        default_value = "http://127.0.0.1:8082",
        env = "RETRACE_EMBED_URL"
    )]
    embed_url: String,
    /// Embed audio windows for sources with sound (needs an embedder with an audio encoder, i.e. gemma2).
    #[arg(long, global = true, default_value_t = true, action = clap::ArgAction::Set)]
    audio: bool,
}

impl Common {
    fn models(&self) -> models::ModelSettings {
        models::ModelSettings {
            models_dir: self.models_dir.clone(),
            detector: self.detector.clone(),
            embedder: self.embedder.clone(),
            device: self.device.clone(),
            conf: self.conf,
            threads: self.threads,
            embed_url: self.embed_url.clone(),
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Index a video file (replay) or a live source (rtsp://, http://) into the database.
    Ingest {
        /// File path, stream URL (rtsp://, http(s):// HLS) or a YouTube URL (resolved with yt-dlp)
        #[arg(long)]
        source: String,
        /// Camera name
        #[arg(long)]
        cam: String,
        /// Frames per second to analyze
        #[arg(long, default_value_t = 4.0)]
        fps: f32,
        /// Analysis width in pixels (height keeps aspect)
        #[arg(long, default_value_t = 640)]
        width: u32,
        /// Wall-clock start of a file source (epoch ms or RFC3339). Default: now minus duration.
        #[arg(long)]
        start: Option<String>,
        /// Stop after N frames (testing)
        #[arg(long)]
        max_frames: Option<usize>,
        /// Minimum crop height in pixels to embed
        #[arg(long, default_value_t = 40)]
        min_crop_h: u32,
    },
    /// Serve the API and UI; optionally ingest sources in the same process (--source name=url, repeatable).
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
        /// MediaMTX playback base URL for live clips, e.g. http://127.0.0.1:9996
        #[arg(long)]
        mediamtx: Option<String>,
        /// Sources to ingest while serving: name=path_or_url (repeatable)
        #[arg(long = "source")]
        sources: Vec<String>,
        #[arg(long, default_value_t = 4.0)]
        fps: f32,
        #[arg(long, default_value_t = 640)]
        width: u32,
    },
    /// Search from the command line.
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long)]
        cam: Option<String>,
        #[arg(long)]
        class: Option<String>,
    },
    /// Caption tracklets with a local VLM (OpenAI-compatible endpoint such as llama-server).
    Annotate {
        #[arg(long, default_value = "http://127.0.0.1:8081/v1")]
        endpoint: String,
        #[arg(long, default_value = "default")]
        model: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        /// Process the backlog once and exit (default: keep polling)
        #[arg(long)]
        once: bool,
        #[arg(long, default_value_t = 120)]
        timeout_s: u64,
    },
    /// Evaluate recall@k and latency on a golden query set (JSON: [{"q": "...", "hits": [tracklet ids]}]).
    Eval {
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 5)]
        k: usize,
    },
}

fn parse_start(s: &str) -> Result<i64> {
    if let Ok(ms) = s.parse::<i64>() {
        return Ok(ms);
    }
    let dt = chrono::DateTime::parse_from_rfc3339(s)
        .with_context(|| format!("bad --start '{s}' (epoch ms or RFC3339)"))?;
    Ok(dt.timestamp_millis())
}

fn ingest_opts(
    c: &Common,
    source: &str,
    cam: &str,
    fps: f32,
    width: u32,
    start_ms: Option<i64>,
    max_frames: Option<usize>,
    min_crop_h: u32,
    live_frames: Option<overlay::LiveFrames>,
) -> ingest::IngestOpts {
    ingest::IngestOpts {
        source: source.into(),
        cam: cam.into(),
        fps,
        width,
        start_ms,
        max_frames,
        crops_dir: c.crops.clone(),
        db_path: c.db.clone(),
        models: c.models(),
        min_crop_h,
        stale_s: 2.0,
        max_track_s: 120.0,
        record: c.record,
        rec_dir: c.rec.clone(),
        retain_hours: c.retain_hours,
        live_frames,
        audio: c.audio,
    }
}

/// YouTube videos are downloaded once (next to the index) and replayed as files; live URLs pass through.
fn prepare(c: &Common, source: &str) -> Result<String> {
    let cache =
        c.db.parent()
            .map(|p| p.join("cache"))
            .unwrap_or_else(|| PathBuf::from("data/cache"));
    ingest::prepare_source(source, &cache)
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,usls=warn".into()),
        )
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        // Logs go to stderr so stdout stays machine-readable (ingest stats JSON, eval lines).
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    if let Some(p) = cli.common.db.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::create_dir_all(&cli.common.crops)?;

    match cli.cmd {
        Cmd::Ingest {
            source,
            cam,
            fps,
            width,
            start,
            max_frames,
            min_crop_h,
        } => {
            let start_ms = start.as_deref().map(parse_start).transpose()?;
            let source = prepare(&cli.common, &source)?;
            let opts = ingest_opts(
                &cli.common,
                &source,
                &cam,
                fps,
                width,
                start_ms,
                max_frames,
                min_crop_h,
                None,
            );
            if ingest::is_live(&source) {
                let stop = Arc::new(AtomicBool::new(false));
                let rt = tokio::runtime::Runtime::new()?;
                let stop2 = stop.clone();
                rt.spawn(async move {
                    shutdown_signal().await;
                    stop2.store(true, std::sync::atomic::Ordering::Relaxed);
                    ingest::kill_children();
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    std::process::exit(0);
                });
                ingest::run_live_forever(opts, stop);
            } else {
                let stats = ingest::run_ingest(&opts, None)?;
                println!("{}", serde_json::to_string_pretty(&stats)?);
            }
        }
        Cmd::Serve {
            bind,
            mediamtx,
            sources,
            fps,
            width,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            let db = db::Db::open(&cli.common.db)?;
            let emb = models::Embedder::load(&cli.common.models())?;
            db.ensure_vec_table(emb.dim, &emb.name)?;
            let live = overlay::LiveFrames::default();
            let state = Arc::new(api::AppState {
                db: Mutex::new(db),
                emb: Mutex::new(emb),
                crops_dir: cli.common.crops.clone(),
                rec_dir: cli.common.rec.clone(),
                mediamtx,
                started: std::time::Instant::now(),
                live: live.clone(),
                desktop: false,
            });
            let stop = Arc::new(AtomicBool::new(false));
            for s in sources {
                let Some((name, src)) = s.split_once('=') else {
                    anyhow::bail!("--source expects name=path_or_url, got '{s}'")
                };
                let src = prepare(&cli.common, src)?;
                let opts = ingest_opts(
                    &cli.common,
                    &src,
                    name,
                    fps,
                    width,
                    None,
                    None,
                    40,
                    Some(live.clone()),
                );
                let stop = stop.clone();
                std::thread::Builder::new()
                    .name(format!("ingest-{name}"))
                    .spawn(move || {
                        if ingest::is_live(&opts.source) {
                            ingest::run_live_forever(opts, stop);
                        } else if let Err(e) = ingest::run_ingest(&opts, Some(stop)) {
                            tracing::error!("ingest failed: {e:#}");
                        }
                    })?;
            }
            // Ctrl-C / SIGTERM: stop ingest loops, take the recorders down with us, then exit.
            rt.spawn(async {
                shutdown_signal().await;
                tracing::info!("shutting down");
                ingest::kill_children();
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                std::process::exit(0);
            });
            rt.block_on(api::serve(state, &bind))?;
            ingest::kill_children();
        }
        Cmd::Search {
            query,
            k,
            cam,
            class,
        } => {
            let db = db::Db::open(&cli.common.db)?;
            let mut emb = models::Embedder::load(&cli.common.models())?;
            let p = search::SearchParams {
                q: query,
                cam,
                class,
                k: Some(k),
                ..Default::default()
            };
            let r = search::search(&db, &mut emb, &p)?;
            println!(
                "query: {}  (rewritten: {})  candidates={} threshold={:.3} total={:.1}ms",
                r.query,
                r.rewritten,
                r.candidates,
                r.threshold,
                r.timing_ms.get("total").cloned().unwrap_or(0.0)
            );
            for (i, h) in r.hits.iter().enumerate() {
                let t = &h.tracklet;
                println!(
                    "{:>2}. id={:<5} score={:.3} {:<8} cam={:<10} {:>5.1}s {:<16} {}",
                    i + 1,
                    t.id,
                    h.score,
                    t.class,
                    t.cam_name,
                    t.dwell_s,
                    t.direction,
                    t.caption_text.clone().unwrap_or_default()
                );
            }
        }
        Cmd::Annotate {
            endpoint,
            model,
            api_key,
            limit,
            once,
            timeout_s,
        } => {
            let rt = tokio::runtime::Runtime::new()?;
            let o = annotate::AnnotateOpts {
                endpoint,
                model,
                api_key,
                limit,
                once,
                timeout_s,
            };
            let n = rt.block_on(annotate::run_annotate(
                &cli.common.db,
                &cli.common.crops,
                &o,
            ))?;
            println!("captioned {n} tracklets");
        }
        Cmd::Eval { golden, k } => {
            #[derive(serde::Deserialize)]
            struct G {
                q: String,
                hits: Vec<i64>,
            }
            let set: Vec<G> = serde_json::from_str(&std::fs::read_to_string(&golden)?)?;
            let db = db::Db::open(&cli.common.db)?;
            let mut emb = models::Embedder::load(&cli.common.models())?;
            let mut recalls = Vec::new();
            let mut aps = Vec::new();
            let mut lat = Vec::new();
            for g in &set {
                let p = search::SearchParams {
                    q: g.q.clone(),
                    k: Some(k.max(20)),
                    mode: Some("all".into()),
                    ..Default::default()
                };
                let r = search::search(&db, &mut emb, &p)?;
                lat.push(r.timing_ms.get("total").cloned().unwrap_or(0.0));
                let top: Vec<i64> = r.hits.iter().take(k).map(|h| h.tracklet.id).collect();
                let rel: std::collections::HashSet<i64> = g.hits.iter().cloned().collect();
                let found = top.iter().filter(|id| rel.contains(id)).count();
                let recall = if rel.is_empty() {
                    0.0
                } else {
                    found as f64 / rel.len().min(k) as f64
                };
                let mut hitn = 0.0;
                let mut ap = 0.0;
                for (i, id) in r.hits.iter().map(|h| h.tracklet.id).enumerate() {
                    if rel.contains(&id) {
                        hitn += 1.0;
                        ap += hitn / (i as f64 + 1.0);
                    }
                }
                let ap = if rel.is_empty() {
                    0.0
                } else {
                    ap / rel.len() as f64
                };
                recalls.push(recall);
                aps.push(ap);
                println!(
                    "{:<50} recall@{k}={:.2} AP={:.2} top={:?}",
                    g.q.chars().take(50).collect::<String>(),
                    recall,
                    ap,
                    top
                );
            }
            lat.sort_by(|a, b| a.total_cmp(b));
            let p50 = lat.get(lat.len() / 2).cloned().unwrap_or(0.0);
            let p95 = lat
                .get((lat.len() as f64 * 0.95) as usize)
                .cloned()
                .unwrap_or(0.0);
            println!(
                "\nqueries={} recall@{k}={:.3} mAP={:.3} p50={:.1}ms p95={:.1}ms embedder={}",
                set.len(),
                recalls.iter().sum::<f64>() / set.len().max(1) as f64,
                aps.iter().sum::<f64>() / set.len().max(1) as f64,
                p50,
                p95,
                emb.name
            );
        }
    }
    Ok(())
}
