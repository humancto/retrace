//! Ingest: ffmpeg frame pipe → detector → tracker → tracklets → crops → embeddings → SQLite.
//! Works for files (replay, faster than real time) and live sources (RTSP/HLS/YouTube Live, with recorder + backpressure).

use crate::db::{Db, NewTracklet};
use crate::models::{Detector, Embedder, ModelSettings};
use crate::overlay::{self, LiveFrames};
use crate::rules::RuleEngine;
use crate::track::{Det, Tracked, Tracker, TrackerConfig};
use anyhow::{Context, Result};
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use image::RgbImage;
use serde::Serialize;
use std::collections::HashMap;

/// A live source with no new frame for this long is restarted (see the stall watchdog).
const STALL_SECS: i64 = 30;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use usls::Image;

#[derive(Debug, Clone)]
pub struct IngestOpts {
    pub source: String,
    pub cam: String,
    pub fps: f32,
    pub width: u32,
    pub start_ms: Option<i64>,
    pub max_frames: Option<usize>,
    pub crops_dir: PathBuf,
    pub db_path: PathBuf,
    pub models: ModelSettings,
    pub min_crop_h: u32,
    pub stale_s: f32,
    pub max_track_s: f32,
    /// Record live sources as 10 s MP4 segments (codec copy) so clips can be played back.
    pub record: bool,
    pub rec_dir: PathBuf,
    pub retain_hours: f64,
    /// When set, every analysed frame is annotated and published for the live preview.
    pub live_frames: Option<LiveFrames>,
    /// Embed audio windows when the source has audio and the embedder supports it.
    pub audio: bool,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct IngestStats {
    pub frames: usize,
    /// Frames read but not analysed because analysis could not keep up with a live source.
    pub skipped: usize,
    pub detections: usize,
    pub tracklets: usize,
    pub alerts: usize,
    pub elapsed_s: f64,
    pub video_s: f64,
}

#[derive(Serialize, Clone)]
struct CropRef {
    path: String,
    t: i64,
    bbox: [f32; 4],
    score: f32,
}

struct CropCand {
    img: RgbImage,
    t: i64,
    bbox: [f32; 4],
    score: f32,
    ctx: Option<RgbImage>,
}

struct Track {
    class_votes: HashMap<String, usize>,
    t0: i64,
    t1: i64,
    path: Vec<(i64, f32, f32, f32, f32)>,
    crops: Vec<CropCand>,
    last_frame: usize,
    n_frames: usize,
    /// Set once the track has been written provisionally (long dwell); later writes update in place.
    db_id: Option<i64>,
    /// Rules that already fired for this track, so a person triggers each rule once.
    fired: std::collections::HashSet<i64>,
    /// Track time (ms) of the last provisional evaluation.
    last_eval: i64,
}

pub fn is_live(source: &str) -> bool {
    let s = source.to_ascii_lowercase();
    is_youtube(source)
        || s.starts_with("rtsp://")
        || s.starts_with("rtsps://")
        || s.starts_with("http://")
        || s.starts_with("https://")
        || s.starts_with("srt://")
        || s.starts_with("/dev/")
        || s.starts_with("avfoundation:")
}

pub fn is_youtube(source: &str) -> bool {
    let s = source.to_ascii_lowercase();
    s.starts_with("youtube:") || s.contains("youtube.com/") || s.contains("youtu.be/")
}

fn is_http(source: &str) -> bool {
    let s = source.to_ascii_lowercase();
    s.starts_with("http://") || s.starts_with("https://")
}

fn is_hls(url: &str) -> bool {
    url.to_ascii_lowercase().contains("m3u8")
}

const YT_LIVE_FMT: &str = "best[height<=720][vcodec^=avc1]/best[height<=720]/best";

/// The User-Agent that yt-dlp used when it resolved the media URL. YouTube binds its signed
/// `googlevideo.com` URLs to the client, and a request with a different User-Agent gets HTTP 403
/// after the first segments, so ffmpeg and ffprobe must send the same one.
static YT_USER_AGENT: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
const FALLBACK_USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36";

/// `["-user_agent", ua]` for media URLs that need yt-dlp's client identity, else empty.
pub(crate) fn user_agent_args(url: &str) -> Vec<String> {
    if !url.contains("googlevideo.com") && !url.contains("youtube.com") {
        return Vec::new();
    }
    let ua = YT_USER_AGENT
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_else(|| FALLBACK_USER_AGENT.to_string());
    vec!["-user_agent".into(), ua]
}
const YT_VOD_FMT: &str = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]/b[height<=720]/b";

fn yt_dlp(args: &[&str]) -> Result<std::process::Output> {
    std::process::Command::new("yt-dlp")
        .args(args)
        .output()
        .context("run yt-dlp (brew install yt-dlp, or pip install yt-dlp)")
}

/// Decides how a source is ingested before anything starts. A YouTube *live* URL (or any stream URL)
/// is returned unchanged and ingested live; a YouTube *video* is downloaded once into `cache_dir`
/// and replayed as a file, so it gets real timestamps and clips are cut straight from the file.
pub fn prepare_source(source: &str, cache_dir: &Path) -> Result<String> {
    if !is_youtube(source) {
        return Ok(source.to_string());
    }
    let url = source.trim_start_matches("youtube:");
    let probe = yt_dlp(&["--no-playlist", "-O", "is_live", url])?;
    let flag = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .to_ascii_lowercase();
    if !probe.status.success() || flag == "true" {
        return Ok(source.to_string());
    }
    std::fs::create_dir_all(cache_dir)?;
    tracing::info!(
        "YouTube video (not live): downloading once to {}",
        cache_dir.display()
    );
    let tmpl = cache_dir.join("%(id)s.%(ext)s");
    let out = yt_dlp(&[
        "--no-playlist",
        "--no-simulate",
        "-f",
        YT_VOD_FMT,
        "--print",
        "after_move:filepath",
        "-o",
        &tmpl.to_string_lossy(),
        url,
    ])?;
    if !out.status.success() {
        anyhow::bail!(
            "yt-dlp download failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let path = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .map(str::to_string)
        .unwrap_or_default();
    anyhow::ensure!(
        Path::new(&path).is_file(),
        "yt-dlp did not report a downloaded file"
    );
    tracing::info!("downloaded {path}");
    Ok(path)
}

/// Turns a YouTube Live URL into a direct HLS media URL via yt-dlp, preferring H.264 at <=720p so
/// that codec-copy recording and browser playback work. Re-run on every reconnect: the URLs expire.
pub fn resolve_source(source: &str) -> Result<String> {
    if !is_youtube(source) {
        return Ok(source.to_string());
    }
    let url = source.trim_start_matches("youtube:");
    // One yt-dlp call prints the media URL and then the User-Agent it used.
    let out = yt_dlp(&[
        "--no-playlist",
        "-f",
        YT_LIVE_FMT,
        "--print",
        "urls",
        "--print",
        "%(http_headers.User-Agent)s",
        url,
    ])?;
    if !out.status.success() {
        anyhow::bail!(
            "yt-dlp failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    // `--print urls` can print more than one line for a merged format; the media URL is the
    // first line and the User-Agent (which never starts with "http") is the last one.
    let first = lines.first().copied().unwrap_or_default().to_string();
    anyhow::ensure!(first.starts_with("http"), "yt-dlp returned no URL");
    if let Some(ua) = lines.last().filter(|l| !l.starts_with("http"))
        && let Ok(mut g) = YT_USER_AGENT.lock()
    {
        *g = Some(ua.to_string());
    }
    Ok(first)
}

/// Input flags shared by the analysis pipe and the recorder. RTSP gets TCP + wall-clock stamps;
/// HTTP gets auto-reconnect; HLS starts at the newest segment so the preview is as live as possible
/// (wall-clock stamping is wrong for HLS: whole segments arrive in a burst and would collapse).
pub(crate) fn live_input_args(url: &str) -> Vec<String> {
    let mut a: Vec<String> = user_agent_args(url);
    if url.starts_with("rtsp") {
        a.extend(
            [
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "10000000",
                "-use_wallclock_as_timestamps",
                "1",
            ]
            .map(String::from),
        );
    }
    if is_http(url) {
        a.extend(
            [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
                // Fail the read when the server sends nothing for 20 s, instead of blocking for ever.
                "-rw_timeout",
                "20000000",
            ]
            .map(String::from),
        );
    }
    if is_hls(url) {
        a.extend(["-live_start_index", "-1"].map(String::from));
    }
    a.extend(["-fflags", "+genpts+discardcorrupt"].map(String::from));
    a
}

/// Starts a codec-copy recorder writing `<rec_dir>/<cam>/<epoch_seconds>.mp4` every ~10 s.
/// Segments are fragmented MP4, so one that is still being written (or was cut off by a crash)
/// plays back up to its last complete fragment.
fn spawn_recorder(url: &str, dir: &Path) -> Result<std::process::Child> {
    std::fs::create_dir_all(dir)?;
    let mut c = std::process::Command::new("ffmpeg");
    c.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
    c.args(live_input_args(url));
    c.args(["-i", url])
        .args([
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-c:v",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            "10",
            "-segment_format",
            "mp4",
            "-segment_format_options",
            "movflags=+frag_keyframe+empty_moov+default_base_moof",
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
        ])
        .arg(dir.join("%s.mp4"))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit());
    c.spawn().context("spawn ffmpeg recorder")
}

/// Deletes recorded segments older than the retention window.
fn sweep_recordings(dir: &Path, retain_hours: f64) {
    let cutoff = now_ms() / 1000 - (retain_hours * 3600.0) as i64;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".mp4")
                && let Ok(ts) = stem.parse::<i64>()
                && ts < cutoff
            {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

pub fn probe_duration_ms(path: &str) -> Option<i64> {
    let out = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().parse::<f64>().ok().map(|d| (d * 1000.0) as i64)
}

/// Helper processes (recorders) so a Ctrl-C / SIGTERM on the main process can take them down;
/// otherwise an orphaned ffmpeg would keep recording after Retrace exits.
static CHILDREN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

fn register_child(pid: u32) {
    if let Ok(mut v) = CHILDREN.lock() {
        v.push(pid);
    }
}

fn unregister_child(pid: u32) {
    if let Ok(mut v) = CHILDREN.lock() {
        v.retain(|p| *p != pid);
    }
}

/// Terminates every registered helper process (best effort, portable: uses `kill`).
pub fn kill_children() {
    let pids: Vec<u32> = CHILDREN.lock().map(|v| v.clone()).unwrap_or_default();
    for pid in pids {
        let _ = std::process::Command::new("kill")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        unregister_child(pid);
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn edge_of(cx: f32, cy: f32) -> &'static str {
    let m = 0.18;
    let mut best = ("interior", 1.0f32);
    for (name, d) in [
        ("left", cx),
        ("right", 1.0 - cx),
        ("top", cy),
        ("bottom", 1.0 - cy),
    ] {
        if d < m && d < best.1 {
            best = (name, d);
        }
    }
    best.0
}

fn direction_of(path: &[(i64, f32, f32, f32, f32)]) -> &'static str {
    if path.len() < 2 {
        return "stationary";
    }
    let (_, x0, y0, _, h0) = path[0];
    let (_, x1, y1, _, h1) = path[path.len() - 1];
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dh = h1 - h0;
    if dx.abs() < 0.06 && dy.abs() < 0.06 {
        return "stationary";
    }
    if dx.abs() >= dy.abs() {
        if dx > 0.0 {
            "left_to_right"
        } else {
            "right_to_left"
        }
    } else if dy > 0.0 || dh > 0.08 {
        "toward_camera"
    } else {
        "away_from_camera"
    }
}

fn path_length(path: &[(i64, f32, f32, f32, f32)]) -> f32 {
    path.windows(2)
        .map(|w| ((w[1].1 - w[0].1).powi(2) + (w[1].2 - w[0].2).powi(2)).sqrt())
        .sum()
}

fn write_jpeg(path: &Path, img: &RgbImage, quality: u8) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, quality);
    enc.encode_image(img)?;
    Ok(())
}

fn downscale(img: &RgbImage, max_w: u32) -> RgbImage {
    if img.width() <= max_w {
        return img.clone();
    }
    let h = (img.height() as f32 * max_w as f32 / img.width() as f32) as u32;
    image::imageops::resize(img, max_w, h.max(1), image::imageops::FilterType::Triangle)
}

/// Runs one ingest session to completion (file) or until the stream ends (live; the caller restarts).
pub fn run_ingest(
    opts: &IngestOpts,
    stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<IngestStats> {
    let t_wall0 = Instant::now();
    let db = Db::open(&opts.db_path)?;
    let mut det = Detector::load(&opts.models)?;
    let mut emb = Embedder::load(&opts.models)?;
    db.ensure_vec_table(emb.dim, &emb.name)?;
    let mut rules = RuleEngine::new();

    let live = is_live(&opts.source);
    let duration_ms = if live {
        0
    } else {
        probe_duration_ms(&opts.source).unwrap_or(0)
    };
    let t_start = if live {
        now_ms()
    } else {
        opts.start_ms.unwrap_or_else(|| now_ms() - duration_ms)
    };
    let cam_id = db.upsert_camera(
        &opts.cam,
        if live { "live" } else { "file" },
        &opts.source,
        0,
        0,
    )?;
    if !live {
        let abs = std::fs::canonicalize(&opts.source)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or(opts.source.clone());
        db.add_source(cam_id, &abs, t_start, duration_ms)?;
    }
    tracing::info!(cam = %opts.cam, source = %opts.source, live, t_start, duration_ms, detector = %det.name, embedder = %emb.name, "ingest start");

    let url = resolve_source(&opts.source)?;
    if url != opts.source {
        tracing::info!(cam = %opts.cam, "resolved {} -> {}…", opts.source, url.chars().take(60).collect::<String>());
    }
    let rec_dir = opts.rec_dir.join(&opts.cam);
    let mut recorder = if live && opts.record {
        match spawn_recorder(&url, &rec_dir) {
            Ok(c) => {
                register_child(c.id());
                Some(c)
            }
            Err(e) => {
                tracing::warn!(cam = %opts.cam, "recorder not started: {e:#}");
                None
            }
        }
    } else {
        None
    };

    // Audio windows run beside the video pipe when the embedder can hear (EmbeddingGemma 2).
    let audio_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let audio_pid = std::sync::Arc::new(std::sync::Mutex::new(None::<u32>));
    let audio_thread = if opts.audio && emb.audio && crate::audio::has_audio(&url) {
        let ao = crate::audio::AudioOpts {
            url: url.clone(),
            cam: opts.cam.clone(),
            live,
            t_start,
            db_path: opts.db_path.clone(),
            models: opts.models.clone(),
            win_s: 10.0,
            hop_s: 5.0,
            min_rms: 0.004,
        };
        let (st, pid) = (audio_stop.clone(), audio_pid.clone());
        Some(
            std::thread::Builder::new()
                .name(format!("audio-{}", opts.cam))
                .spawn(move || {
                    if let Err(e) = crate::audio::run_audio(&ao, st, pid) {
                        tracing::warn!(cam = %ao.cam, "audio: {e:#}");
                    }
                })?,
        )
    } else {
        None
    };

    let mut cmd = FfmpegCommand::new();
    cmd.args(["-nostdin"]); // ffmpeg-sidecar sets its own loglevel; it parses stderr to learn the output frame format
    if live {
        cmd.args(live_input_args(&url));
    }
    cmd.input(&url)
        .args([
            "-an",
            "-vf",
            &format!("fps={},scale={}:-2", opts.fps, opts.width),
        ])
        .rawvideo();
    let mut child = cmd.spawn().context("spawn ffmpeg (is ffmpeg on PATH?)")?;

    // Stall watchdog (live sources). An HTTP/HLS server can keep the connection open and send
    // nothing. ffmpeg then blocks on read, the frame loop never returns, and the restart loop in
    // `run_live_forever` never runs. When no frame arrives for STALL_SECS, kill ffmpeg: the loop
    // ends, the source is resolved again (YouTube media URLs expire), and ffmpeg restarts.
    let last_frame_ms = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(now_ms()));
    let watchdog_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    if live {
        let pid = child.as_inner().id();
        let last = last_frame_ms.clone();
        let stop_flag = watchdog_stop.clone();
        let cam = opts.cam.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_secs(5));
                if stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let age_s = (now_ms() - last.load(std::sync::atomic::Ordering::Relaxed)) / 1000;
                if age_s >= STALL_SECS {
                    tracing::warn!(cam = %cam, "no frames for {age_s} s: restarting ffmpeg");
                    // ffmpeg handles SIGTERM in its main loop, which can be blocked on the network
                    // read here, so escalate to SIGKILL after a short grace period.
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .status();
                    std::thread::sleep(Duration::from_secs(3));
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &pid.to_string()])
                        .status();
                    return;
                }
            }
        });
    }

    let mut tracker = Tracker::new(TrackerConfig {
        max_age: (opts.fps * 2.0).max(4.0) as usize,
        max_age_tentative: 3,
        min_hits: 2,
        high_conf: opts.models.conf,
        ..TrackerConfig::default()
    });
    let mut tracks: HashMap<usize, Track> = HashMap::new();
    let mut stats = IngestStats::default();
    let stale_frames = (opts.stale_s * opts.fps).max(2.0) as usize;
    let max_track_ms = (opts.max_track_s * 1000.0) as i64;
    let mut frame_idx = 0usize;
    let mut last_log = Instant::now();
    let mut frame_w = 0u32;
    let mut frame_h = 0u32;
    let mut last_ts = t_start;
    let mut last_frame_wall = Instant::now();
    let mut ema_dt = 0.0f64;
    let mut proc_ema = 0.0f64;
    let mut skip_every = 1usize;

    // Writes (or rewrites) a track as a tracklet record: crops, context frame, embedding, row and
    // vector. Used both for provisional evaluation of long-dwelling tracks and at track close.
    let persist = |id: usize,
                   tr: &mut Track,
                   db: &Db,
                   emb: &mut Embedder,
                   rules: &mut RuleEngine,
                   stats: &mut IngestStats,
                   frame_w: u32,
                   frame_h: u32|
     -> Result<()> {
        if tr.crops.is_empty() || tr.n_frames < 2 {
            return Ok(());
        }
        let class = tr
            .class_votes
            .iter()
            .max_by_key(|(_, n)| **n)
            .map(|(c, _)| c.clone())
            .unwrap_or("object".into());
        tr.crops.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        tr.crops.truncate(3);
        let stem = format!("{}/{}-{}", opts.cam, tr.t0, id);
        let mut refs = Vec::new();
        let mut imgs: Vec<RgbImage> = Vec::new();
        for (k, c) in tr.crops.iter().enumerate() {
            let rel = format!("{stem}-{k}.jpg");
            write_jpeg(&opts.crops_dir.join(&rel), &c.img, 88)?;
            refs.push(CropRef {
                path: rel,
                t: c.t,
                bbox: c.bbox,
                score: c.score,
            });
            imgs.push(c.img.clone());
        }
        if let Some(ctx) = tr.crops[0].ctx.as_ref() {
            write_jpeg(&opts.crops_dir.join(format!("{stem}-ctx.jpg")), ctx, 80)?;
        }
        let v = emb.embed_image_set(&imgs)?;
        let secs = ((tr.t1 - tr.t0) as f32 / 1000.0).max(0.25);
        let nt = NewTracklet {
            cam: cam_id,
            class: class.clone(),
            t0: tr.t0,
            t1: tr.t1,
            path_json: serde_json::to_string(
                &tr.path
                    .iter()
                    .map(|p| vec![p.0 as f64, p.1 as f64, p.2 as f64, p.3 as f64, p.4 as f64])
                    .collect::<Vec<_>>(),
            )?,
            entry_edge: edge_of(tr.path[0].1, tr.path[0].2).into(),
            exit_edge: edge_of(tr.path[tr.path.len() - 1].1, tr.path[tr.path.len() - 1].2).into(),
            dwell_s: secs as f64,
            speed: (path_length(&tr.path) / secs) as f64,
            direction: direction_of(&tr.path).into(),
            best_crops_json: serde_json::to_string(&refs)?,
            n_frames: tr.n_frames as i64,
            bbox_json: serde_json::to_string(&tr.crops[0].bbox)?,
            frame_w: frame_w as i64,
            frame_h: frame_h as i64,
            emb: v,
        };
        let tid = match tr.db_id {
            Some(tid) => {
                db.update_tracklet(tid, &nt)?;
                tid
            }
            None => {
                let tid = db.insert_tracklet(&nt)?;
                tr.db_id = Some(tid);
                stats.tracklets += 1;
                tid
            }
        };
        let fired = rules.evaluate(db, tid, &nt, &opts.cam, &mut tr.fired)?;
        stats.alerts += fired;
        Ok(())
    };

    for ev in child.iter()? {
        if let Some(s) = &stop
            && s.load(std::sync::atomic::Ordering::Relaxed)
        {
            break;
        }
        match ev {
            FfmpegEvent::OutputFrame(f) => {
                frame_idx += 1;
                stats.frames += 1;
                last_frame_ms.store(now_ms(), std::sync::atomic::Ordering::Relaxed);
                if let Some(m) = opts.max_frames
                    && frame_idx > m
                {
                    break;
                }
                // Backpressure for live sources: if analysis is slower than the requested fps the
                // pipe would fill and the stream would drift behind real time, so analyse every
                // n-th frame instead (reading and dropping a frame is cheap, detection is not).
                if live && skip_every > 1 && !frame_idx.is_multiple_of(skip_every) {
                    stats.skipped += 1;
                    continue;
                }
                let proc_t0 = Instant::now();
                frame_w = f.width;
                frame_h = f.height;
                let ts = if live {
                    now_ms()
                } else {
                    t_start + (f.timestamp as f64 * 1000.0) as i64
                };
                last_ts = ts;
                let Some(frame) = RgbImage::from_raw(f.width, f.height, f.data) else {
                    continue;
                };
                let img = Image::from(frame.clone());
                let (fw, fh) = (f.width as f32, f.height as f32);
                let dets: Vec<Det> = det
                    .detect(&img)?
                    .iter()
                    .map(|h| {
                        let (x, y, w, h2) = h.xywh();
                        let (x, y) = (x.max(0.0), y.max(0.0));
                        Det {
                            x,
                            y,
                            w: w.min(fw - x).max(1.0),
                            h: h2.min(fh - y).max(1.0),
                            conf: h.confidence().unwrap_or(0.5),
                            class: h.name().unwrap_or("object").to_string(),
                        }
                    })
                    .collect();
                stats.detections += dets.iter().filter(|d| d.conf >= opts.models.conf).count();
                let tracked: Vec<Tracked> = tracker.update(&dets);
                if frame_idx <= 12 || frame_idx.is_multiple_of(40) {
                    let ids: Vec<String> = tracked
                        .iter()
                        .map(|t| {
                            format!(
                                "t{}:{}:{:.2}{}",
                                t.id,
                                t.class,
                                t.conf,
                                if t.confirmed { "" } else { "?" }
                            )
                        })
                        .collect();
                    tracing::debug!(
                        frame = frame_idx,
                        dets = dets.len(),
                        tracked = tracked.len(),
                        ?ids,
                        "frame"
                    );
                }
                for tk in tracked.iter().filter(|t| t.confirmed) {
                    let id = tk.id;
                    let (x, y, w, h) = (tk.x, tk.y, tk.w, tk.h);
                    let cx = (x + w / 2.0) / fw;
                    let cy = (y + h / 2.0) / fh;
                    let conf = tk.conf;
                    let name = tk.class.clone();
                    let tr = tracks.entry(id).or_insert_with(|| Track {
                        class_votes: HashMap::new(),
                        t0: ts,
                        t1: ts,
                        path: vec![],
                        crops: vec![],
                        last_frame: frame_idx,
                        n_frames: 0,
                        db_id: None,
                        fired: std::collections::HashSet::new(),
                        last_eval: 0,
                    });
                    *tr.class_votes.entry(name).or_insert(0) += 1;
                    tr.t1 = ts;
                    tr.last_frame = frame_idx;
                    tr.n_frames += 1;
                    tr.path.push((ts, cx, cy, w / fw, h / fh));
                    if h >= opts.min_crop_h as f32 {
                        let score = (w * h).sqrt() * conf;
                        let worst = tr
                            .crops
                            .iter()
                            .map(|c| c.score)
                            .fold(f32::INFINITY, f32::min);
                        if tr.crops.len() < 3 || score > worst {
                            let crop = image::imageops::crop_imm(
                                &frame, x as u32, y as u32, w as u32, h as u32,
                            )
                            .to_image();
                            let ctx = Some(downscale(&frame, 480));
                            tr.crops.push(CropCand {
                                img: crop,
                                t: ts,
                                bbox: [x / fw, y / fh, w / fw, h / fh],
                                score,
                                ctx,
                            });
                            if tr.crops.len() > 3 {
                                let mut worst_i = 0;
                                for (i, c) in tr.crops.iter().enumerate() {
                                    if c.score < tr.crops[worst_i].score {
                                        worst_i = i;
                                    }
                                }
                                tr.crops.remove(worst_i);
                            }
                        }
                    }
                }
                // Live preview: boxes, ids, trails and a status bar, published for /live.mjpeg.
                if let Some(lf) = opts.live_frames.as_ref() {
                    let now = Instant::now();
                    let dt = now.duration_since(last_frame_wall).as_secs_f64();
                    last_frame_wall = now;
                    ema_dt = if ema_dt == 0.0 {
                        dt
                    } else {
                        0.8 * ema_dt + 0.2 * dt
                    };
                    let mut canvas = frame;
                    for tk in &tracked {
                        let id = tk.id;
                        let (x, y, w, h) = (tk.x, tk.y, tk.w, tk.h);
                        let class = tk.class.as_str();
                        let color = overlay::class_color(class);
                        if !tk.confirmed {
                            // Seen once: drawn thin until the tracker confirms it on the next frame.
                            overlay::draw_rect(
                                &mut canvas,
                                x as i32,
                                y as i32,
                                w as i32,
                                h as i32,
                                1,
                                [210, 210, 210],
                            );
                            continue;
                        }
                        if let Some(tr) = tracks.get(&id) {
                            let trail: Vec<(f32, f32)> = tr
                                .path
                                .iter()
                                .rev()
                                .take(48)
                                .map(|p| (p.1, p.2 + p.4 / 2.0))
                                .collect();
                            overlay::draw_trail(&mut canvas, &trail, color);
                        }
                        let dwell = tracks
                            .get(&id)
                            .map(|t| (t.t1 - t.t0) as f64 / 1000.0)
                            .unwrap_or(0.0);
                        let label = if dwell >= 1.0 {
                            format!("{} {} · {:.0}S", class, id, dwell)
                        } else {
                            format!("{} {}", class, id)
                        };
                        overlay::draw_box_label(
                            &mut canvas,
                            x as i32,
                            y as i32,
                            w as i32,
                            h as i32,
                            &label,
                            color,
                        );
                    }
                    let clock = chrono::DateTime::from_timestamp_millis(ts)
                        .map(|d| {
                            d.with_timezone(&chrono::Local)
                                .format("%H:%M:%S")
                                .to_string()
                        })
                        .unwrap_or_default();
                    let status = format!(
                        "{} {} {} {:.1} FPS{} {} TRACKS",
                        if live { "LIVE" } else { "REPLAY" },
                        opts.cam,
                        clock,
                        if ema_dt > 0.0 { 1.0 / ema_dt } else { 0.0 },
                        if skip_every > 1 {
                            format!(" 1/{skip_every}")
                        } else {
                            String::new()
                        },
                        tracker.alive()
                    );
                    overlay::draw_status(&mut canvas, &status);
                    if let Ok(jpeg) = overlay::encode_jpeg(&canvas, 72) {
                        lf.publish(&opts.cam, jpeg, ts, tracks.len(), f.width, f.height);
                    }
                }
                // Provisional evaluation: a track that has dwelt >= 8 s is written and checked
                // against the rules every 10 s of track time, so "loitering" alerts fire while the
                // person is still there and the tracklet is searchable before it closes.
                let provisional: Vec<usize> = tracks
                    .iter()
                    .filter(|(_, t)| {
                        !t.crops.is_empty()
                            && ts - t.t0 >= 8_000
                            && ts - t.last_eval.max(t.t0) >= 10_000
                    })
                    .map(|(id, _)| *id)
                    .collect();
                let mut provisional = provisional;
                provisional.sort_unstable(); // HashMap order is random; keep tracklet ids reproducible
                for id in provisional {
                    if let Some(tr) = tracks.get_mut(&id) {
                        tr.last_eval = ts;
                        persist(
                            id, tr, &db, &mut emb, &mut rules, &mut stats, frame_w, frame_h,
                        )?;
                    }
                }
                // Close stale or over-long tracks.
                let stale: Vec<usize> = tracks
                    .iter()
                    .filter(|(_, t)| {
                        frame_idx - t.last_frame > stale_frames || ts - t.t0 > max_track_ms
                    })
                    .map(|(id, _)| *id)
                    .collect();
                let mut stale = stale;
                stale.sort_unstable();
                for id in stale {
                    if let Some(mut tr) = tracks.remove(&id) {
                        persist(
                            id, &mut tr, &db, &mut emb, &mut rules, &mut stats, frame_w, frame_h,
                        )?;
                    }
                }
                if live {
                    let dt = proc_t0.elapsed().as_secs_f64();
                    proc_ema = if proc_ema == 0.0 {
                        dt
                    } else {
                        0.8 * proc_ema + 0.2 * dt
                    };
                    skip_every = ((proc_ema * opts.fps as f64 * 1.1).ceil() as usize).clamp(1, 8);
                }
                if last_log.elapsed() > Duration::from_secs(5) {
                    last_log = Instant::now();
                    if recorder.is_some() {
                        sweep_recordings(&rec_dir, opts.retain_hours);
                    }
                    let fps =
                        (stats.frames - stats.skipped) as f64 / t_wall0.elapsed().as_secs_f64();
                    tracing::info!(cam = %opts.cam, frames = stats.frames, skipped = stats.skipped, open = tracks.len(), tracklets = stats.tracklets, alerts = stats.alerts, fps = format!("{fps:.1}"), "ingest");
                }
            }
            FfmpegEvent::Log(LogLevel::Error, msg) | FfmpegEvent::Error(msg) => {
                tracing::warn!(cam = %opts.cam, "ffmpeg: {msg}")
            }
            _ => {}
        }
    }
    watchdog_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = child.kill();
    let _ = child.wait();
    if let Some(mut r) = recorder.take() {
        unregister_child(r.id());
        let _ = r.kill();
        let _ = r.wait();
    }
    if let Some(t) = audio_thread {
        audio_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if live && let Some(pid) = audio_pid.lock().ok().and_then(|p| *p) {
            let _ = std::process::Command::new("kill")
                .arg(pid.to_string())
                .status();
        }
        let _ = t.join();
    }
    let mut remaining: Vec<(usize, Track)> = tracks.drain().collect();
    remaining.sort_by_key(|(id, _)| *id);
    for (id, mut tr) in remaining {
        persist(
            id, &mut tr, &db, &mut emb, &mut rules, &mut stats, frame_w, frame_h,
        )?;
    }
    stats.elapsed_s = t_wall0.elapsed().as_secs_f64();
    stats.video_s = ((last_ts - t_start) as f64 / 1000.0).max(0.0);
    tracing::info!(cam = %opts.cam, ?stats, "ingest done");
    Ok(stats)
}

/// Keeps a live source running: restarts ffmpeg with backoff when the stream drops.
pub fn run_live_forever(opts: IngestOpts, stop: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let mut backoff = 1u64;
    loop {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        match run_ingest(&opts, Some(stop.clone())) {
            Ok(s) if s.frames > 50 => backoff = 1,
            Ok(_) => backoff = (backoff * 2).min(30),
            Err(e) => {
                tracing::error!(cam = %opts.cam, "ingest error: {e:#}");
                backoff = (backoff * 2).min(30);
            }
        }
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        tracing::info!(cam = %opts.cam, "restarting source in {backoff}s");
        std::thread::sleep(Duration::from_secs(backoff));
    }
}
