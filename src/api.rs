//! HTTP API + embedded UI (axum).

use crate::db::Db;
use crate::models::Embedder;
use crate::overlay::LiveFrames;
use crate::rules::backtest;
use crate::search::{SearchParams, search};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Path as AxPath, Query, State},
    http::{StatusCode, header},
    response::{
        IntoResponse, Redirect, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::get,
};
use futures::stream::Stream;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct AppState {
    pub db: Mutex<Db>,
    pub emb: Mutex<Embedder>,
    pub crops_dir: PathBuf,
    pub rec_dir: PathBuf,
    pub mediamtx: Option<String>,
    pub started: std::time::Instant,
    /// Latest annotated frame per camera (filled by in-process ingest threads).
    pub live: LiveFrames,
    /// True when the desktop app serves the engine (the UI then shows desktop hints, not CLI ones).
    pub desktop: bool,
}

type S = State<Arc<AppState>>;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(ui))
        .route("/api/stats", get(stats))
        .route("/api/search", get(api_search))
        .route("/api/tracklets", get(list_tracklets))
        .route("/api/tracklets/{id}", get(get_tracklet))
        .route("/api/similar/{id}", get(similar))
        .route("/api/live", get(live))
        .route("/api/events", get(list_events).post(post_event))
        .route("/api/rules", get(list_rules).post(post_rule))
        .route("/api/rules/{id}", axum::routing::delete(delete_rule))
        .route("/api/alerts", get(list_alerts))
        .route("/thumb/{id}", get(thumb))
        .route("/ctx/{id}", get(ctx))
        .route("/clip", get(clip))
        .route("/api/live-cams", get(live_cams))
        .route("/frame.jpg", get(frame_jpg))
        .route("/live.mjpeg", get(live_mjpeg))
        .with_state(state)
}

#[derive(Deserialize)]
struct CamQ {
    cam: String,
}

/// Cameras with a live preview available, newest frame first.
async fn live_cams(State(st): S) -> Response {
    let mut cams: Vec<serde_json::Value> = st
        .live
        .cams()
        .into_iter()
        .filter_map(|(name, _, _)| st.live.get(&name).map(|f| (name, f)))
        .map(|(name, f)| {
            serde_json::json!({"cam": name, "t": f.t, "seq": f.seq, "tracks": f.tracks,
                "width": f.width, "height": f.height})
        })
        .collect();
    cams.sort_by_key(|v| -v["t"].as_i64().unwrap_or(0));
    Json(cams).into_response()
}

/// The latest annotated frame as a single JPEG.
async fn frame_jpg(State(st): S, Query(q): Query<CamQ>) -> Response {
    match st.live.get(&q.cam) {
        Some(f) => (
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            f.jpeg,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Multipart MJPEG stream of annotated frames: `<img src="/live.mjpeg?cam=front">` just works.
async fn live_mjpeg(State(st): S, Query(q): Query<CamQ>) -> Response {
    const BOUNDARY: &str = "retraceframe";
    let live = st.live.clone();
    let cam = q.cam.clone();
    let stream = futures::stream::unfold(0u64, move |last_seq| {
        let live = live.clone();
        let cam = cam.clone();
        async move {
            // Wait for a new frame (poll at 20 Hz; frames arrive at the analysis fps).
            let mut waited = 0u32;
            loop {
                if let Some(f) = live.get(&cam)
                    && f.seq != last_seq
                {
                    let mut part = format!(
                            "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Timestamp: {}\r\n\r\n",
                            f.jpeg.len(),
                            f.t
                        )
                        .into_bytes();
                    part.extend_from_slice(&f.jpeg);
                    part.extend_from_slice(b"\r\n");
                    return Some((
                        Ok::<_, std::io::Error>(axum::body::Bytes::from(part)),
                        f.seq,
                    ));
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                waited += 1;
                if waited > 600 {
                    // 30 s without a frame: emit a keep-alive boundary so proxies do not time out.
                    return Some((
                        Ok(axum::body::Bytes::from(format!("--{BOUNDARY}\r\n"))),
                        last_seq,
                    ));
                }
            }
        }
    });
    (
        [
            (
                header::CONTENT_TYPE,
                format!("multipart/x-mixed-replace; boundary={BOUNDARY}"),
            ),
            (header::CACHE_CONTROL, "no-store".to_string()),
            (header::PRAGMA, "no-cache".to_string()),
        ],
        axum::body::Body::from_stream(stream),
    )
        .into_response()
}

async fn ui() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../ui/index.html"),
    )
}

fn err(e: anyhow::Error) -> Response {
    tracing::warn!("api error: {e:#}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({"error": e.to_string()})),
    )
        .into_response()
}

async fn stats(State(st): S) -> Response {
    let db = st.db.lock().unwrap();
    match db.stats() {
        Ok(mut v) => {
            v["uptime_s"] = serde_json::json!(st.started.elapsed().as_secs());
            v["mediamtx"] = serde_json::json!(st.mediamtx);
            v["desktop"] = serde_json::json!(st.desktop);
            Json(v).into_response()
        }
        Err(e) => err(e),
    }
}

async fn api_search(State(st): S, Query(p): Query<SearchParams>) -> Response {
    if p.q.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "q is required"})),
        )
            .into_response();
    }
    let st2 = st.clone();
    let res = tokio::task::spawn_blocking(move || {
        let db = st2.db.lock().unwrap();
        let mut emb = st2.emb.lock().unwrap();
        let r = search(&db, &mut emb, &p);
        let _ = db.audit("search", &p.q);
        r
    })
    .await
    .map_err(|e| anyhow::anyhow!(e));
    match res {
        Ok(Ok(r)) => Json(r).into_response(),
        Ok(Err(e)) | Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct ListQ {
    cam: Option<String>,
    t0: Option<i64>,
    t1: Option<i64>,
    limit: Option<usize>,
    since: Option<i64>,
}

async fn list_tracklets(State(st): S, Query(q): Query<ListQ>) -> Response {
    let db = st.db.lock().unwrap();
    let r = (|| -> Result<serde_json::Value> {
        if let Some(since) = q.since {
            return Ok(serde_json::json!(
                db.recent_tracklets(since, q.limit.unwrap_or(50))?
            ));
        }
        let cam = match &q.cam {
            Some(n) if !n.is_empty() => match db.camera_by_name(n)? {
                Some(c) => Some(c.id),
                None => anyhow::bail!("unknown camera '{n}'"),
            },
            _ => None,
        };
        let t0 = q.t0.unwrap_or(0);
        let t1 = q.t1.unwrap_or(i64::MAX / 2);
        Ok(serde_json::json!(db.tracklets_in(
            cam,
            t0,
            t1,
            q.limit.unwrap_or(500)
        )?))
    })();
    match r {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn get_tracklet(State(st): S, AxPath(id): AxPath<i64>) -> Response {
    let db = st.db.lock().unwrap();
    match db.tracklet(id) {
        Ok(Some(t)) => Json(t).into_response(),
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => err(e),
    }
}

async fn similar(State(st): S, AxPath(id): AxPath<i64>, Query(q): Query<ListQ>) -> Response {
    let db = st.db.lock().unwrap();
    let r = (|| -> Result<serde_json::Value> {
        let Some(v) = db.emb_of(id)? else {
            anyhow::bail!("no vector for tracklet {id}")
        };
        let hits = db.knn(
            &v,
            q.limit.unwrap_or(24).clamp(1, 500) + 1,
            None,
            None,
            None,
            None,
        )?;
        let mut out = Vec::new();
        for (hid, score) in hits {
            if hid == id {
                continue;
            }
            if let Some(t) = db.tracklet(hid)? {
                out.push(serde_json::json!({"tracklet": t, "score": score}));
            }
        }
        Ok(serde_json::json!({"hits": out}))
    })();
    match r {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

/// Server-sent events: new tracklets and alerts, polled from the database so a separate ingest
/// process (or several) shows up here too.
async fn live(State(st): S) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = async_stream(st);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

fn async_stream(st: Arc<AppState>) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        let (mut last_t, mut last_a) = {
            let db = st.db.lock().unwrap();
            let lt = db
                .recent_tracklets(0, 1)
                .ok()
                .and_then(|v| v.first().map(|t| t.id))
                .unwrap_or(0);
            let la = db
                .alerts(0, 1)
                .ok()
                .and_then(|v| v.first().map(|a| a.id))
                .unwrap_or(0);
            (lt, la)
        };
        loop {
            tokio::time::sleep(Duration::from_millis(800)).await;
            if tx.is_closed() {
                return; // client went away during a quiet period; do not poll the db forever
            }
            let (ts, als) = {
                let db = st.db.lock().unwrap();
                (
                    db.recent_tracklets(last_t, 20).unwrap_or_default(),
                    db.alerts(last_a, 20).unwrap_or_default(),
                )
            };
            for t in ts.iter().rev() {
                last_t = last_t.max(t.id);
                if tx
                    .send(
                        Event::default()
                            .event("tracklet")
                            .data(serde_json::to_string(t).unwrap_or_default()),
                    )
                    .await
                    .is_err()
                {
                    return;
                }
            }
            for a in als.iter().rev() {
                last_a = last_a.max(a.id);
                if tx
                    .send(
                        Event::default()
                            .event("alert")
                            .data(serde_json::to_string(a).unwrap_or_default()),
                    )
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok)
}
use futures::StreamExt;

#[derive(Deserialize)]
struct EventIn {
    cam: Option<String>,
    ts: Option<i64>,
    source: Option<String>,
    r#type: String,
    attrs: Option<serde_json::Value>,
}

async fn post_event(State(st): S, Json(e): Json<EventIn>) -> Response {
    let db = st.db.lock().unwrap();
    let r = (|| -> Result<serde_json::Value> {
        let cam = match &e.cam {
            Some(n) if !n.is_empty() => match db.camera_by_name(n)? {
                Some(c) => Some(c.id),
                None => anyhow::bail!("unknown camera '{n}'"),
            },
            _ => None,
        };
        let ts =
            e.ts.unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
        let id = db.insert_event(
            cam,
            ts,
            e.source.as_deref().unwrap_or("http"),
            &e.r#type,
            &e.attrs.unwrap_or(serde_json::json!({})),
        )?;
        Ok(serde_json::json!({"id": id, "ts": ts}))
    })();
    match r {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn list_events(State(st): S, Query(q): Query<ListQ>) -> Response {
    let db = st.db.lock().unwrap();
    match db.events(
        q.t0.unwrap_or(0),
        q.t1.unwrap_or(i64::MAX / 2),
        q.limit.unwrap_or(200),
    ) {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct RuleIn {
    name: Option<String>,
    text: String,
    min_score: Option<f64>,
    class: Option<String>,
    cam: Option<String>,
    min_dwell_s: Option<f64>,
}

async fn post_rule(State(st): S, Json(r): Json<RuleIn>) -> Response {
    let st2 = st.clone();
    let res = tokio::task::spawn_blocking(move || -> Result<serde_json::Value> {
        let db = st2.db.lock().unwrap();
        let mut emb = st2.emb.lock().unwrap();
        let (_, v) = emb.embed_query(&r.text)?;
        let cam = match &r.cam {
            Some(n) if !n.is_empty() => match db.camera_by_name(n)? {
                Some(c) => Some(c.id),
                None => anyhow::bail!("unknown camera '{n}'"),
            },
            _ => None,
        };
        let class = r.class.as_deref().filter(|c| !c.is_empty() && *c != "any");
        let min_score = r.min_score.unwrap_or(0.26);
        let name = r.name.clone().unwrap_or_else(|| r.text.clone());
        let id = db.insert_rule(
            &name,
            &r.text,
            min_score,
            class,
            cam,
            r.min_dwell_s.unwrap_or(0.0),
            &v,
        )?;
        let now = chrono::Utc::now().timestamp_millis();
        let bt = backtest(
            &db,
            &v,
            min_score,
            class,
            cam,
            r.min_dwell_s.unwrap_or(0.0),
            now - 7 * 86_400_000,
            now + 86_400_000,
        )?;
        db.audit("rule.create", &r.text)?;
        Ok(serde_json::json!({"id": id, "name": name, "backtest": bt}))
    })
    .await
    .map_err(|e| anyhow::anyhow!(e));
    match res {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) | Err(e) => err(e),
    }
}

async fn list_rules(State(st): S) -> Response {
    let db = st.db.lock().unwrap();
    match db.rules(false) {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn delete_rule(State(st): S, AxPath(id): AxPath<i64>) -> Response {
    let db = st.db.lock().unwrap();
    match db.delete_rule(id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err(e),
    }
}

async fn list_alerts(State(st): S, Query(q): Query<ListQ>) -> Response {
    let db = st.db.lock().unwrap();
    match db.alerts(q.since.unwrap_or(0), q.limit.unwrap_or(100)) {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

#[derive(Deserialize)]
struct ThumbQ {
    k: Option<usize>,
}

async fn thumb(State(st): S, AxPath(id): AxPath<i64>, Query(q): Query<ThumbQ>) -> Response {
    let rel = {
        let db = st.db.lock().unwrap();
        match db.tracklet(id) {
            Ok(Some(t)) => {
                let crops: Vec<serde_json::Value> =
                    serde_json::from_str(&t.best_crops_json).unwrap_or_default();
                crops
                    .get(q.k.unwrap_or(0))
                    .and_then(|c| c["path"].as_str().map(|s| s.to_string()))
            }
            _ => None,
        }
    };
    serve_jpeg(st.crops_dir.clone(), rel).await
}

async fn ctx(State(st): S, AxPath(id): AxPath<i64>) -> Response {
    let rel = {
        let db = st.db.lock().unwrap();
        match db.tracklet(id) {
            Ok(Some(t)) => {
                let crops: Vec<serde_json::Value> =
                    serde_json::from_str(&t.best_crops_json).unwrap_or_default();
                crops.first().and_then(|c| c["path"].as_str()).map(|p| {
                    p.rsplit_once('-')
                        .map(|(a, _)| format!("{a}-ctx.jpg"))
                        .unwrap_or_default()
                })
            }
            _ => None,
        }
    };
    serve_jpeg(st.crops_dir.clone(), rel).await
}

async fn serve_jpeg(dir: PathBuf, rel: Option<String>) -> Response {
    let Some(rel) = rel else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if rel.contains("..") {
        return StatusCode::BAD_REQUEST.into_response();
    }
    match tokio::fs::read(dir.join(rel)).await {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "image/jpeg"),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[derive(Deserialize)]
struct ClipQ {
    cam: String,
    t0: i64,
    t1: Option<i64>,
    pre: Option<i64>,
    dur: Option<i64>,
}

/// Returns an MP4 for [t0 - pre, t0 - pre + dur]. File sources are cut with ffmpeg; live sources
/// redirect to the MediaMTX playback server when one is configured.
async fn clip(State(st): S, Query(q): Query<ClipQ>) -> Response {
    let pre = q.pre.unwrap_or(4000).max(0);
    let start = q.t0 - pre;
    let dur = q
        .dur
        .unwrap_or_else(|| (q.t1.unwrap_or(q.t0 + 8000) - start + 3000).clamp(3000, 60_000))
        .clamp(1000, 60_000);
    let (cam, src) = {
        let db = st.db.lock().unwrap();
        let cam = match db.camera_by_name(&q.cam) {
            Ok(Some(c)) => c,
            _ => return StatusCode::NOT_FOUND.into_response(),
        };
        let src = db.source_for(cam.id, start.max(q.t0)).ok().flatten();
        (cam, src)
    };
    if let Some(src) = src {
        let offset = ((start - src.t_start).max(0) as f64) / 1000.0;
        let secs = dur as f64 / 1000.0;
        let path = src.path.clone();
        let out = tokio::task::spawn_blocking(move || {
            crate::ingest::output_tracked(std::process::Command::new(crate::tools::ffmpeg()).args(
                [
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-ss",
                    &format!("{offset:.3}"),
                    "-i",
                    &path,
                    "-t",
                    &format!("{secs:.3}"),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    "-vf",
                    "scale='min(960,iw)':-2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                    "-f",
                    "mp4",
                    "-",
                ],
            ))
        })
        .await;
        return match out {
            Ok(Ok(o)) if o.status.success() => (
                [
                    (header::CONTENT_TYPE, "video/mp4"),
                    (header::CACHE_CONTROL, "public, max-age=3600"),
                ],
                o.stdout,
            )
                .into_response(),
            Ok(Ok(o)) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                String::from_utf8_lossy(&o.stderr).to_string(),
            )
                .into_response(),
            Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
    }
    // Built-in recordings: ~10 s codec-copied segments named by epoch second. A segment runs from
    // its own timestamp to the next segment's (or now); tiny files are segments that never got data.
    let seg_dir = st.rec_dir.join(&cam.name);
    if let Ok(rd) = std::fs::read_dir(&seg_dir) {
        let mut segs: Vec<(i64, PathBuf)> = rd
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                let size = e.metadata().map(|m| m.len()).unwrap_or(0);
                n.strip_suffix(".mp4")
                    .and_then(|s| s.parse::<i64>().ok())
                    .filter(|_| size > 4096)
                    .map(|t| (t * 1000, e.path()))
            })
            .collect();
        segs.sort();
        let end = start + dur;
        let now = chrono::Utc::now().timestamp_millis();
        let chosen: Vec<&(i64, PathBuf)> = segs
            .iter()
            .enumerate()
            .filter(|(i, (t, _))| {
                let seg_end = segs.get(i + 1).map(|n| n.0).unwrap_or(now + 1000);
                *t < end && seg_end > start
            })
            .map(|(_, s)| s)
            .collect();
        if !chosen.is_empty() {
            let first = chosen[0].0;
            let offset = ((start - first).max(0) as f64) / 1000.0;
            let secs = dur as f64 / 1000.0;
            // The concat demuxer resolves relative paths against the list file, so make them absolute.
            let list = chosen
                .iter()
                .map(|(_, p)| {
                    let abs = p.canonicalize().unwrap_or_else(|_| p.clone());
                    format!(
                        "file '{}'",
                        abs.display().to_string().replace('\'', "'\\''")
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            // A private, unpredictable path (O_EXCL) rather than a name built from query values.
            let tmp = match tempfile::Builder::new()
                .prefix("retrace-concat-")
                .suffix(".txt")
                .tempfile()
                .and_then(|f| f.keep().map(|(_, p)| p).map_err(|e| e.error))
            {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, "concat list temp file");
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
            };
            let written = std::fs::write(&tmp, list);
            if written.is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
            if written.is_ok() {
                let tmp2 = tmp.clone();
                let out = tokio::task::spawn_blocking(move || {
                    let o = crate::ingest::output_tracked(
                        std::process::Command::new(crate::tools::ffmpeg())
                            .args([
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                "-nostdin",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                            ])
                            .arg(&tmp2)
                            .args([
                                "-ss",
                                &format!("{offset:.3}"),
                                "-t",
                                &format!("{secs:.3}"),
                                "-c:a",
                                "aac",
                                "-b:a",
                                "64k",
                                "-vf",
                                "scale='min(960,iw)':-2",
                                "-c:v",
                                "libx264",
                                "-preset",
                                "veryfast",
                                "-crf",
                                "23",
                                "-pix_fmt",
                                "yuv420p",
                                "-movflags",
                                "+frag_keyframe+empty_moov+default_base_moof",
                                "-f",
                                "mp4",
                                "-",
                            ]),
                    );
                    let _ = std::fs::remove_file(&tmp2);
                    o
                })
                .await;
                return match out {
                    Ok(Ok(o)) if o.status.success() => (
                        [
                            (header::CONTENT_TYPE, "video/mp4"),
                            (header::CACHE_CONTROL, "public, max-age=3600"),
                        ],
                        o.stdout,
                    )
                        .into_response(),
                    Ok(Ok(o)) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        String::from_utf8_lossy(&o.stderr).to_string(),
                    )
                        .into_response(),
                    Ok(Err(e)) => {
                        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                    }
                    Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
                };
            }
        }
    }
    if let Some(base) = &st.mediamtx {
        let start_rfc = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(start)
            .map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
            .unwrap_or_default();
        let url = format!(
            "{}/get?path={}&start={}&duration={}&format=fmp4",
            base.trim_end_matches('/'),
            percent_encoding::utf8_percent_encode(&cam.name, percent_encoding::NON_ALPHANUMERIC),
            start_rfc,
            dur / 1000
        );
        return Redirect::temporary(&url).into_response();
    }
    (
        StatusCode::NOT_FOUND,
        "no recording covers this time (enable --record for live sources, or configure --mediamtx)",
    )
        .into_response()
}

/// Triggers a graceful shutdown of a server started with [`bind`]. Cheap to clone; triggering more
/// than once, or after the server has stopped, is harmless.
#[derive(Clone)]
pub struct Shutdown(Arc<tokio::sync::watch::Sender<bool>>);

impl Shutdown {
    pub fn trigger(&self) {
        self.0.send_replace(true);
    }
}

/// A bound server that has not started yet. Await it (on a tokio runtime) to serve until
/// [`Shutdown::trigger`] is called; in-flight requests then finish and the future returns.
pub struct ServeHandle {
    shutdown: Shutdown,
    fut: futures::future::BoxFuture<'static, Result<()>>,
}

impl ServeHandle {
    pub fn shutdown(&self) -> Shutdown {
        self.shutdown.clone()
    }
}

impl std::future::IntoFuture for ServeHandle {
    type Output = Result<()>;
    type IntoFuture = futures::future::BoxFuture<'static, Result<()>>;
    fn into_future(self) -> Self::IntoFuture {
        self.fut
    }
}

/// Binds the API + UI to `addr` now (so the port is known before anything is served; use port 0
/// for any free port) and returns the bound address plus the server future. Needs no runtime;
/// the returned future must run on a tokio runtime.
pub fn bind(
    state: Arc<AppState>,
    addr: impl std::net::ToSocketAddrs,
) -> Result<(std::net::SocketAddr, ServeHandle)> {
    bind_router(router(state), addr)
}

fn bind_router(
    app: Router,
    addr: impl std::net::ToSocketAddrs,
) -> Result<(std::net::SocketAddr, ServeHandle)> {
    let std_listener = std::net::TcpListener::bind(addr)?;
    std_listener.set_nonblocking(true)?;
    let local = std_listener.local_addr()?;
    let (tx, mut rx) = tokio::sync::watch::channel(false);
    let fut = Box::pin(async move {
        let listener = tokio::net::TcpListener::from_std(std_listener)?;
        tracing::info!("retrace UI on http://{local}/");
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                // Err means every Shutdown handle is gone: nobody can ask to stop, so serve on.
                if rx.wait_for(|stop| *stop).await.is_err() {
                    std::future::pending::<()>().await;
                }
            })
            .await?;
        Ok(())
    });
    Ok((
        local,
        ServeHandle {
            shutdown: Shutdown(Arc::new(tx)),
            fut,
        },
    ))
}

/// Serves the API + UI on `bind` until the process exits (the CLI path).
pub async fn serve(state: Arc<AppState>, bind_addr: &str) -> Result<()> {
    let (_, handle) = bind(state, bind_addr)?;
    handle.await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn fetch_ping(addr: std::net::SocketAddr) -> String {
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(b"GET /ping HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        out
    }

    #[test]
    fn bind_reports_port_then_shuts_down_gracefully() {
        let app = Router::new().route("/ping", get(|| async { "pong" }));
        // No runtime exists here: bind must work synchronously and report the real port.
        let (addr, handle) = bind_router(app, "127.0.0.1:0").unwrap();
        assert_ne!(addr.port(), 0);
        let stop = handle.shutdown();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let server = tokio::spawn(handle.into_future());
            assert!(fetch_ping(addr).await.ends_with("pong"));
            stop.trigger();
            stop.trigger();
            tokio::time::timeout(Duration::from_secs(5), server)
                .await
                .expect("server stops after trigger")
                .unwrap()
                .unwrap();
        });
    }

    #[test]
    fn trigger_before_start_still_stops() {
        let app = Router::new().route("/ping", get(|| async { "pong" }));
        let (_, handle) = bind_router(app, "127.0.0.1:0").unwrap();
        handle.shutdown().trigger();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            tokio::time::timeout(Duration::from_secs(5), handle.into_future())
                .await
                .expect("server stops")
                .unwrap();
        });
    }
}
