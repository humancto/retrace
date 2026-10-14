//! Query pipeline: rewrite → embed → filtered exact KNN → caption BM25 → reciprocal-rank fusion →
//! dynamic threshold. Mirrors the incumbent's design (crop vectors, filter-then-exact-search,
//! per-query confidence band) with captions as a second signal.

use crate::db::{AudioHit, Db, Hit};
use crate::models::Embedder;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SearchParams {
    pub q: String,
    pub cam: Option<String>,
    pub class: Option<String>,
    pub t0: Option<i64>,
    pub t1: Option<i64>,
    pub k: Option<usize>,
    pub direction: Option<String>,
    pub min_dwell_s: Option<f64>,
    pub entry: Option<String>,
    pub exit: Option<String>,
    /// "all" returns every candidate; default applies the dynamic threshold.
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub rewritten: String,
    pub plan: serde_json::Value,
    pub hits: Vec<Hit>,
    /// Audio windows that matched (empty unless the embedder has an audio encoder).
    pub audio_hits: Vec<AudioHit>,
    pub candidates: usize,
    pub threshold: f64,
    pub timing_ms: HashMap<&'static str, f64>,
}

/// Cheap query understanding: an obvious subject word becomes a class filter (the LLM parser in a
/// later phase replaces this). Vehicle words map to COCO's vehicle classes via a wildcard.
pub fn infer_class(q: &str) -> Option<&'static str> {
    let w: Vec<String> = q
        .to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    let has = |xs: &[&str]| w.iter().any(|t| xs.contains(&t.as_str()));
    if has(&["car", "sedan", "suv", "hatchback", "taxi"]) {
        return Some("car");
    }
    if has(&["truck", "van", "pickup", "lorry"]) {
        return Some("truck");
    }
    if has(&["bus"]) {
        return Some("bus");
    }
    if has(&["bicycle", "bike", "cyclist"]) {
        return Some("bicycle");
    }
    if has(&["motorcycle", "motorbike", "scooter"]) {
        return Some("motorcycle");
    }
    if has(&[
        "person",
        "people",
        "man",
        "woman",
        "someone",
        "somebody",
        "pedestrian",
        "guy",
        "girl",
        "boy",
        "kid",
        "child",
        "adult",
        "worker",
    ]) {
        return Some("person");
    }
    None
}

pub fn rewrite(q: &str) -> String {
    let t = q.trim();
    let low = t.to_ascii_lowercase();
    if low.starts_with("a photo") || low.starts_with("a picture") || low.starts_with("an image") {
        t.to_string()
    } else {
        format!("a photo of {t}")
    }
}

/// Per-query confidence band: keep scores >= max(top - delta, mean + sigma_k * std), at least `min_keep`.
pub fn dynamic_threshold(scores: &[f64], delta: f64, sigma_k: f64, min_keep: usize) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let top = scores.iter().cloned().fold(f64::MIN, f64::max);
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let thr = (top - delta).max(mean + sigma_k * std);
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let floor = sorted
        .get(min_keep.saturating_sub(1))
        .cloned()
        .unwrap_or(sorted[sorted.len() - 1]);
    thr.min(floor)
}

pub fn search(db: &Db, emb: &mut Embedder, p: &SearchParams) -> Result<SearchResponse> {
    let mut timing = HashMap::new();
    let t_all = Instant::now();
    let k = p.k.unwrap_or(60).clamp(1, 500);

    let t = Instant::now();
    let (rewritten, qv) = emb.embed_query(&p.q)?;
    timing.insert("embed", t.elapsed().as_secs_f64() * 1000.0);

    let cam_id = match &p.cam {
        Some(name) if !name.is_empty() => match db.camera_by_name(name)? {
            Some(c) => Some(c.id),
            None => anyhow::bail!("unknown camera '{name}'"),
        },
        _ => None,
    };
    let inferred = infer_class(&p.q);
    let class = p
        .class
        .as_deref()
        .filter(|c| !c.is_empty() && *c != "any")
        .or(inferred);

    let t = Instant::now();
    let knn = db.knn(&qv, k.max(40), cam_id, class, p.t0, p.t1)?;
    timing.insert("knn", t.elapsed().as_secs_f64() * 1000.0);

    let t = Instant::now();
    let fts = db.fts(&p.q, k)?;
    timing.insert("fts", t.elapsed().as_secs_f64() * 1000.0);

    // Reciprocal-rank fusion: vector rank dominates; captions add a second vote when present.
    let mut fused: HashMap<i64, (f64, Option<usize>, Option<usize>, f64)> = HashMap::new();
    for (rank, (id, sim)) in knn.iter().enumerate() {
        fused.insert(
            *id,
            (1.0 / (60.0 + rank as f64), Some(rank + 1), None, *sim),
        );
    }
    for (rank, id) in fts.iter().enumerate() {
        let e = fused.entry(*id).or_insert((0.0, None, None, 0.0));
        e.0 += 0.6 / (60.0 + rank as f64);
        e.2 = Some(rank + 1);
    }

    let t = Instant::now();
    let mut hits: Vec<Hit> = Vec::new();
    for (id, (fscore, vr, tr, sim)) in fused {
        let Some(row) = db.tracklet(id)? else {
            continue;
        };
        if let Some(d) = &p.direction
            && !d.is_empty()
            && d != "any"
            && &row.direction != d
        {
            continue;
        }
        if let Some(m) = p.min_dwell_s
            && row.dwell_s < m
        {
            continue;
        }
        if let Some(e) = &p.entry
            && !e.is_empty()
            && e != "any"
            && &row.entry_edge != e
        {
            continue;
        }
        if let Some(e) = &p.exit
            && !e.is_empty()
            && e != "any"
            && &row.exit_edge != e
        {
            continue;
        }
        // Score shown to the user is the cosine similarity when we have one, else the fused score scaled.
        let score = if vr.is_some() { sim } else { fscore * 10.0 };
        hits.push(Hit {
            tracklet: row,
            score,
            vec_rank: vr,
            text_rank: tr,
        });
    }
    hits.sort_by(|a, b| {
        let fa = a.vec_rank.map(|r| 1.0 / (60.0 + r as f64)).unwrap_or(0.0)
            + a.text_rank.map(|r| 0.6 / (60.0 + r as f64)).unwrap_or(0.0);
        let fb = b.vec_rank.map(|r| 1.0 / (60.0 + r as f64)).unwrap_or(0.0)
            + b.text_rank.map(|r| 0.6 / (60.0 + r as f64)).unwrap_or(0.0);
        fb.total_cmp(&fa)
    });
    let candidates = hits.len();
    let sims: Vec<f64> = hits
        .iter()
        .filter(|h| h.vec_rank.is_some())
        .map(|h| h.score)
        .collect();
    let threshold = dynamic_threshold(&sims, 0.06, 1.0, 5);
    if p.mode.as_deref() != Some("all") {
        hits.retain(|h| h.vec_rank.is_none() || h.score >= threshold);
    }
    hits.truncate(k);
    timing.insert("fuse", t.elapsed().as_secs_f64() * 1000.0);

    // Audio windows share the vector space with the crops when the embedder is multimodal. They get
    // their own confidence band: sound and image similarities are not on the same scale.
    let mut audio_hits = Vec::new();
    let explicit_class = p
        .class
        .as_deref()
        .map(|c| !c.is_empty() && c != "any")
        .unwrap_or(false);
    if emb.audio && !explicit_class {
        let t = Instant::now();
        let mut a = db.knn_audio(&qv, 30, cam_id, p.t0, p.t1)?;
        let scores: Vec<f64> = a.iter().map(|h| h.score).collect();
        let thr = dynamic_threshold(&scores, 0.05, 1.0, 3);
        if p.mode.as_deref() != Some("all") {
            a.retain(|h| h.score >= thr);
        }
        // Collapse overlapping windows (hop < win): keep the best of each run.
        a.sort_by(|x, y| x.cam.cmp(&y.cam).then(x.t0.cmp(&y.t0)));
        let mut merged: Vec<AudioHit> = Vec::new();
        for h in a {
            if let Some(last) = merged.last_mut()
                && last.cam == h.cam
                && h.t0 <= last.t1
            {
                if h.score > last.score {
                    last.score = h.score;
                    last.id = h.id;
                }
                last.t1 = last.t1.max(h.t1);
                continue;
            }
            merged.push(h);
        }
        merged.sort_by(|x, y| y.score.total_cmp(&x.score));
        merged.truncate(12);
        audio_hits = merged;
        timing.insert("audio", t.elapsed().as_secs_f64() * 1000.0);
    }
    timing.insert("total", t_all.elapsed().as_secs_f64() * 1000.0);

    let plan = serde_json::json!({
        "semantic": rewritten,
        "filters": {"cam": p.cam, "class": class, "t0": p.t0, "t1": p.t1, "direction": p.direction, "min_dwell_s": p.min_dwell_s, "entry": p.entry, "exit": p.exit},
        "engine": "sqlite-vec exact cosine KNN + FTS5 BM25, RRF",
        "embedder": emb.name,
        "audio": emb.audio,
        "k": k,
        "threshold": threshold,
    });
    Ok(SearchResponse {
        query: p.q.clone(),
        rewritten,
        plan,
        hits,
        audio_hits,
        candidates,
        threshold,
        timing_ms: timing,
    })
}
