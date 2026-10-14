//! Standing queries: a rule is a text embedding plus scalar constraints, evaluated against every
//! new tracklet at ingest time. This is what an "AI alert" is underneath.

use crate::db::{Db, NewTracklet, Rule};
use crate::models::cosine;
use anyhow::Result;
use std::time::{Duration, Instant};

pub struct RuleEngine {
    rules: Vec<Rule>,
    loaded: Option<Instant>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            rules: vec![],
            loaded: None,
        }
    }

    fn refresh(&mut self, db: &Db) -> Result<()> {
        let due = self
            .loaded
            .map(|t| t.elapsed() > Duration::from_secs(3))
            .unwrap_or(true);
        if due {
            self.rules = db.rules(true)?;
            self.loaded = Some(Instant::now());
        }
        Ok(())
    }

    /// Returns the number of alerts fired for this tracklet.
    pub fn evaluate(
        &mut self,
        db: &Db,
        tracklet_id: i64,
        t: &NewTracklet,
        cam_name: &str,
        already: &mut std::collections::HashSet<i64>,
    ) -> Result<usize> {
        self.refresh(db)?;
        let mut fired = 0;
        for r in &self.rules {
            if already.contains(&r.id) || r.emb.len() != t.emb.len() {
                continue;
            }
            if let Some(c) = &r.class
                && c != &t.class
            {
                continue;
            }
            if let Some(c) = r.cam
                && c != t.cam
            {
                continue;
            }
            if t.dwell_s < r.min_dwell_s {
                continue;
            }
            let score = cosine(&r.emb, &t.emb);
            if score >= r.min_score {
                let msg = format!(
                    "{} on {} matched \"{}\" (score {:.3}, dwell {:.0}s, {})",
                    t.class, cam_name, r.text, score, t.dwell_s, t.direction
                );
                db.insert_alert(r.id, tracklet_id, t.t1, score, &msg)?;
                already.insert(r.id);
                tracing::info!(rule = %r.name, tracklet = tracklet_id, score = format!("{score:.3}"), "ALERT {msg}");
                fired += 1;
            }
        }
        Ok(fired)
    }
}

/// Backtest a rule against existing tracklets: how often would it have fired?
pub fn backtest(
    db: &Db,
    emb: &[f32],
    min_score: f64,
    class: Option<&str>,
    cam: Option<i64>,
    min_dwell_s: f64,
    t0: i64,
    t1: i64,
) -> Result<serde_json::Value> {
    let hits = db.knn(emb, 500, cam, class, Some(t0), Some(t1))?;
    let mut matched = Vec::new();
    for (id, score) in hits {
        if score < min_score {
            continue;
        }
        if let Some(t) = db.tracklet(id)? {
            if t.dwell_s < min_dwell_s {
                continue;
            }
            matched.push(serde_json::json!({"id": id, "score": score, "t0": t.t0, "cam": t.cam_name, "class": t.class}));
        }
    }
    let span_days = ((t1 - t0) as f64 / 86_400_000.0).max(1.0 / 24.0);
    Ok(
        serde_json::json!({ "matches": matched.len(), "per_day": matched.len() as f64 / span_days, "examples": matched.iter().take(12).collect::<Vec<_>>() }),
    )
}
