//! A small multi-object tracker built for *sparse* analysis (2–8 fps), which is how Retrace samples
//! video: between two analysed frames a walking person can move most of a box width, so plain
//! IoU association with a one-frame confirmation window (ByteTrack defaults) drops tracks.
//!
//! Design, in order of what matters at low fps:
//! - constant-velocity prediction in pixel space (velocity is an EMA over matches);
//! - a two-part gate: IoU with the *predicted* box, or centre distance relative to box size;
//! - ByteTrack-style two-stage association: confident detections first, then the leftover tracks
//!   get a second chance with low-confidence detections (recovers partial occlusions);
//! - tentative tracks live for a few frames before they are dropped, confirmed after `min_hits`;
//! - greedy assignment by cost: a frame rarely has more than a few dozen objects.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Det {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub conf: f32,
    pub class: String,
}

#[derive(Debug, Clone)]
pub struct Tracked {
    pub id: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub conf: f32,
    pub class: String,
    /// False until the track has been seen `min_hits` times.
    pub confirmed: bool,
}

#[derive(Debug, Clone)]
struct Track {
    id: usize,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    vx: f32,
    vy: f32,
    conf: f32,
    class_votes: HashMap<String, usize>,
    last_class: String,
    last_seen: usize,
    hits: usize,
}

impl Track {
    fn predicted(&self, frame: usize) -> (f32, f32, f32, f32) {
        let dt = frame.saturating_sub(self.last_seen) as f32;
        (self.x + self.vx * dt, self.y + self.vy * dt, self.w, self.h)
    }
    fn class(&self) -> &str {
        self.class_votes
            .iter()
            .max_by_key(|(_, n)| **n)
            .map(|(c, _)| c.as_str())
            .unwrap_or(&self.last_class)
    }
}

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Frames a confirmed track survives without a detection.
    pub max_age: usize,
    /// Frames a tentative (unconfirmed) track survives without a detection.
    pub max_age_tentative: usize,
    /// Detections needed before a track is reported.
    pub min_hits: usize,
    /// Detections at or above this confidence drive the first association stage.
    pub high_conf: f32,
    /// Minimum IoU between the predicted box and a detection to match (stage 1).
    pub min_iou: f32,
    /// Alternative gate: centre distance / max(box w, h) must be below this.
    pub max_center_dist: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_age: 8,
            max_age_tentative: 3,
            min_hits: 2,
            high_conf: 0.35,
            min_iou: 0.05,
            max_center_dist: 0.9,
        }
    }
}

#[derive(Debug)]
pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: usize,
    frame: usize,
}

fn group(class: &str) -> u8 {
    match class {
        "person" => 0,
        "bicycle" | "motorcycle" => 1,
        _ => 2,
    }
}

fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = (a.0 + a.2).min(b.0 + b.2);
    let y1 = (a.1 + a.3).min(b.1 + b.3);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let union = a.2 * a.3 + b.2 * b.3 - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn center_dist_rel(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax, ay) = (a.0 + a.2 / 2.0, a.1 + a.3 / 2.0);
    let (bx, by) = (b.0 + b.2 / 2.0, b.1 + b.3 / 2.0);
    let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
    d / a.2.max(a.3).max(1.0)
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            next_id: 1,
            frame: 0,
        }
    }

    /// Cost of matching track `t` to detection `d` at the current frame, or None when gated out.
    fn cost(&self, t: &Track, d: &Det) -> Option<f32> {
        if group(t.class()) != group(&d.class) {
            return None;
        }
        let p = t.predicted(self.frame);
        let b = (d.x, d.y, d.w, d.h);
        let i = iou(p, b);
        let c = center_dist_rel(p, b);
        if i >= self.cfg.min_iou || c <= self.cfg.max_center_dist {
            // Mostly IoU, with a distance term so that of two overlapping candidates the nearer wins.
            Some((1.0 - i) + 0.25 * c.min(2.0))
        } else {
            None
        }
    }

    /// Greedy assignment of `tracks` (indices into self.tracks) to `dets` (indices into the slice).
    fn assign(&self, tracks: &[usize], dets: &[usize], all: &[Det]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for &ti in tracks {
            for &di in dets {
                if let Some(c) = self.cost(&self.tracks[ti], &all[di]) {
                    pairs.push((c, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut used_t = std::collections::HashSet::new();
        let mut used_d = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (_, ti, di) in pairs {
            if used_t.contains(&ti) || used_d.contains(&di) {
                continue;
            }
            used_t.insert(ti);
            used_d.insert(di);
            out.push((ti, di));
        }
        out
    }

    fn apply(&mut self, ti: usize, d: &Det) {
        let frame = self.frame;
        let t = &mut self.tracks[ti];
        let dt = frame.saturating_sub(t.last_seen).max(1) as f32;
        let (nvx, nvy) = ((d.x - t.x) / dt, (d.y - t.y) / dt);
        if t.hits == 0 {
            t.vx = nvx;
            t.vy = nvy;
        } else {
            t.vx = 0.6 * t.vx + 0.4 * nvx;
            t.vy = 0.6 * t.vy + 0.4 * nvy;
        }
        t.x = d.x;
        t.y = d.y;
        t.w = 0.5 * t.w + 0.5 * d.w;
        t.h = 0.5 * t.h + 0.5 * d.h;
        t.conf = d.conf;
        t.last_seen = frame;
        t.hits += 1;
        *t.class_votes.entry(d.class.clone()).or_insert(0) += 1;
        t.last_class = d.class.clone();
    }

    /// Advances one frame with this frame's detections and returns the tracks seen in it.
    pub fn update(&mut self, dets: &[Det]) -> Vec<Tracked> {
        self.frame += 1;
        let frame = self.frame;
        let (high, low): (Vec<usize>, Vec<usize>) =
            (0..dets.len()).partition(|&i| dets[i].conf >= self.cfg.high_conf);
        let all_tracks: Vec<usize> = (0..self.tracks.len()).collect();

        // Stage 1: every track vs confident detections.
        let m1 = self.assign(&all_tracks, &high, dets);
        let mut matched_t: std::collections::HashSet<usize> = m1.iter().map(|m| m.0).collect();
        let mut matched_d: std::collections::HashSet<usize> = m1.iter().map(|m| m.1).collect();
        for (ti, di) in &m1 {
            self.apply(*ti, &dets[*di]);
        }
        // Stage 2: leftover tracks vs low-confidence detections (occlusion, motion blur).
        let rest_t: Vec<usize> = all_tracks
            .iter()
            .cloned()
            .filter(|t| !matched_t.contains(t))
            .collect();
        let m2 = self.assign(&rest_t, &low, dets);
        for (ti, di) in &m2 {
            self.apply(*ti, &dets[*di]);
            matched_t.insert(*ti);
            matched_d.insert(*di);
        }
        // New tentative tracks from unmatched confident detections.
        for &di in &high {
            if matched_d.contains(&di) {
                continue;
            }
            let d = &dets[di];
            let mut votes = HashMap::new();
            votes.insert(d.class.clone(), 1);
            self.tracks.push(Track {
                id: self.next_id,
                x: d.x,
                y: d.y,
                w: d.w,
                h: d.h,
                vx: 0.0,
                vy: 0.0,
                conf: d.conf,
                class_votes: votes,
                last_class: d.class.clone(),
                last_seen: frame,
                hits: 1,
            });
            self.next_id += 1;
        }
        // Age out.
        let cfg = &self.cfg;
        self.tracks.retain(|t| {
            let age = frame - t.last_seen;
            if t.hits >= cfg.min_hits {
                age <= cfg.max_age
            } else {
                age <= cfg.max_age_tentative
            }
        });
        // Report tracks seen in this frame.
        self.tracks
            .iter()
            .filter(|t| t.last_seen == frame)
            .map(|t| Tracked {
                id: t.id,
                x: t.x,
                y: t.y,
                w: t.w,
                h: t.h,
                conf: t.conf,
                class: t.class().to_string(),
                confirmed: t.hits >= self.cfg.min_hits,
            })
            .collect()
    }

    /// Confirmed tracks currently alive (seen within max_age), for the preview.
    pub fn alive(&self) -> usize {
        self.tracks
            .iter()
            .filter(|t| t.hits >= self.cfg.min_hits)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, conf: f32) -> Det {
        Det {
            x,
            y,
            w: 40.0,
            h: 100.0,
            conf,
            class: "person".into(),
        }
    }

    #[test]
    fn keeps_identity_across_large_steps() {
        // A person moving 30 px per analysed frame (IoU between frames ≈ 0.14).
        let mut t = Tracker::new(TrackerConfig::default());
        let mut ids = Vec::new();
        for i in 0..8 {
            let out = t.update(&[det(100.0 + 30.0 * i as f32, 200.0, 0.8)]);
            assert_eq!(out.len(), 1);
            ids.push(out[0].id);
        }
        assert!(ids.iter().all(|&i| i == ids[0]), "ids: {ids:?}");
    }

    #[test]
    fn survives_a_missed_frame_and_low_conf() {
        let mut t = Tracker::new(TrackerConfig::default());
        let a = t.update(&[det(100.0, 200.0, 0.9)])[0].id;
        t.update(&[det(120.0, 200.0, 0.9)]);
        assert!(t.update(&[]).is_empty());
        // Comes back as a low-confidence detection: stage 2 must recover it.
        let out = t.update(&[det(160.0, 200.0, 0.2)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, a);
    }

    #[test]
    fn separate_people_get_separate_ids() {
        let mut t = Tracker::new(TrackerConfig::default());
        let out = t.update(&[det(100.0, 200.0, 0.9), det(400.0, 200.0, 0.9)]);
        assert_eq!(out.len(), 2);
        let out = t.update(&[det(110.0, 200.0, 0.9), det(390.0, 200.0, 0.9)]);
        assert_eq!(out.len(), 2);
        assert_ne!(out[0].id, out[1].id);
        assert!(out.iter().all(|o| o.confirmed));
    }
}
