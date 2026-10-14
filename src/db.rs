//! SQLite storage: tracklets, crop vectors (sqlite-vec), captions (FTS5), events, rules, alerts.
//! One file, WAL mode, one connection per thread.

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use zerocopy::IntoBytes;

static VEC_INIT: std::sync::Once = std::sync::Once::new();

pub fn register_vec() {
    VEC_INIT.call_once(|| unsafe {
        // sqlite-vec registers itself as an auto-extension on every new connection.
        type Init = unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3,
            *mut *mut std::os::raw::c_char,
            *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int;
        let f: Init = std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ());
        rusqlite::ffi::sqlite3_auto_extension(Some(f));
    });
}

pub struct Db {
    pub conn: Connection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub id: i64,
    pub name: String,
    pub kind: String,
    pub source: String,
    pub width: i64,
    pub height: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub id: i64,
    pub cam: i64,
    pub path: String,
    pub t_start: i64,
    pub duration_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackletRow {
    pub id: i64,
    pub cam: i64,
    pub cam_name: String,
    pub class: String,
    pub t0: i64,
    pub t1: i64,
    pub path_json: String,
    pub entry_edge: String,
    pub exit_edge: String,
    pub dwell_s: f64,
    pub speed: f64,
    pub direction: String,
    pub best_crops_json: String,
    pub n_frames: i64,
    pub caption_json: Option<String>,
    pub caption_text: Option<String>,
    pub bbox_json: String,
    pub frame_w: i64,
    pub frame_h: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewTracklet {
    pub cam: i64,
    pub class: String,
    pub t0: i64,
    pub t1: i64,
    pub path_json: String,
    pub entry_edge: String,
    pub exit_edge: String,
    pub dwell_s: f64,
    pub speed: f64,
    pub direction: String,
    pub best_crops_json: String,
    pub n_frames: i64,
    pub bbox_json: String,
    pub frame_w: i64,
    pub frame_h: i64,
    pub emb: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub id: i64,
    pub name: String,
    pub text: String,
    pub min_score: f64,
    pub class: Option<String>,
    pub cam: Option<i64>,
    pub min_dwell_s: f64,
    pub enabled: bool,
    #[serde(skip_serializing)]
    pub emb: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: i64,
    pub rule_id: i64,
    pub rule_name: String,
    pub tracklet_id: i64,
    pub ts: i64,
    pub score: f64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: i64,
    pub cam: Option<i64>,
    pub ts: i64,
    pub source: String,
    pub r#type: String,
    pub attrs: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub tracklet: TrackletRow,
    pub score: f64,
    pub vec_rank: Option<usize>,
    pub text_rank: Option<usize>,
}

/// An audio window that matched a query (multimodal embedder only).
#[derive(Debug, Clone, Serialize)]
pub struct AudioHit {
    pub id: i64,
    pub cam: i64,
    pub cam_name: String,
    pub t0: i64,
    pub t1: i64,
    pub score: f64,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        register_vec();
        let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS cameras(id INTEGER PRIMARY KEY, name TEXT UNIQUE, kind TEXT, source TEXT, width INTEGER DEFAULT 0, height INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, cam INTEGER, path TEXT, t_start INTEGER, duration_ms INTEGER);
            CREATE TABLE IF NOT EXISTS tracklets(
                id INTEGER PRIMARY KEY, cam INTEGER, class TEXT, t0 INTEGER, t1 INTEGER,
                path_json TEXT, entry_edge TEXT, exit_edge TEXT, dwell_s REAL, speed REAL, direction TEXT,
                best_crops_json TEXT, n_frames INTEGER, bbox_json TEXT, frame_w INTEGER, frame_h INTEGER,
                caption_json TEXT, caption_text TEXT, created INTEGER);
            CREATE INDEX IF NOT EXISTS tracklets_cam_t0 ON tracklets(cam, t0);
            CREATE VIRTUAL TABLE IF NOT EXISTS captions USING fts5(tracklet_id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, cam INTEGER, ts INTEGER, source TEXT, type TEXT, attrs_json TEXT);
            CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
            CREATE TABLE IF NOT EXISTS rules(id INTEGER PRIMARY KEY, name TEXT, text TEXT, min_score REAL, class TEXT, cam INTEGER, min_dwell_s REAL DEFAULT 0, enabled INTEGER DEFAULT 1, emb_json TEXT, created INTEGER);
            CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY, rule_id INTEGER, tracklet_id INTEGER, ts INTEGER, score REAL, message TEXT);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts INTEGER, action TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS audio_windows(id INTEGER PRIMARY KEY, cam INTEGER, t0 INTEGER, t1 INTEGER, rms REAL, created INTEGER);
            CREATE INDEX IF NOT EXISTS audio_windows_cam_t ON audio_windows(cam, t0);
            "#,
        )?;
        Ok(())
    }

    /// Creates the vector table for a given embedding dimension (once); records the model name.
    pub fn ensure_vec_table(&self, dim: usize, model: &str) -> Result<()> {
        if let Some(existing) = self.meta("emb_dim")? {
            let d: usize = existing.parse().unwrap_or(0);
            anyhow::ensure!(
                d == dim,
                "index was built with dim {d} ({}), got {dim} ({model}); use a fresh --db",
                self.meta("emb_model")?.unwrap_or_default()
            );
            return Ok(());
        }
        self.conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec_tracklets USING vec0(
                id INTEGER PRIMARY KEY,
                emb FLOAT[{dim}] distance_metric=cosine,
                cam INTEGER, class TEXT, t0 INTEGER, t1 INTEGER
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_audio USING vec0(
                id INTEGER PRIMARY KEY,
                emb FLOAT[{dim}] distance_metric=cosine,
                cam INTEGER, t0 INTEGER, t1 INTEGER
            );"
        ))?;
        self.set_meta("emb_dim", &dim.to_string())?;
        self.set_meta("emb_model", model)?;
        Ok(())
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute("INSERT INTO meta(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value])?;
        Ok(())
    }

    pub fn upsert_camera(
        &self,
        name: &str,
        kind: &str,
        source: &str,
        w: u32,
        h: u32,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO cameras(name, kind, source, width, height) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(name) DO UPDATE SET kind=excluded.kind, source=excluded.source,
             width=CASE WHEN excluded.width>0 THEN excluded.width ELSE cameras.width END,
             height=CASE WHEN excluded.height>0 THEN excluded.height ELSE cameras.height END",
            params![name, kind, source, w, h],
        )?;
        Ok(self
            .conn
            .query_row("SELECT id FROM cameras WHERE name=?1", [name], |r| r.get(0))?)
    }

    pub fn cameras(&self) -> Result<Vec<Camera>> {
        let mut st = self
            .conn
            .prepare("SELECT id,name,kind,source,width,height FROM cameras ORDER BY id")?;
        let rows = st.query_map([], |r| {
            Ok(Camera {
                id: r.get(0)?,
                name: r.get(1)?,
                kind: r.get(2)?,
                source: r.get(3)?,
                width: r.get(4)?,
                height: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn camera_by_name(&self, name: &str) -> Result<Option<Camera>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id,name,kind,source,width,height FROM cameras WHERE name=?1",
                [name],
                |r| {
                    Ok(Camera {
                        id: r.get(0)?,
                        name: r.get(1)?,
                        kind: r.get(2)?,
                        source: r.get(3)?,
                        width: r.get(4)?,
                        height: r.get(5)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn add_source(&self, cam: i64, path: &str, t_start: i64, duration_ms: i64) -> Result<i64> {
        self.conn.execute(
            "DELETE FROM sources WHERE cam=?1 AND path=?2",
            params![cam, path],
        )?;
        self.conn.execute(
            "INSERT INTO sources(cam,path,t_start,duration_ms) VALUES(?1,?2,?3,?4)",
            params![cam, path, t_start, duration_ms],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn source_for(&self, cam: i64, t: i64) -> Result<Option<SourceFile>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id,cam,path,t_start,duration_ms FROM sources WHERE cam=?1 AND t_start<=?2 AND t_start+duration_ms>=?2 ORDER BY t_start DESC LIMIT 1",
                params![cam, t],
                |r| Ok(SourceFile { id: r.get(0)?, cam: r.get(1)?, path: r.get(2)?, t_start: r.get(3)?, duration_ms: r.get(4)? }),
            )
            .optional()?)
    }

    pub fn insert_tracklet(&self, t: &NewTracklet) -> Result<i64> {
        let now = chrono::Utc::now().timestamp_millis();
        self.conn.execute(
            "INSERT INTO tracklets(cam,class,t0,t1,path_json,entry_edge,exit_edge,dwell_s,speed,direction,best_crops_json,n_frames,bbox_json,frame_w,frame_h,created)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
            params![t.cam, t.class, t.t0, t.t1, t.path_json, t.entry_edge, t.exit_edge, t.dwell_s, t.speed, t.direction, t.best_crops_json, t.n_frames, t.bbox_json, t.frame_w, t.frame_h, now],
        )?;
        let id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO vec_tracklets(id, emb, cam, class, t0, t1) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, t.emb.as_bytes(), t.cam, t.class, t.t0, t.t1],
        )?;
        Ok(id)
    }

    /// Rewrites a provisional tracklet (and its vector) with the current state of the track.
    pub fn update_tracklet(&self, id: i64, t: &NewTracklet) -> Result<()> {
        self.conn.execute(
            "UPDATE tracklets SET class=?2, t1=?3, path_json=?4, entry_edge=?5, exit_edge=?6, dwell_s=?7, speed=?8, direction=?9, best_crops_json=?10, n_frames=?11, bbox_json=?12 WHERE id=?1",
            params![id, t.class, t.t1, t.path_json, t.entry_edge, t.exit_edge, t.dwell_s, t.speed, t.direction, t.best_crops_json, t.n_frames, t.bbox_json],
        )?;
        self.conn
            .execute("DELETE FROM vec_tracklets WHERE id=?1", [id])?;
        self.conn.execute(
            "INSERT INTO vec_tracklets(id, emb, cam, class, t0, t1) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, t.emb.as_bytes(), t.cam, t.class, t.t0, t.t1],
        )?;
        Ok(())
    }

    pub fn tracklet(&self, id: i64) -> Result<Option<TrackletRow>> {
        let mut st = self
            .conn
            .prepare(&format!("{} WHERE t.id=?1", TRACKLET_SELECT))?;
        Ok(st.query_row([id], row_to_tracklet).optional()?)
    }

    pub fn tracklets_in(
        &self,
        cam: Option<i64>,
        t0: i64,
        t1: i64,
        limit: usize,
    ) -> Result<Vec<TrackletRow>> {
        let mut sql = format!("{} WHERE t.t1>=?1 AND t.t0<=?2", TRACKLET_SELECT);
        if cam.is_some() {
            sql.push_str(" AND t.cam=?3");
        }
        sql.push_str(" ORDER BY t.t0 LIMIT ?4");
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map(
            params![t0, t1, cam.unwrap_or(0), limit as i64],
            row_to_tracklet,
        )?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn recent_tracklets(&self, since_id: i64, limit: usize) -> Result<Vec<TrackletRow>> {
        let mut st = self.conn.prepare(&format!(
            "{} WHERE t.id>?1 ORDER BY t.id DESC LIMIT ?2",
            TRACKLET_SELECT
        ))?;
        let rows = st.query_map(params![since_id, limit as i64], row_to_tracklet)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn tracklet_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM tracklets", [], |r| r.get(0))?)
    }

    pub fn uncaptioned(&self, limit: usize) -> Result<Vec<TrackletRow>> {
        let mut st = self.conn.prepare(&format!(
            "{} WHERE t.caption_json IS NULL ORDER BY t.id DESC LIMIT ?1",
            TRACKLET_SELECT
        ))?;
        let rows = st.query_map(params![limit as i64], row_to_tracklet)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn set_caption(&self, id: i64, caption_json: &str, caption_text: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE tracklets SET caption_json=?2, caption_text=?3 WHERE id=?1",
            params![id, caption_json, caption_text],
        )?;
        self.conn
            .execute("DELETE FROM captions WHERE tracklet_id=?1", params![id])?;
        self.conn.execute(
            "INSERT INTO captions(tracklet_id, text) VALUES(?1, ?2)",
            params![id, caption_text],
        )?;
        Ok(())
    }

    pub fn emb_of(&self, id: i64) -> Result<Option<Vec<f32>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row("SELECT emb FROM vec_tracklets WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .optional()?;
        Ok(blob.map(|b| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }))
    }

    /// Exact cosine KNN over the crop vectors with scalar pre-filters. Returns (id, cosine similarity).
    pub fn knn(
        &self,
        q: &[f32],
        k: usize,
        cam: Option<i64>,
        class: Option<&str>,
        t0: Option<i64>,
        t1: Option<i64>,
    ) -> Result<Vec<(i64, f64)>> {
        let mut sql =
            String::from("SELECT id, distance FROM vec_tracklets WHERE emb MATCH ?1 AND k = ?2");
        let mut args: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(q.as_bytes().to_vec()), Box::new(k as i64)];
        if let Some(c) = cam {
            sql.push_str(&format!(" AND cam = ?{}", args.len() + 1));
            args.push(Box::new(c));
        }
        if let Some(c) = class {
            sql.push_str(&format!(" AND class = ?{}", args.len() + 1));
            args.push(Box::new(c.to_string()));
        }
        if let Some(t) = t0 {
            sql.push_str(&format!(" AND t1 >= ?{}", args.len() + 1));
            args.push(Box::new(t));
        }
        if let Some(t) = t1 {
            sql.push_str(&format!(" AND t0 <= ?{}", args.len() + 1));
            args.push(Box::new(t));
        }
        let mut st = self.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = st.query_map(refs.as_slice(), |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (id, dist) = r?;
            out.push((id, 1.0 - dist));
        }
        Ok(out)
    }

    /// Stores one embedded audio window (same vector space as the crops when the embedder is multimodal).
    pub fn insert_audio_window(
        &self,
        cam: i64,
        t0: i64,
        t1: i64,
        rms: f64,
        emb: &[f32],
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO audio_windows(cam, t0, t1, rms, created) VALUES(?1, ?2, ?3, ?4, ?5)",
            params![cam, t0, t1, rms, chrono::Utc::now().timestamp_millis()],
        )?;
        let id = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO vec_audio(id, emb, cam, t0, t1) VALUES(?1, ?2, ?3, ?4, ?5)",
            params![id, emb.as_bytes().to_vec(), cam, t0, t1],
        )?;
        Ok(id)
    }

    pub fn audio_window_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM audio_windows", [], |r| r.get(0))?)
    }

    /// Exact cosine KNN over audio windows. Returns (id, cam, t0, t1, similarity).
    pub fn knn_audio(
        &self,
        q: &[f32],
        k: usize,
        cam: Option<i64>,
        t0: Option<i64>,
        t1: Option<i64>,
    ) -> Result<Vec<AudioHit>> {
        if self.audio_window_count()? == 0 {
            return Ok(vec![]);
        }
        let mut sql = String::from(
            "SELECT id, cam, t0, t1, distance FROM vec_audio WHERE emb MATCH ?1 AND k = ?2",
        );
        let mut args: Vec<Box<dyn rusqlite::ToSql>> =
            vec![Box::new(q.as_bytes().to_vec()), Box::new(k as i64)];
        if let Some(c) = cam {
            sql.push_str(&format!(" AND cam = ?{}", args.len() + 1));
            args.push(Box::new(c));
        }
        if let Some(t) = t0 {
            sql.push_str(&format!(" AND t1 >= ?{}", args.len() + 1));
            args.push(Box::new(t));
        }
        if let Some(t) = t1 {
            sql.push_str(&format!(" AND t0 <= ?{}", args.len() + 1));
            args.push(Box::new(t));
        }
        let mut st = self.conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|b| b.as_ref()).collect();
        let rows = st.query_map(refs.as_slice(), |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, f64>(4)?,
            ))
        })?;
        let cams = self.cameras()?;
        let mut out = Vec::new();
        for r in rows {
            let (id, cam, t0, t1, dist) = r?;
            let cam_name = cams
                .iter()
                .find(|c| c.id == cam)
                .map(|c| c.name.clone())
                .unwrap_or_default();
            out.push(AudioHit {
                id,
                cam,
                cam_name,
                t0,
                t1,
                score: 1.0 - dist,
            });
        }
        Ok(out)
    }

    /// BM25 over VLM captions. Returns tracklet ids best-first.
    pub fn fts(&self, query: &str, limit: usize) -> Result<Vec<i64>> {
        // Build a forgiving OR query from the words; FTS5 syntax errors must not break search.
        let terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(|w| format!("\"{}\"", w.to_lowercase()))
            .collect();
        if terms.is_empty() {
            return Ok(vec![]);
        }
        let q = terms.join(" OR ");
        let mut st = self.conn.prepare("SELECT tracklet_id FROM captions WHERE captions MATCH ?1 ORDER BY bm25(captions) LIMIT ?2")?;
        let rows = st.query_map(params![q, limit as i64], |r| r.get::<_, i64>(0));
        match rows {
            Ok(rows) => Ok(rows.filter_map(|r| r.ok()).collect()),
            Err(_) => Ok(vec![]),
        }
    }

    pub fn insert_event(
        &self,
        cam: Option<i64>,
        ts: i64,
        source: &str,
        typ: &str,
        attrs: &serde_json::Value,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO events(cam,ts,source,type,attrs_json) VALUES(?1,?2,?3,?4,?5)",
            params![cam, ts, source, typ, attrs.to_string()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn events(&self, t0: i64, t1: i64, limit: usize) -> Result<Vec<Event>> {
        let mut st = self.conn.prepare("SELECT id,cam,ts,source,type,attrs_json FROM events WHERE ts BETWEEN ?1 AND ?2 ORDER BY ts DESC LIMIT ?3")?;
        let rows = st.query_map(params![t0, t1, limit as i64], |r| {
            let attrs: String = r.get(5)?;
            Ok(Event {
                id: r.get(0)?,
                cam: r.get(1)?,
                ts: r.get(2)?,
                source: r.get(3)?,
                r#type: r.get(4)?,
                attrs: serde_json::from_str(&attrs).unwrap_or(serde_json::Value::Null),
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn insert_rule(
        &self,
        name: &str,
        text: &str,
        min_score: f64,
        class: Option<&str>,
        cam: Option<i64>,
        min_dwell_s: f64,
        emb: &[f32],
    ) -> Result<i64> {
        let now = chrono::Utc::now().timestamp_millis();
        self.conn.execute(
            "INSERT INTO rules(name,text,min_score,class,cam,min_dwell_s,enabled,emb_json,created) VALUES(?1,?2,?3,?4,?5,?6,1,?7,?8)",
            params![name, text, min_score, class, cam, min_dwell_s, serde_json::to_string(emb)?, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn rules(&self, only_enabled: bool) -> Result<Vec<Rule>> {
        let sql = if only_enabled {
            "SELECT id,name,text,min_score,class,cam,min_dwell_s,enabled,emb_json FROM rules WHERE enabled=1 ORDER BY id"
        } else {
            "SELECT id,name,text,min_score,class,cam,min_dwell_s,enabled,emb_json FROM rules ORDER BY id"
        };
        let mut st = self.conn.prepare(sql)?;
        let rows = st.query_map([], |r| {
            let emb_json: String = r.get(8)?;
            Ok(Rule {
                id: r.get(0)?,
                name: r.get(1)?,
                text: r.get(2)?,
                min_score: r.get(3)?,
                class: r.get(4)?,
                cam: r.get(5)?,
                min_dwell_s: r.get(6)?,
                enabled: r.get::<_, i64>(7)? == 1,
                emb: serde_json::from_str(&emb_json).unwrap_or_default(),
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn delete_rule(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM rules WHERE id=?1", [id])?;
        Ok(())
    }

    pub fn insert_alert(
        &self,
        rule_id: i64,
        tracklet_id: i64,
        ts: i64,
        score: f64,
        message: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO alerts(rule_id,tracklet_id,ts,score,message) VALUES(?1,?2,?3,?4,?5)",
            params![rule_id, tracklet_id, ts, score, message],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn alerts(&self, since_id: i64, limit: usize) -> Result<Vec<Alert>> {
        let mut st = self.conn.prepare("SELECT a.id,a.rule_id,COALESCE(r.name,''),a.tracklet_id,a.ts,a.score,a.message FROM alerts a LEFT JOIN rules r ON r.id=a.rule_id WHERE a.id>?1 ORDER BY a.id DESC LIMIT ?2")?;
        let rows = st.query_map(params![since_id, limit as i64], |r| {
            Ok(Alert {
                id: r.get(0)?,
                rule_id: r.get(1)?,
                rule_name: r.get(2)?,
                tracklet_id: r.get(3)?,
                ts: r.get(4)?,
                score: r.get(5)?,
                message: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn audit(&self, action: &str, detail: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO audit(ts,action,detail) VALUES(?1,?2,?3)",
            params![chrono::Utc::now().timestamp_millis(), action, detail],
        )?;
        Ok(())
    }

    pub fn stats(&self) -> Result<serde_json::Value> {
        let tracklets = self.tracklet_count()?;
        let captions: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tracklets WHERE caption_json IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        let events: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))?;
        let alerts: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM alerts", [], |r| r.get(0))?;
        let rules: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM rules WHERE enabled=1", [], |r| {
                    r.get(0)
                })?;
        let (tmin, tmax): (Option<i64>, Option<i64>) =
            self.conn
                .query_row("SELECT MIN(t0), MAX(t1) FROM tracklets", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?;
        let by_class: Vec<(String, i64)> = {
            let mut st = self
                .conn
                .prepare("SELECT class, COUNT(*) FROM tracklets GROUP BY class ORDER BY 2 DESC")?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<Result<_, _>>()?
        };
        let audio_windows = self.audio_window_count().unwrap_or(0);
        Ok(serde_json::json!({
            "tracklets": tracklets, "captioned": captions, "events": events, "alerts": alerts, "rules": rules,
            "audio_windows": audio_windows,
            "t_min": tmin, "t_max": tmax, "by_class": by_class,
            "emb_model": self.meta("emb_model")?, "emb_dim": self.meta("emb_dim")?,
            "cameras": self.cameras()?,
        }))
    }
}

const TRACKLET_SELECT: &str = "SELECT t.id,t.cam,COALESCE(c.name,''),t.class,t.t0,t.t1,t.path_json,t.entry_edge,t.exit_edge,t.dwell_s,t.speed,t.direction,t.best_crops_json,t.n_frames,t.caption_json,t.caption_text,t.bbox_json,t.frame_w,t.frame_h FROM tracklets t LEFT JOIN cameras c ON c.id=t.cam";

fn row_to_tracklet(r: &rusqlite::Row) -> rusqlite::Result<TrackletRow> {
    Ok(TrackletRow {
        id: r.get(0)?,
        cam: r.get(1)?,
        cam_name: r.get(2)?,
        class: r.get(3)?,
        t0: r.get(4)?,
        t1: r.get(5)?,
        path_json: r.get(6)?,
        entry_edge: r.get(7)?,
        exit_edge: r.get(8)?,
        dwell_s: r.get(9)?,
        speed: r.get(10)?,
        direction: r.get(11)?,
        best_crops_json: r.get(12)?,
        n_frames: r.get(13)?,
        caption_json: r.get(14)?,
        caption_text: r.get(15)?,
        bbox_json: r.get(16)?,
        frame_w: r.get(17)?,
        frame_h: r.get(18)?,
    })
}
