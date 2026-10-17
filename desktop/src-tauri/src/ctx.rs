//! App-wide state, managed by Tauri.

use crate::paths::Paths;
use crate::settings::Settings;
use crate::setup::View;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use tracing_appender::non_blocking::WorkerGuard;

/// The running engine.
pub struct Engine {
    pub state: Arc<retrace::api::AppState>,
    pub shutdown: retrace::api::Shutdown,
    /// The embedder the engine loaded; ingests use the same one so the vectors match.
    pub embedder: String,
}

/// One file for the ingest worker. `cam` overrides the name made from the file stem.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    pub path: PathBuf,
    pub cam: Option<String>,
}

pub struct Ctx {
    pub paths: Paths,
    pub settings: Mutex<Settings>,
    pub setup: Mutex<View>,
    pub setup_running: AtomicBool,
    pub engine: Mutex<Option<Engine>>,
    /// The engine port for the navigation filter (0 until the engine is bound).
    pub port: Arc<AtomicU16>,
    /// Files to index, one at a time. Items wait here until the engine is ready.
    pub queue_tx: Sender<Job>,
    pub queue_rx: Mutex<Option<Receiver<Job>>>,
    /// Set on quit: ingest loops stop.
    pub stop: Arc<AtomicBool>,
    /// One receiver per worker/live thread; it disconnects when the thread ends.
    pub threads: Mutex<Vec<(String, Receiver<()>)>>,
    pub cleaned_up: AtomicBool,
    /// A quit is running on its own thread (see `request_quit` in main.rs).
    pub quit_started: AtomicBool,
    /// Cleanup has finished: the process may exit now.
    pub quit_done: AtomicBool,
    /// Flushes the log file; cleanup drops it last so the final lines are written.
    pub log_guard: Mutex<Option<WorkerGuard>>,
}

impl Ctx {
    pub fn new(paths: Paths, settings: Settings) -> Self {
        let (queue_tx, rx) = std::sync::mpsc::channel();
        Self {
            paths,
            settings: Mutex::new(settings),
            setup: Mutex::new(View::default()),
            setup_running: AtomicBool::new(false),
            engine: Mutex::new(None),
            port: Arc::new(AtomicU16::new(0)),
            queue_tx,
            queue_rx: Mutex::new(Some(rx)),
            stop: Arc::new(AtomicBool::new(false)),
            threads: Mutex::new(Vec::new()),
            cleaned_up: AtomicBool::new(false),
            quit_started: AtomicBool::new(false),
            quit_done: AtomicBool::new(false),
            log_guard: Mutex::new(None),
        }
    }

    /// Queues a file for indexing (the worker starts once the engine runs).
    pub fn open_file(&self, p: PathBuf) {
        self.queue(Job { path: p, cam: None });
    }

    pub fn queue(&self, job: Job) {
        tracing::info!("queued for indexing: {}", job.path.display());
        let _ = self.queue_tx.send(job);
    }

    /// Registers a thread for the quit join; the thread must keep the returned sender alive.
    pub fn track_thread(&self, name: &str) -> Sender<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.threads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((name.to_string(), rx));
        tx
    }

    /// True when a thread with this name was registered (live cameras run until quit).
    pub fn has_thread(&self, name: &str) -> bool {
        self.threads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .any(|(n, _)| n == name)
    }

    /// Like `track_thread`, but `None` when the name is already registered (check and register
    /// under one lock).
    pub fn track_unique_thread(&self, name: &str) -> Option<Sender<()>> {
        let mut threads = self.threads.lock().unwrap_or_else(PoisonError::into_inner);
        if threads.iter().any(|(n, _)| n == name) {
            return None;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        threads.push((name.to_string(), rx));
        Some(tx)
    }
}
