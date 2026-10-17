//! App-wide state, managed by Tauri.

use crate::paths::Paths;
use crate::settings::Settings;
use crate::setup::View;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};

/// The running engine.
pub struct Engine {
    pub state: Arc<retrace::api::AppState>,
    pub shutdown: retrace::api::Shutdown,
    /// The embedder the engine loaded; ingests use the same one so the vectors match.
    pub embedder: String,
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
    pub queue_tx: Sender<PathBuf>,
    pub queue_rx: Mutex<Option<Receiver<PathBuf>>>,
    /// Set on quit: ingest loops stop.
    pub stop: Arc<AtomicBool>,
    /// One receiver per worker/live thread; it disconnects when the thread ends.
    pub threads: Mutex<Vec<(String, Receiver<()>)>>,
    pub cleaned_up: AtomicBool,
    /// A quit is running on its own thread (see `request_quit` in main.rs).
    pub quit_started: AtomicBool,
    /// Cleanup has finished: the process may exit now.
    pub quit_done: AtomicBool,
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
        }
    }

    /// Queues a file for indexing (the worker starts once the engine runs).
    pub fn open_file(&self, p: PathBuf) {
        tracing::info!("queued for indexing: {}", p.display());
        let _ = self.queue_tx.send(p);
    }

    /// Registers a thread for the quit join; the thread must keep the returned sender alive.
    pub fn track_thread(&self, name: &str) -> Sender<()> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.threads.lock().unwrap().push((name.to_string(), rx));
        tx
    }
}
