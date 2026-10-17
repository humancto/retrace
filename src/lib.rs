//! Retrace: a fully local physical-AI video investigator.
//! ingest → tracklets → crop embeddings → SQLite (sqlite-vec + FTS5) → search / rules / UI.
//!
//! The `retrace` binary (src/main.rs) is a thin CLI over this library; the desktop app links the
//! same modules.

pub mod annotate;
pub mod api;
pub mod audio;
pub mod db;
pub mod ingest;
pub mod models;
pub mod overlay;
pub mod rules;
pub mod search;
pub mod tools;
pub mod track;
