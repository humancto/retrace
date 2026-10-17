//! `settings.json` in the app data root.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Overrides where tools come from: `download` or `system`.
pub const ENV_SETUP: &str = "RETRACE_SETUP";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolSource {
    /// Pinned builds downloaded into `<root>/bin`.
    #[default]
    Download,
    /// Tools already installed on the machine (absolute paths in `system_tools`).
    System,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SystemTools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub yt_dlp: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// siglip2-b16-256 (default) or gemma2-native[:dim].
    pub embedder: String,
    pub detector: String,
    pub fps: f32,
    pub width: u32,
    pub min_crop_h: u32,
    pub last_dir: Option<PathBuf>,
    pub tools: ToolSource,
    pub system_tools: Option<SystemTools>,
}

impl Default for Settings {
    // The e2e values (scripts/e2e.sh), so the app indexes like the tested CLI path.
    fn default() -> Self {
        Self {
            embedder: "siglip2-b16-256".into(),
            detector: "deim-dfine-s".into(),
            fps: 4.0,
            width: 640,
            min_crop_h: 40,
            last_dir: None,
            tools: ToolSource::Download,
            system_tools: None,
        }
    }
}

impl Settings {
    /// Reads `path`; a missing or unreadable file gives the defaults (a bad file is logged and
    /// kept on disk as `settings.json.bad`).
    pub fn load(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        match serde_json::from_str(&text) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("settings.json is not valid ({e}); using defaults");
                let _ = std::fs::rename(path, path.with_extension("json.bad"));
                Self::default()
            }
        }
    }

    /// Writes atomically (temp file + rename).
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// The tool source after the `RETRACE_SETUP` override.
    pub fn effective_tools(&self) -> ToolSource {
        match std::env::var(ENV_SETUP).ok().as_deref() {
            Some("system") => ToolSource::System,
            Some("download") => ToolSource::Download,
            _ => self.tools,
        }
    }

    pub fn models(&self, models_dir: &Path) -> retrace::models::ModelSettings {
        retrace::models::ModelSettings {
            models_dir: Some(models_dir.to_path_buf()),
            detector: self.detector.clone(),
            embedder: self.embedder.clone(),
            device: "cpu".into(),
            conf: 0.35,
            threads: 0,
            embed_url: "http://127.0.0.1:8082".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_e2e() {
        let s = Settings::default();
        assert_eq!(s.embedder, "siglip2-b16-256");
        assert_eq!(s.detector, "deim-dfine-s");
        assert_eq!((s.fps, s.width, s.min_crop_h), (4.0, 640, 40));
    }

    #[test]
    fn roundtrip_and_partial_file() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("settings.json");
        assert_eq!(Settings::load(&p), Settings::default());
        std::fs::write(&p, r#"{"embedder":"gemma2-native:256","tools":"system"}"#).unwrap();
        let s = Settings::load(&p);
        assert_eq!(s.embedder, "gemma2-native:256");
        assert_eq!(s.tools, ToolSource::System);
        assert_eq!(s.fps, 4.0);
        s.save(&p).unwrap();
        assert_eq!(Settings::load(&p), s);
    }

    #[test]
    fn bad_file_falls_back_and_is_kept() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("settings.json");
        std::fs::write(&p, "{not json").unwrap();
        assert_eq!(Settings::load(&p), Settings::default());
        assert!(t.path().join("settings.json.bad").exists());
    }

    #[test]
    fn models_use_the_flat_dir() {
        let m = Settings::default().models(Path::new("/r/models"));
        assert_eq!(m.models_dir.as_deref(), Some(Path::new("/r/models")));
        assert_eq!(m.device, "cpu");
    }
}
