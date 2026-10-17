//! The app data layout. Every path is absolute: a .app starts with CWD `/`, so no relative
//! default from the CLI is safe here.

use std::path::{Path, PathBuf};

/// Overrides the app data root (tests and unattended runs).
pub const ENV_APP_DATA: &str = "RETRACE_APP_DATA";
/// The bundle identifier, also the directory name under the OS data dir.
pub const IDENTIFIER: &str = "com.humancto.retrace";

#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
    pub bin: PathBuf,
    pub models: PathBuf,
    pub cache: PathBuf,
    pub logs: PathBuf,
    pub rec: PathBuf,
    pub settings: PathBuf,
}

impl Paths {
    /// `RETRACE_APP_DATA` when set (made absolute), else `<data dir>/com.humancto.retrace`.
    pub fn from_env() -> anyhow::Result<Self> {
        let root = match std::env::var_os(ENV_APP_DATA).filter(|v| !v.is_empty()) {
            Some(v) => absolute(Path::new(&v))?,
            None => dirs::data_dir()
                .ok_or_else(|| anyhow::anyhow!("no data directory for this user"))?
                .join(IDENTIFIER),
        };
        Ok(Self::new(root))
    }

    pub fn new(root: PathBuf) -> Self {
        Self {
            bin: root.join("bin"),
            models: root.join("models"),
            cache: root.join("cache"),
            logs: root.join("logs"),
            rec: root.join("rec"),
            settings: root.join("settings.json"),
            root,
        }
    }

    /// Creates every directory of the layout.
    pub fn create(&self) -> std::io::Result<()> {
        for d in [
            &self.root,
            &self.bin,
            &self.models,
            &self.cache,
            &self.logs,
            &self.rec,
        ] {
            std::fs::create_dir_all(d)?;
        }
        Ok(())
    }

    /// `index-<embedder>.db`: one index per embedder, because vectors of two models do not mix.
    pub fn db_for(&self, embedder: &str) -> PathBuf {
        self.root.join(format!("index-{}.db", sanitize(embedder)))
    }

    /// `crops-<embedder>/`.
    pub fn crops_for(&self, embedder: &str) -> PathBuf {
        self.root.join(format!("crops-{}", sanitize(embedder)))
    }
}

/// A file-name-safe form of an embedder name: `gemma2-native:256` → `gemma2-native-256`.
pub fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches('.').to_string();
    if s.is_empty() { "default".into() } else { s }
}

/// `p` made absolute against the current directory (no symlink resolution).
pub fn absolute(p: &Path) -> anyhow::Result<PathBuf> {
    Ok(if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()?.join(p)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_embedder_names() {
        assert_eq!(sanitize("siglip2-b16-256"), "siglip2-b16-256");
        assert_eq!(sanitize("gemma2-native:256"), "gemma2-native-256");
        assert_eq!(sanitize("a/b\\c d"), "a-b-c-d");
        assert_eq!(sanitize("../.."), "-");
        assert_eq!(sanitize(""), "default");
    }

    #[test]
    fn layout_is_absolute_and_per_embedder() {
        let p = Paths::new(PathBuf::from("/x/app"));
        assert_eq!(p.bin, PathBuf::from("/x/app/bin"));
        assert_eq!(
            p.db_for("gemma2-native:256"),
            PathBuf::from("/x/app/index-gemma2-native-256.db")
        );
        assert_eq!(
            p.crops_for("siglip2-b16-256"),
            PathBuf::from("/x/app/crops-siglip2-b16-256")
        );
    }

    #[test]
    fn create_makes_every_dir() {
        let t = tempfile::tempdir().unwrap();
        let p = Paths::new(t.path().join("root"));
        p.create().unwrap();
        for d in [&p.bin, &p.models, &p.cache, &p.logs, &p.rec] {
            assert!(d.is_dir(), "{}", d.display());
        }
    }
}
