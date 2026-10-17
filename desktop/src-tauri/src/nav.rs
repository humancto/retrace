//! What the window may load and download. The window starts on the bundled setup page and then
//! shows the engine at `http://127.0.0.1:<port>/`; nothing else loads inside it.

use std::path::{Path, PathBuf};
use tauri::Url;

/// Overrides the download folder (tests).
pub const ENV_DOWNLOAD_DIR: &str = "RETRACE_DOWNLOAD_DIR";

#[derive(Debug, PartialEq, Eq)]
pub enum Nav {
    Allow,
    /// Open in the default browser instead.
    External,
    Deny,
}

fn is_engine(url: &Url, port: u16) -> bool {
    port != 0
        && url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
        && url.port() == Some(port)
}

fn is_bundled(url: &Url) -> bool {
    url.scheme() == "tauri"
        || (matches!(url.scheme(), "http" | "https") && url.host_str() == Some("tauri.localhost"))
}

pub fn classify(url: &Url, port: u16) -> Nav {
    if is_bundled(url) || is_engine(url, port) || url.as_str() == "about:blank" {
        Nav::Allow
    } else if matches!(url.scheme(), "http" | "https" | "mailto") {
        Nav::External
    } else {
        Nav::Deny
    }
}

/// Downloads are allowed from the engine only (direct links and blob/data URLs it makes).
pub fn download_allowed(url: &Url, port: u16) -> bool {
    match url.scheme() {
        "blob" => Url::parse(url.path()).is_ok_and(|inner| is_engine(&inner, port)),
        "data" => true,
        _ => is_engine(url, port),
    }
}

/// Where a download goes: the user's Downloads folder (or `RETRACE_DOWNLOAD_DIR`), with a free
/// name based on the suggested one.
pub fn download_target(suggested: &Path, url: &Url, fallback_dir: &Path) -> PathBuf {
    let dir = std::env::var_os(ENV_DOWNLOAD_DIR)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(dirs::download_dir)
        .unwrap_or_else(|| fallback_dir.to_path_buf());
    let name = suggested
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .or_else(|| {
            url.path_segments()
                .and_then(|mut s| s.next_back().map(str::to_string))
                .filter(|n| !n.is_empty())
        })
        .unwrap_or_else(|| "retrace-download".into());
    unique_in(&dir, &clean_name(&name))
}

fn clean_name(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':' | '\0') {
                '_'
            } else {
                c
            }
        })
        .collect();
    let s = s.trim_start_matches('.').to_string();
    if s.is_empty() {
        "retrace-download".into()
    } else {
        s
    }
}

/// `dir/name`, or `dir/stem (n).ext` when that exists, or a name with the current time in
/// nanoseconds when all of those exist. Never a path that exists, so a download never overwrites
/// a file.
pub fn unique_in(dir: &Path, name: &str) -> PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let p = Path::new(name);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(name);
    let ext = p.extension().and_then(|s| s.to_str());
    (1..10_000)
        .map(|n| match ext {
            Some(e) => dir.join(format!("{stem} ({n}).{e}")),
            None => dir.join(format!("{stem} ({n})")),
        })
        .find(|c| !c.exists())
        .unwrap_or_else(|| {
            let mut n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            loop {
                let c = match ext {
                    Some(e) => dir.join(format!("{stem} ({n}).{e}")),
                    None => dir.join(format!("{stem} ({n})")),
                };
                if !c.exists() {
                    return c;
                }
                n += 1;
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn navigation_policy() {
        assert_eq!(classify(&u("tauri://localhost/setup.html"), 0), Nav::Allow);
        assert_eq!(
            classify(&u("http://tauri.localhost/setup.html"), 0),
            Nav::Allow
        );
        assert_eq!(classify(&u("http://127.0.0.1:5000/"), 5000), Nav::Allow);
        assert_eq!(classify(&u("http://127.0.0.1:5001/"), 5000), Nav::External);
        assert_eq!(classify(&u("http://127.0.0.1:5000/"), 0), Nav::External);
        assert_eq!(
            classify(&u("https://github.com/humancto/retrace"), 5000),
            Nav::External
        );
        assert_eq!(classify(&u("file:///etc/passwd"), 5000), Nav::Deny);
    }

    #[test]
    fn download_policy() {
        assert!(download_allowed(&u("http://127.0.0.1:7/clip?id=3"), 7));
        assert!(download_allowed(&u("blob:http://127.0.0.1:7/0b6c-11"), 7));
        assert!(!download_allowed(&u("blob:https://evil.example/0b6c"), 7));
        assert!(!download_allowed(&u("https://example.com/a.mp4"), 7));
        assert!(download_allowed(&u("data:application/json,{}"), 7));
    }

    #[test]
    fn unique_names() {
        let t = tempfile::tempdir().unwrap();
        assert_eq!(unique_in(t.path(), "a.mp4"), t.path().join("a.mp4"));
        std::fs::write(t.path().join("a.mp4"), b"").unwrap();
        assert_eq!(unique_in(t.path(), "a.mp4"), t.path().join("a (1).mp4"));
        std::fs::write(t.path().join("a (1).mp4"), b"").unwrap();
        assert_eq!(unique_in(t.path(), "a.mp4"), t.path().join("a (2).mp4"));
        assert_eq!(clean_name("../x/y.mp4"), "_x_y.mp4");
    }

    #[test]
    fn unique_never_returns_an_existing_path() {
        let t = tempfile::tempdir().unwrap();
        std::fs::write(t.path().join("b"), b"").unwrap();
        for n in 1..10_000 {
            std::fs::write(t.path().join(format!("b ({n})")), b"").unwrap();
        }
        let p = unique_in(t.path(), "b");
        assert!(!p.exists(), "{}", p.display());
        assert_eq!(p.parent(), Some(t.path()));
    }

    #[test]
    fn target_uses_suggested_name_then_url() {
        let t = tempfile::tempdir().unwrap();
        let p = download_target(
            Path::new(""),
            &u("http://127.0.0.1:7/files/clip-21.mp4"),
            t.path(),
        );
        assert_eq!(p.file_name().unwrap(), "clip-21.mp4");
        let p = download_target(Path::new("/tmp/report.json"), &u("blob:x"), t.path());
        assert_eq!(p.file_name().unwrap(), "report.json");
    }
}
