//! Locates the external tools Retrace runs: ffmpeg, ffprobe and yt-dlp.
//!
//! Search order for a tool `name`:
//! 1. the override directory set with [`set_tools_dir`] (the desktop app's own `bin/`);
//! 2. the directory of the running executable (the same rule ffmpeg-sidecar uses);
//! 3. each directory in `PATH`;
//! 4. `/opt/homebrew/bin` and `/usr/local/bin` (a Finder-launched app gets a PATH without them);
//! 5. the bare name, so the OS reports "not found" when the command runs.
//!
//! [`resolve_in`] is the pure function behind the search; the global accessors only supply its
//! inputs. Nothing here changes the process environment.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

static TOOLS_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Directories checked after PATH.
pub const EXTRA_DIRS: &[&str] = &["/opt/homebrew/bin", "/usr/local/bin"];

/// Sets (or replaces) the directory that is searched first for every tool.
pub fn set_tools_dir(dir: impl Into<PathBuf>) {
    if let Ok(mut d) = TOOLS_DIR.write() {
        *d = Some(dir.into());
    }
}

/// Removes the override directory set with [`set_tools_dir`].
pub fn clear_tools_dir() {
    if let Ok(mut d) = TOOLS_DIR.write() {
        *d = None;
    }
}

/// The override directory, if one is set.
pub fn tools_dir() -> Option<PathBuf> {
    TOOLS_DIR.read().ok().and_then(|d| d.clone())
}

/// Resolves `name` with the process inputs (override dir, executable dir, PATH, Homebrew dirs).
pub fn resolve(name: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let path = std::env::var_os("PATH");
    let extra: Vec<&Path> = EXTRA_DIRS.iter().map(Path::new).collect();
    resolve_in(
        name,
        tools_dir().as_deref(),
        exe_dir.as_deref(),
        path.as_deref(),
        &extra,
    )
}

/// The ffmpeg to run.
pub fn ffmpeg() -> PathBuf {
    resolve("ffmpeg")
}

/// The ffprobe to run.
pub fn ffprobe() -> PathBuf {
    resolve("ffprobe")
}

/// The yt-dlp to run.
pub fn yt_dlp() -> PathBuf {
    resolve("yt-dlp")
}

/// `["--ffmpeg-location", <dir>]` when an override directory is set (the desktop app) and ffmpeg
/// resolves to an absolute path, so yt-dlp merges and remuxes with the same ffmpeg as Retrace.
/// Empty for the CLI (no override directory), so yt-dlp keeps its own search and the user's
/// yt-dlp config, as before.
pub fn yt_dlp_ffmpeg_args() -> Vec<String> {
    if tools_dir().is_none() {
        return vec![];
    }
    ffmpeg_location_args(&ffmpeg())
}

fn ffmpeg_location_args(ffmpeg: &Path) -> Vec<String> {
    match ffmpeg.parent() {
        Some(dir) if ffmpeg.is_absolute() => {
            vec![
                "--ffmpeg-location".into(),
                dir.to_string_lossy().into_owned(),
            ]
        }
        _ => vec![],
    }
}

/// Pure resolver: the first executable file called `name` (plus the platform suffix, `.exe` on
/// Windows) in `override_dir`, `exe_dir`, each entry of `path`, then `extra`; else the bare name.
pub fn resolve_in(
    name: &str,
    override_dir: Option<&Path>,
    exe_dir: Option<&Path>,
    path: Option<&OsStr>,
    extra: &[&Path],
) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    let from_path: Vec<PathBuf> = path
        .map(|p| std::env::split_paths(p).collect())
        .unwrap_or_default();
    override_dir
        .into_iter()
        .chain(exe_dir)
        .chain(from_path.iter().map(PathBuf::as_path))
        .chain(extra.iter().copied())
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| d.join(&file))
        .find(|p| is_executable(p))
        .unwrap_or_else(|| PathBuf::from(file))
}

#[cfg(unix)]
fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(p: &Path) -> bool {
    p.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(dir: &Path, name: &str, exec: bool) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&p, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if exec { 0o755 } else { 0o644 };
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = exec;
        p
    }

    fn join(dirs: &[&Path]) -> std::ffi::OsString {
        std::env::join_paths(dirs).unwrap()
    }

    #[test]
    fn override_dir_wins() {
        let t = tempfile::tempdir().unwrap();
        let (o, e, p, x) = (
            t.path().join("o"),
            t.path().join("e"),
            t.path().join("p"),
            t.path().join("x"),
        );
        let want = tool(&o, "ffmpeg", true);
        for d in [&e, &p, &x] {
            tool(d, "ffmpeg", true);
        }
        let path = join(&[&p]);
        let got = resolve_in("ffmpeg", Some(&o), Some(&e), Some(&path), &[&x]);
        assert_eq!(got, want);
    }

    #[test]
    fn exe_dir_before_path() {
        let t = tempfile::tempdir().unwrap();
        let (e, p) = (t.path().join("e"), t.path().join("p"));
        let want = tool(&e, "ffprobe", true);
        tool(&p, "ffprobe", true);
        let path = join(&[&p]);
        let missing = t.path().join("missing");
        let got = resolve_in("ffprobe", Some(&missing), Some(&e), Some(&path), &[]);
        assert_eq!(got, want);
    }

    #[test]
    fn path_order_is_kept() {
        let t = tempfile::tempdir().unwrap();
        let (a, b) = (t.path().join("a"), t.path().join("b"));
        tool(&a, "yt-dlp", true);
        tool(&b, "yt-dlp", true);
        let path = join(&[&b, &a]);
        let got = resolve_in("yt-dlp", None, None, Some(&path), &[]);
        assert_eq!(
            got,
            b.join(format!("yt-dlp{}", std::env::consts::EXE_SUFFIX))
        );
    }

    #[test]
    fn extra_dirs_after_path() {
        let t = tempfile::tempdir().unwrap();
        let (p, x) = (t.path().join("p"), t.path().join("x"));
        std::fs::create_dir_all(&p).unwrap();
        let want = tool(&x, "ffmpeg", true);
        let path = join(&[&p]);
        let got = resolve_in("ffmpeg", None, None, Some(&path), &[&x]);
        assert_eq!(got, want);
    }

    #[cfg(unix)]
    #[test]
    fn skips_non_executable_and_directories() {
        let t = tempfile::tempdir().unwrap();
        let (a, b, c) = (t.path().join("a"), t.path().join("b"), t.path().join("c"));
        tool(&a, "ffmpeg", false);
        std::fs::create_dir_all(b.join("ffmpeg")).unwrap();
        let want = tool(&c, "ffmpeg", true);
        let path = join(&[&a, &b]);
        let got = resolve_in("ffmpeg", None, None, Some(&path), &[&c]);
        assert_eq!(got, want);
    }

    #[test]
    fn falls_back_to_bare_name() {
        let t = tempfile::tempdir().unwrap();
        let path = join(&[t.path()]);
        let got = resolve_in("ffmpeg", Some(t.path()), None, Some(&path), &[]);
        assert_eq!(
            got,
            PathBuf::from(format!("ffmpeg{}", std::env::consts::EXE_SUFFIX))
        );
        let got = resolve_in("ffmpeg", None, None, None, &[]);
        assert!(!got.is_absolute());
    }

    #[test]
    fn empty_path_entries_are_ignored() {
        let got = resolve_in("ffmpeg", None, None, Some(OsStr::new("")), &[]);
        assert!(!got.is_absolute());
    }

    #[test]
    fn ffmpeg_location_only_for_absolute_paths() {
        assert!(ffmpeg_location_args(Path::new("ffmpeg")).is_empty());
        let abs = std::env::temp_dir().join("bin").join("ffmpeg");
        let args = ffmpeg_location_args(&abs);
        assert_eq!(args[0], "--ffmpeg-location");
        assert_eq!(PathBuf::from(&args[1]), abs.parent().unwrap());
    }
}
