//! ffmpeg, ffprobe and yt-dlp for the app: pinned downloads into `<root>/bin`, or the tools that
//! are already installed. The GPL ffmpeg build is downloaded at run time, never bundled.
//!
//! Every URL was checked with `curl -sIL`, and every sha256 was computed from a downloaded copy
//! and matched against the digest the publisher gives (martin-riedl `.sha256` files, GitHub
//! release asset digests).

use anyhow::{Context, Result};
use retrace::models::ModelFile;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    /// A zip archive holding one file called `member`.
    Zip { member: &'static str },
    /// The executable itself.
    Raw,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolAsset {
    /// The executable name in `bin/`: ffmpeg, ffprobe or yt-dlp.
    pub name: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub packing: Packing,
}

const MR: &str = "https://ffmpeg.martin-riedl.de/download";
/// ffmpeg 9.0.2 release build by Martin Riedl (GPL, has libx264 and aac).
pub const FFMPEG_VERSION: &str = "9.0.2";
pub const YT_DLP_VERSION: &str = "2026.08.19";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const ASSETS: &[ToolAsset] = &[
    ToolAsset {
        name: "ffmpeg",
        url: "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffmpeg.zip",
        size: 28_395_699,
        sha256: "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924",
        packing: Packing::Zip { member: "ffmpeg" },
    },
    ToolAsset {
        name: "ffprobe",
        url: "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2/ffprobe.zip",
        size: 28_317_701,
        sha256: "fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6",
        packing: Packing::Zip { member: "ffprobe" },
    },
    ToolAsset {
        name: "yt-dlp",
        url: "https://github.com/yt-dlp/yt-dlp/releases/download/2026.08.19/yt-dlp_macos",
        size: 37_146_048,
        sha256: "0f192b7ec147ab6288885d6351d9ab67367640029b4377576ef46dd79cf7b202",
        packing: Packing::Raw,
    },
];

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const ASSETS: &[ToolAsset] = &[
    ToolAsset {
        name: "ffmpeg",
        url: "https://ffmpeg.martin-riedl.de/download/linux/amd64/1789931100_9.0.2/ffmpeg.zip",
        size: 33_326_445,
        sha256: "fa8ecf4abbd290d98f7d188b8649cc6b391ae209a98452be955a15aab1909d7f",
        packing: Packing::Zip { member: "ffmpeg" },
    },
    ToolAsset {
        name: "ffprobe",
        url: "https://ffmpeg.martin-riedl.de/download/linux/amd64/1789931100_9.0.2/ffprobe.zip",
        size: 33_223_234,
        sha256: "3f428c49070be3d24ec338602b76d412e401ffcb8a5641ef0e729181a232fc32",
        packing: Packing::Zip { member: "ffprobe" },
    },
    ToolAsset {
        name: "yt-dlp",
        url: "https://github.com/yt-dlp/yt-dlp/releases/download/2026.08.19/yt-dlp_linux",
        size: 40_446_224,
        sha256: "58162f9bfdc27458ea47bfcb311cf47028f17d8154a8bf7d689861d46399230a",
        packing: Packing::Raw,
    },
];

/// No pinned builds for this platform: setup uses the installed tools.
#[cfg(not(any(
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "linux", target_arch = "x86_64")
)))]
pub const ASSETS: &[ToolAsset] = &[];

/// Where the downloaded archive goes before extraction.
fn download_dest(bin: &Path, a: &ToolAsset) -> PathBuf {
    match a.packing {
        Packing::Raw => bin.join(a.name),
        Packing::Zip { .. } => bin.join(".downloads").join(format!("{}.zip", a.name)),
    }
}

/// The assets whose executable is not in `bin` yet, each with the file to fetch.
pub fn plan(bin: &Path, assets: &[ToolAsset]) -> Vec<(ToolAsset, ModelFile)> {
    assets
        .iter()
        .filter(|a| !is_installed(bin, a))
        .map(|a| {
            (
                *a,
                ModelFile {
                    label: a.name.to_string(),
                    url: a.url.to_string(),
                    dest: download_dest(bin, a),
                    size: a.size,
                    sha256: Some(a.sha256),
                },
            )
        })
        .collect()
}

/// True when `bin/<name>` is an executable file (and, for a raw download, has the pinned size,
/// so a half-written file is never taken as installed). A symlink is never a pinned download:
/// "Use the installed tools" links `bin/` to the system tools, and after a change back to
/// Download those links must be replaced, not used.
pub fn is_installed(bin: &Path, a: &ToolAsset) -> bool {
    let p = bin.join(a.name);
    let is_link = std::fs::symlink_metadata(&p)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if is_link || !is_executable(&p) {
        return false;
    }
    match a.packing {
        Packing::Raw => std::fs::metadata(&p)
            .map(|m| m.len() == a.size)
            .unwrap_or(false),
        Packing::Zip { .. } => true,
    }
}

/// Turns a verified download into `bin/<name>` with mode 0755. A zip is extracted to a temp file,
/// renamed into place, then deleted.
pub fn install(bin: &Path, a: &ToolAsset, downloaded: &Path) -> Result<PathBuf> {
    let target = bin.join(a.name);
    match a.packing {
        Packing::Raw => {
            if downloaded != target {
                std::fs::rename(downloaded, &target)?;
            }
        }
        Packing::Zip { member } => {
            let tmp = bin.join(format!(".{}.tmp", a.name));
            extract_member(downloaded, member, &tmp)?;
            make_executable(&tmp)?;
            std::fs::rename(&tmp, &target)
                .with_context(|| format!("rename {} -> {}", tmp.display(), target.display()))?;
            let _ = std::fs::remove_file(downloaded);
        }
    }
    make_executable(&target)?;
    Ok(target)
}

/// Copies the zip entry called `member` (at any depth) to `out`.
pub fn extract_member(zip_path: &Path, member: &str, out: &Path) -> Result<()> {
    let f =
        std::fs::File::open(zip_path).with_context(|| format!("open {}", zip_path.display()))?;
    let mut z = zip::ZipArchive::new(f).context("read zip")?;
    let idx = (0..z.len())
        .find(|&i| {
            z.by_index(i)
                .ok()
                .map(|e| {
                    e.is_file()
                        && Path::new(e.name()).file_name().and_then(|n| n.to_str()) == Some(member)
                })
                .unwrap_or(false)
        })
        .with_context(|| format!("{member} is not in {}", zip_path.display()))?;
    let mut entry = z.by_index(idx)?;
    let mut w = std::fs::File::create(out).with_context(|| format!("create {}", out.display()))?;
    std::io::copy(&mut entry, &mut w)?;
    w.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn make_executable(p: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755))
        .with_context(|| format!("chmod {}", p.display()))
}

#[cfg(not(unix))]
fn make_executable(_: &Path) -> Result<()> {
    Ok(())
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

/// The installed tools found on PATH or in the Homebrew dirs (never `bin/` itself).
pub fn find_system(path: Option<&std::ffi::OsStr>) -> Option<crate::settings::SystemTools> {
    let extra: Vec<&Path> = retrace::tools::EXTRA_DIRS.iter().map(Path::new).collect();
    let find = |n: &str| {
        let p = retrace::tools::resolve_in(n, None, None, path, &extra);
        p.is_absolute().then_some(p)
    };
    Some(crate::settings::SystemTools {
        ffmpeg: find("ffmpeg")?,
        ffprobe: find("ffprobe")?,
        yt_dlp: find("yt-dlp")?,
    })
}

/// Points `bin/<name>` at the installed tools with symlinks, so the engine still has one tools
/// directory to search.
#[cfg(unix)]
pub fn link_system(bin: &Path, t: &crate::settings::SystemTools) -> Result<()> {
    for (name, src) in [
        ("ffmpeg", &t.ffmpeg),
        ("ffprobe", &t.ffprobe),
        ("yt-dlp", &t.yt_dlp),
    ] {
        let link = bin.join(name);
        if std::fs::symlink_metadata(&link).is_ok() {
            std::fs::remove_file(&link).with_context(|| format!("remove {}", link.display()))?;
        }
        std::os::unix::fs::symlink(src, &link)
            .with_context(|| format!("link {} -> {}", link.display(), src.display()))?;
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn link_system(_: &Path, _: &crate::settings::SystemTools) -> Result<()> {
    anyhow::bail!("installed tools are not supported on this platform yet")
}

/// True when `ffmpeg -encoders` output lists libx264 (clip export encodes H.264 with it).
pub fn has_libx264(encoders_output: &str) -> bool {
    encoders_output
        .lines()
        .any(|l| l.split_whitespace().nth(1) == Some("libx264"))
}

/// Runs `ffmpeg -hide_banner -encoders` and checks for libx264. Also proves the binary runs.
pub fn check_ffmpeg(ffmpeg: &Path) -> Result<bool> {
    let out = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-encoders"])
        .output()
        .with_context(|| format!("run {}", ffmpeg.display()))?;
    anyhow::ensure!(
        out.status.success(),
        "{} -encoders failed",
        ffmpeg.display()
    );
    Ok(has_libx264(&String::from_utf8_lossy(&out.stdout)))
}

/// Runs `<tool> <arg>` and returns the first line of output.
pub fn version_line(tool: &Path, arg: &str) -> Result<String> {
    let out = std::process::Command::new(tool)
        .arg(arg)
        .output()
        .with_context(|| format!("run {}", tool.display()))?;
    anyhow::ensure!(out.status.success(), "{} {arg} failed", tool.display());
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .to_string())
}

pub const THIRD_PARTY: &str = "\
Retrace Desktop: third-party software

Retrace downloads these programs and model weights on first run. They are not part of the
download of Retrace itself. Each one has its own licence.

ffmpeg and ffprobe 9.0.2
  Build: Martin Riedl, https://ffmpeg.martin-riedl.de/
  Licence: GNU General Public License version 3 or later (built with --enable-gpl --enable-version3)
  Source code: https://ffmpeg.org/download.html and https://git.ffmpeg.org/ffmpeg.git (tag n9.0.2)
  Build scripts: https://git.martin-riedl.de/ffmpeg/build-script

yt-dlp 2026.08.19
  https://github.com/yt-dlp/yt-dlp
  Licence: The Unlicense (public domain)

D-FINE object detector (deim-dfine-s), ONNX export by jamjamjon/usls
  https://github.com/Peterande/D-FINE and https://github.com/jamjamjon/assets
  Licence: Apache License 2.0

SigLIP 2 (siglip2-base-patch16-256), ONNX export by onnx-community
  https://huggingface.co/google/siglip2-base-patch16-256
  Licence: Apache License 2.0

EmbeddingGemma 2 (only when you select it)
  Licence: Gemma Terms of Use, https://ai.google.dev/gemma/terms

ONNX Runtime (in the app): MIT License
llama.cpp (in the app): MIT License
Tauri (in the app): MIT License or Apache License 2.0
";

/// Writes `THIRD-PARTY.txt` to the app root and the tool licence notes to `bin/`.
pub fn write_notices(root: &Path, bin: &Path) -> Result<PathBuf> {
    let p = root.join("THIRD-PARTY.txt");
    std::fs::write(&p, THIRD_PARTY)?;
    std::fs::write(
        bin.join("LICENSE-ffmpeg.txt"),
        format!(
            "ffmpeg/ffprobe {FFMPEG_VERSION}, build by Martin Riedl ({MR}).\n\
             Licence: GPL-3.0-or-later. Source: https://ffmpeg.org/download.html\n"
        ),
    )?;
    std::fs::write(
        bin.join("LICENSE-yt-dlp.txt"),
        format!(
            "yt-dlp {YT_DLP_VERSION}, https://github.com/yt-dlp/yt-dlp\nLicence: The Unlicense.\n"
        ),
    )?;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const TOOL_NAMES: [&str; 3] = ["ffmpeg", "ffprobe", "yt-dlp"];

    fn zip_with(path: &Path, name: &str, body: &[u8]) {
        let f = std::fs::File::create(path).unwrap();
        let mut w = zip::ZipWriter::new(f);
        w.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        w.write_all(body).unwrap();
        w.finish().unwrap();
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn pinned_assets_cover_every_tool_with_a_digest() {
        if ASSETS.is_empty() {
            return; // no pinned builds for this platform
        }
        for n in TOOL_NAMES {
            let a = ASSETS.iter().find(|a| a.name == n).expect(n);
            assert!(a.url.starts_with("https://"));
            assert_eq!(a.sha256.len(), 64);
            assert!(a.size > 1_000_000);
        }
        // The GPL build with libx264, not a LGPL one.
        assert!(ASSETS.iter().any(|a| a.url.starts_with(MR)));
    }

    #[test]
    fn libx264_detection() {
        let out = " V....D libx264              libx264 H.264\n A....D aac   AAC\n";
        assert!(has_libx264(out));
        assert!(!has_libx264(
            " V....D libx264rgb  x\n V....D h264_videotoolbox  y\n"
        ));
    }

    #[test]
    fn plan_skips_installed_and_install_extracts() {
        let t = tempfile::tempdir().unwrap();
        let bin = t.path().join("bin");
        std::fs::create_dir_all(bin.join(".downloads")).unwrap();
        let assets = [
            ToolAsset {
                name: "ffmpeg",
                url: "https://example.invalid/ffmpeg.zip",
                size: 1,
                sha256: "00",
                packing: Packing::Zip { member: "ffmpeg" },
            },
            ToolAsset {
                name: "yt-dlp",
                url: "https://example.invalid/yt-dlp",
                size: 5,
                sha256: "00",
                packing: Packing::Raw,
            },
        ];
        let p = plan(&bin, &assets);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].1.dest, bin.join(".downloads/ffmpeg.zip"));
        assert_eq!(p[1].1.dest, bin.join("yt-dlp"));

        zip_with(&p[0].1.dest, "dir/ffmpeg", b"#!/bin/sh\necho hi\n");
        let exe = install(&bin, &assets[0], &p[0].1.dest).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), b"#!/bin/sh\necho hi\n");
        assert!(!p[0].1.dest.exists(), "zip removed after extraction");

        std::fs::write(bin.join("yt-dlp"), b"12345").unwrap();
        install(&bin, &assets[1], &bin.join("yt-dlp")).unwrap();
        let left = plan(&bin, &assets);
        assert!(left.is_empty(), "{left:?}");

        // A raw tool with the wrong size is not taken as installed.
        std::fs::write(bin.join("yt-dlp"), b"123").unwrap();
        assert_eq!(plan(&bin, &assets).len(), 1);
    }

    #[test]
    fn extract_reports_missing_member() {
        let t = tempfile::tempdir().unwrap();
        let z = t.path().join("a.zip");
        zip_with(&z, "other", b"x");
        let e = extract_member(&z, "ffmpeg", &t.path().join("out")).unwrap_err();
        assert!(e.to_string().contains("ffmpeg is not in"), "{e}");
    }

    #[cfg(unix)]
    #[test]
    fn system_tools_found_and_linked() {
        use std::os::unix::fs::PermissionsExt;
        let t = tempfile::tempdir().unwrap();
        let sys = t.path().join("sys");
        std::fs::create_dir_all(&sys).unwrap();
        for n in TOOL_NAMES {
            let p = sys.join(n);
            std::fs::write(&p, "#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let st = find_system(Some(sys.as_os_str())).unwrap();
        assert_eq!(st.ffmpeg, sys.join("ffmpeg"));
        let bin = t.path().join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        link_system(&bin, &st).unwrap();
        link_system(&bin, &st).unwrap(); // replaces existing links
        assert_eq!(
            std::fs::read_link(bin.join("yt-dlp")).unwrap(),
            sys.join("yt-dlp")
        );
        let r = retrace::tools::resolve_in("ffprobe", Some(&bin), None, None, &[]);
        assert_eq!(r, bin.join("ffprobe"));

        // Back to Download: the links are not the pinned builds, so they are fetched again,
        // and installing replaces the link (not the system tool it points to).
        let ffmpeg = ToolAsset {
            name: "ffmpeg",
            url: "https://example.invalid/ffmpeg.zip",
            size: 1,
            sha256: "00",
            packing: Packing::Zip { member: "ffmpeg" },
        };
        assert!(!is_installed(&bin, &ffmpeg));
        assert_eq!(plan(&bin, &[ffmpeg]).len(), 1);
        let z = t.path().join("ffmpeg.zip");
        zip_with(&z, "ffmpeg", b"#!/bin/sh\necho pinned\n");
        install(&bin, &ffmpeg, &z).unwrap();
        assert!(is_installed(&bin, &ffmpeg));
        assert_eq!(
            std::fs::read(sys.join("ffmpeg")).unwrap(),
            b"#!/bin/sh\n",
            "the system tool is untouched"
        );
    }

    #[test]
    fn notices_written() {
        let t = tempfile::tempdir().unwrap();
        let p = write_notices(t.path(), t.path()).unwrap();
        assert!(
            std::fs::read_to_string(p)
                .unwrap()
                .contains("GNU General Public")
        );
        assert!(t.path().join("LICENSE-ffmpeg.txt").exists());
    }
}
