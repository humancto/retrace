//! Model prefetch: downloads the weights a [`ModelSettings`] needs before the first load, with
//! byte progress, resumable `.part` files, size and sha256 checks, and an atomic rename.
//!
//! Where files go:
//! - `models_dir` set: the flat layout that [`super::resolve`] and `siglip_local` already check
//!   (`<models_dir>/deim/dfine-s-coco.onnx`, `<models_dir>/siglip2-b16-256/vision_model.onnx`, …).
//! - `models_dir` unset: the usls cache layout (`<cache>/usls/<owner>/<repo>/<tag>/<file>` for a
//!   GitHub release asset, `<cache>/usls/<owner>/<repo>/<file>` for a Hugging Face file), which is
//!   where usls looks before it downloads. usls itself is not changed.
//!
//! The EmbeddingGemma 2 GGUFs (`gemma2-native`) always go to `<models_dir or ./models>/embeddinggemma-2/`,
//! the directory `load_native` reads.
//!
//! A file that already exists with the expected size counts as present and is not hashed again.

use super::{GH, ModelSettings};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const HF: &str = "https://huggingface.co";

/// One file to fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFile {
    /// Short display name, e.g. `siglip2-b16-256/vision_model.onnx`.
    pub label: String,
    pub url: String,
    pub dest: PathBuf,
    /// Exact size in bytes.
    pub size: u64,
    /// Lowercase hex sha256, when the publisher gives one (or it was pinned from a verified copy).
    pub sha256: Option<&'static str>,
}

impl ModelFile {
    /// True when `dest` exists with the expected size.
    pub fn is_present(&self) -> bool {
        std::fs::metadata(&self.dest)
            .map(|m| m.is_file() && m.len() == self.size)
            .unwrap_or(false)
    }

    /// `<dest>.part`, the in-progress download.
    pub fn part_path(&self) -> PathBuf {
        let mut name = self.dest.file_name().unwrap_or_default().to_os_string();
        name.push(".part");
        self.dest.with_file_name(name)
    }
}

/// Progress of a [`prefetch`] call. Byte counts include files that were already present.
#[derive(Debug, Clone, Copy)]
pub struct Progress<'a> {
    pub file: &'a ModelFile,
    /// 0-based index of `file` in the plan.
    pub index: usize,
    pub count: usize,
    /// Bytes of `file` on disk so far.
    pub file_done: u64,
    /// Bytes of every file on disk so far.
    pub done: u64,
    /// Bytes of every file in the plan.
    pub total: u64,
}

/// `(owner/repo, pinned commit, [(path in repo, size, sha256)])`
type HfRepo = (
    &'static str,
    &'static str,
    &'static [(&'static str, u64, Option<&'static str>)],
);

// Tokenizer files are identical (same git blob) in the three SigLIP 2 repos.
const SIGLIP_TOK: (&str, u64, Option<&str>) = (
    "tokenizer.json",
    34_363_039,
    Some("cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
);
const SIGLIP_TOK_CFG: (&str, u64, Option<&str>) = (
    "tokenizer_config.json",
    47_240,
    Some("7c3a247599e741bceba1a3fe0285aea88d1044dc1fad2caa1e48cdd9fd25f630"),
);
const SIGLIP_SPECIAL: (&str, u64, Option<&str>) = (
    "special_tokens_map.json",
    636,
    Some("baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351"),
);

const SIGLIP2_B16_256: HfRepo = (
    "onnx-community/siglip2-base-patch16-256-ONNX",
    "d1114256522a37ffa257a0a58017348ab0058db2",
    &[
        SIGLIP_TOK,
        SIGLIP_TOK_CFG,
        SIGLIP_SPECIAL,
        (
            "onnx/text_model.onnx",
            1_129_469_657,
            Some("d3de4a6bbbfcb429b6615ac496790353cf4a4fc0f19fbbe7179e523ae60daaef"),
        ),
        // Last: `siglip_local` treats the directory as complete once this file exists.
        (
            "onnx/vision_model.onnx",
            371_992_072,
            Some("f5cb16728a704703f05516ded628397e11dbca4de2eb5db04b0c0bcee988aa7a"),
        ),
    ],
);

const SIGLIP2_B16_384: HfRepo = (
    "onnx-community/siglip2-base-patch16-384-ONNX",
    "99f5d40afb5331b81eecee4bb02c360fbb9cdf8a",
    &[
        SIGLIP_TOK,
        SIGLIP_TOK_CFG,
        SIGLIP_SPECIAL,
        (
            "onnx/text_model.onnx",
            1_129_469_657,
            Some("d28c21c7f12c38b0ec43aacb7ce2228fba6bd6b20641802ef2b29809ece46af8"),
        ),
        (
            "onnx/vision_model.onnx",
            372_975_112,
            Some("49ae4958b1098ca995e929d646f7be05a69c65e6344beae07d58c6598ffc5210"),
        ),
    ],
);

const SIGLIP2_SO400M_384: HfRepo = (
    "onnx-community/siglip2-so400m-patch16-384-ONNX",
    "29f74e673c298fd07482d8e20138179282001eb7",
    &[
        SIGLIP_TOK,
        SIGLIP_TOK_CFG,
        SIGLIP_SPECIAL,
        (
            "onnx/text_model.onnx",
            599_026,
            Some("1be6a35f94aa0d1aa15024569ccb7637c7157809624922e50e22d746d43745a5"),
        ),
        (
            "onnx/vision_model.onnx",
            1_713_609_535,
            Some("e3f730b2a37c69ac08c2159fe3c7fab97f23fbe3f32319f9953375ba82ecb1e6"),
        ),
    ],
);

/// EmbeddingGemma 2, Q8_0 text model + Q8_0 mmproj (vision + audio). Exact names, because
/// `llama_embed::find_model_files` globs the directory.
const GEMMA2_GGUF: HfRepo = (
    "ggml-org/embeddinggemma-2-GGUF",
    "bfcd298762cc34d0357ece5ebdd31791a3a374d8",
    &[
        (
            "embeddinggemma-2-Q8_0.gguf",
            309_855_456,
            Some("2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"),
        ),
        (
            "mmproj-embeddinggemma-2-Q8_0.gguf",
            554_821_024,
            Some("c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f"),
        ),
    ],
);

/// GitHub release assets (jamjamjon/assets): `(tag, file, size, sha256)`. GitHub publishes no
/// digest for these; dfine-s-coco was pinned from two independent downloads.
const GH_DEIM_S: (&str, &str, u64, Option<&str>) = (
    "deim",
    "dfine-s-coco.onnx",
    41_761_986,
    Some("2b71a81cf05be32bc8f3bd50a4346cba55b2b7b68e1a49ccc06e4fa552d394f7"),
);
const GH_DEIM_M: (&str, &str, u64, Option<&str>) = ("deim", "dfine-m-coco.onnx", 77_864_465, None);
const GH_CLIP: &[(&str, &str, u64, Option<&str>)] = &[
    (
        "clip",
        "tokenizer.json",
        2_224_003,
        Some("a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4"),
    ),
    (
        "clip",
        "tokenizer_config.json",
        592,
        Some("34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad"),
    ),
    (
        "clip",
        "special_tokens_map.json",
        389,
        Some("f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf"),
    ),
    ("clip", "vit-b16-textual.onnx", 253_867_625, None),
    ("clip", "vit-b16-visual.onnx", 344_900_662, None),
];

/// The usls cache root (`<OS cache dir>/usls`), where usls looks for downloaded files.
pub fn usls_cache_dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("usls"))
}

/// The files `s` needs, in download order, with destinations as described in the module docs.
/// Custom detector paths/URLs and the `gemma2` sidecar embedder add nothing.
pub fn plan(s: &ModelSettings) -> Result<Vec<ModelFile>> {
    plan_in(s, usls_cache_dir().as_deref())
}

/// [`plan`] with an explicit usls cache root (used when `models_dir` is unset).
pub fn plan_in(s: &ModelSettings, usls_cache: Option<&Path>) -> Result<Vec<ModelFile>> {
    let flat = s.models_dir.as_deref();
    let need_cache =
        || usls_cache.context("no cache directory for the usls layout: set a models dir instead");
    let mut out = Vec::new();

    let gh = |out: &mut Vec<ModelFile>,
              (tag, file, size, sha): (&str, &str, u64, Option<&'static str>)|
     -> Result<()> {
        let dest = match flat {
            Some(d) => d.join(tag).join(file),
            None => need_cache()?
                .join("jamjamjon")
                .join("assets")
                .join(tag)
                .join(file),
        };
        out.push(ModelFile {
            label: format!("{tag}/{file}"),
            url: format!("{GH}/{tag}/{file}"),
            dest,
            size,
            sha256: sha,
        });
        Ok(())
    };
    // `sub`: the flat-layout directory name `siglip_local` uses.
    let hf = |out: &mut Vec<ModelFile>, (repo, rev, files): HfRepo, sub: &str| -> Result<()> {
        for &(path, size, sha) in files {
            let base = path.rsplit('/').next().unwrap_or(path);
            let dest = match flat {
                Some(d) => d.join(sub).join(base),
                None => need_cache()?.join(repo).join(path),
            };
            out.push(ModelFile {
                label: format!("{sub}/{base}"),
                url: format!("{HF}/{repo}/resolve/{rev}/{path}"),
                dest,
                size,
                sha256: sha,
            });
        }
        Ok(())
    };

    match s.detector.as_str() {
        "deim-dfine-s" => gh(&mut out, GH_DEIM_S)?,
        "deim-dfine-m" => gh(&mut out, GH_DEIM_M)?,
        "deim-dfine-n" => anyhow::bail!(
            "deim-dfine-n is not published in jamjamjon/assets: pick deim-dfine-s or a local .onnx"
        ),
        _ => {} // a path or URL the user supplied
    }

    match s.embedder.as_str() {
        "clip-b16" => {
            for &f in GH_CLIP {
                gh(&mut out, f)?;
            }
        }
        "siglip2-b16-256" => hf(&mut out, SIGLIP2_B16_256, "siglip2-b16-256")?,
        "siglip2-b16-384" => hf(&mut out, SIGLIP2_B16_384, "siglip2-b16-384")?,
        "siglip2-so400m-384" => hf(&mut out, SIGLIP2_SO400M_384, "siglip2-so400m-patch16-384")?,
        e if e == "gemma2-native" || e.starts_with("gemma2-native:") => {
            if !cfg!(feature = "gemma-native") {
                anyhow::bail!(
                    "this build has no in-process EmbeddingGemma 2: rebuild with `--features gemma-native`"
                );
            }
            let dir = flat.unwrap_or(Path::new("models")).join("embeddinggemma-2");
            let (repo, rev, files) = GEMMA2_GGUF;
            for &(path, size, sha) in files {
                out.push(ModelFile {
                    label: format!("embeddinggemma-2/{path}"),
                    url: format!("{HF}/{repo}/resolve/{rev}/{path}"),
                    dest: dir.join(path),
                    size,
                    sha256: sha,
                });
            }
        }
        "gemma2" => {} // the sidecar owns its weights
        other => anyhow::bail!("unknown embedder '{other}'"),
    }
    Ok(out)
}

/// Downloads every file of [`plan`] that is not present yet and returns the destinations.
/// `progress` runs before the first byte of each file, during the transfer, and once more when the
/// file is in place.
pub fn prefetch(s: &ModelSettings, progress: impl FnMut(&Progress)) -> Result<Vec<PathBuf>> {
    prefetch_files(&plan(s)?, progress)
}

/// [`prefetch`] for an explicit file list.
pub fn prefetch_files(
    files: &[ModelFile],
    mut progress: impl FnMut(&Progress),
) -> Result<Vec<PathBuf>> {
    let agent = agent();
    let total: u64 = files.iter().map(|f| f.size).sum();
    let mut before = 0u64;
    for (index, file) in files.iter().enumerate() {
        let mut report = |file_done: u64| {
            progress(&Progress {
                file,
                index,
                count: files.len(),
                file_done,
                done: before + file_done,
                total,
            })
        };
        if file.is_present() {
            report(file.size);
        } else {
            tracing::info!(file = %file.label, bytes = file.size, "downloading {}", file.url);
            report(0);
            download(&agent, file, &mut report)
                .with_context(|| format!("download {} ({})", file.label, file.url))?;
            report(file.size);
        }
        before += file.size;
    }
    Ok(files.iter().map(|f| f.dest.clone()).collect())
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        // 206/416 are answers to handle, not errors.
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(30)))
        // Per read, not per file: a 1 GB file may take minutes.
        .timeout_recv_body(None)
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .proxy(ureq::Proxy::try_from_env())
        .build()
        .into()
}

/// Fetches one file into `<dest>.part` (resuming what is there), checks size and sha256, then
/// renames it to `dest`. A checksum mismatch deletes the part and downloads once more from zero.
pub fn download(
    agent: &ureq::Agent,
    file: &ModelFile,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<()> {
    let part = file.part_path();
    if let Some(dir) = file.dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let mut last_err = None;
    for attempt in 0..2 {
        if attempt > 0 {
            let _ = std::fs::remove_file(&part);
        }
        transfer(agent, file, &part, on_bytes)?;
        match verify(file, &part) {
            Ok(()) => {
                std::fs::rename(&part, &file.dest).with_context(|| {
                    format!("rename {} -> {}", part.display(), file.dest.display())
                })?;
                return Ok(());
            }
            Err(e) => {
                tracing::warn!(file = %file.label, "{e:#}; downloading again");
                last_err = Some(e);
            }
        }
    }
    let _ = std::fs::remove_file(&part);
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("download failed")))
}

/// Brings `part` to `file.size` bytes, resuming with a Range request when it already has some.
fn transfer(
    agent: &ureq::Agent,
    file: &ModelFile,
    part: &Path,
    on_bytes: &mut dyn FnMut(u64),
) -> Result<()> {
    let mut have = std::fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have > file.size {
        std::fs::remove_file(part)?;
        have = 0;
    }
    if have == file.size {
        return Ok(()); // complete; verify decides
    }
    let mut req = agent.get(&file.url);
    if have > 0 {
        req = req.header("Range", &format!("bytes={have}-"));
    }
    let resp = req.call()?;
    let status = resp.status().as_u16();
    let append = match status {
        206 if have > 0 && content_range_start(&resp) == Some(have) => true,
        200 => {
            // The server ignored the range: start over from byte 0.
            have = 0;
            false
        }
        206 => {
            // A range that does not start at `have` cannot be appended, and its body does not
            // start at byte 0 either. Drop the partial file so the next try starts clean.
            let _ = std::fs::remove_file(part);
            anyhow::bail!("server answered a different range than bytes={have}-");
        }
        416 => anyhow::bail!("server rejected the resume range at {have} bytes"),
        _ => anyhow::bail!("HTTP {status}"),
    };
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(part)
        .with_context(|| format!("open {}", part.display()))?;
    if have > 0 {
        tracing::info!(file = %file.label, from = have, "resuming download");
    }
    on_bytes(have);
    let mut body = resp.into_body();
    let mut reader = body.as_reader();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        have += n as u64;
        anyhow::ensure!(
            have <= file.size,
            "server sent more than the expected {} bytes",
            file.size
        );
        on_bytes(have);
    }
    out.sync_all()?;
    anyhow::ensure!(
        have == file.size,
        "connection ended at {have} of {} bytes (run again to resume)",
        file.size
    );
    Ok(())
}

fn content_range_start(resp: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    // "bytes <start>-<end>/<total>"
    let v = resp.headers().get("content-range")?.to_str().ok()?;
    v.trim()
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .trim()
        .parse()
        .ok()
}

fn verify(file: &ModelFile, part: &Path) -> Result<()> {
    let len = std::fs::metadata(part)?.len();
    anyhow::ensure!(
        len == file.size,
        "size mismatch: {len} bytes, expected {}",
        file.size
    );
    if let Some(want) = file.sha256 {
        let got = sha256_file(part)?;
        anyhow::ensure!(
            got.eq_ignore_ascii_case(want),
            "sha256 mismatch: got {got}, expected {want}"
        );
    }
    Ok(())
}

/// Lowercase hex sha256 of a file.
pub fn sha256_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        /// Honour Range with 206.
        Ranges,
        /// Ignore Range, always 200 with the full body.
        NoRanges,
        /// Serve a body with one byte flipped.
        Corrupt,
        /// Send only the first half of the body, then close.
        Truncate,
        /// Answer a Range with 206 for a range that starts 1000 bytes before the one asked for.
        WrongRange,
    }

    struct Server {
        url: String,
        ranges: Arc<Mutex<Vec<Option<String>>>>,
    }

    fn serve(body: Vec<u8>, mode: Mode) -> Server {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file.bin", l.local_addr().unwrap());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let seen = ranges.clone();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { break };
                let mut req = Vec::new();
                let mut b = [0u8; 1];
                while !req.ends_with(b"\r\n\r\n") {
                    if s.read(&mut b).unwrap_or(0) == 0 {
                        break;
                    }
                    req.push(b[0]);
                }
                let req = String::from_utf8_lossy(&req).to_string();
                let range = req
                    .lines()
                    .find_map(|l| l.strip_prefix("Range: ").or(l.strip_prefix("range: ")))
                    .map(|r| r.trim().to_string());
                seen.lock().unwrap().push(range.clone());
                let mut data = body.clone();
                if mode == Mode::Corrupt {
                    data[0] ^= 0xff;
                }
                let start = match (mode, &range) {
                    (Mode::Ranges, Some(r)) => r
                        .trim_start_matches("bytes=")
                        .trim_end_matches('-')
                        .parse::<usize>()
                        .unwrap(),
                    (Mode::WrongRange, Some(r)) => {
                        r.trim_start_matches("bytes=")
                            .trim_end_matches('-')
                            .parse::<usize>()
                            .unwrap()
                            - 1000
                    }
                    _ => 0,
                };
                let slice = &data[start..];
                let head = if start > 0 {
                    format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                        slice.len(),
                        start,
                        data.len() - 1,
                        data.len()
                    )
                } else {
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        slice.len()
                    )
                };
                let _ = s.write_all(head.as_bytes());
                let send = if mode == Mode::Truncate {
                    &slice[..slice.len() / 2]
                } else {
                    slice
                };
                let _ = s.write_all(send);
            }
        });
        Server { url, ranges }
    }

    fn body() -> Vec<u8> {
        (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    fn file_for(dir: &Path, url: &str, data: &[u8], pinned: bool) -> ModelFile {
        let sha: Option<&'static str> = if pinned {
            let mut h = Sha256::new();
            h.update(data);
            let hex: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            Some(Box::leak(hex.into_boxed_str()))
        } else {
            None
        };
        ModelFile {
            label: "test/file.bin".into(),
            url: url.into(),
            dest: dir.join("sub").join("file.bin"),
            size: data.len() as u64,
            sha256: sha,
        }
    }

    #[test]
    fn downloads_fresh_with_progress() {
        let data = body();
        let srv = serve(data.clone(), Mode::Ranges);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, true);
        let mut seen = Vec::new();
        let out = prefetch_files(std::slice::from_ref(&f), |p| {
            seen.push((p.file_done, p.done, p.total))
        })
        .unwrap();
        assert_eq!(out, vec![f.dest.clone()]);
        assert_eq!(std::fs::read(&f.dest).unwrap(), data);
        assert!(!f.part_path().exists());
        assert!(
            seen.windows(2).all(|w| w[0].0 <= w[1].0),
            "progress goes up"
        );
        assert_eq!(seen.last().unwrap(), &(f.size, f.size, f.size));
        assert_eq!(srv.ranges.lock().unwrap().as_slice(), &[None]);
    }

    #[test]
    fn resumes_from_part_file() {
        let data = body();
        let srv = serve(data.clone(), Mode::Ranges);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, true);
        std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
        std::fs::write(f.part_path(), &data[..100_000]).unwrap();
        let mut first = None;
        download(&agent(), &f, &mut |n| {
            first.get_or_insert(n);
        })
        .unwrap();
        assert_eq!(first, Some(100_000), "progress starts at the resumed size");
        assert_eq!(std::fs::read(&f.dest).unwrap(), data);
        assert_eq!(
            srv.ranges.lock().unwrap().as_slice(),
            &[Some("bytes=100000-".to_string())]
        );
    }

    #[test]
    fn restarts_when_server_ignores_range() {
        let data = body();
        let srv = serve(data.clone(), Mode::NoRanges);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, true);
        std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
        // Garbage in the part file must not survive a 200 answer.
        std::fs::write(f.part_path(), vec![0xAAu8; 50_000]).unwrap();
        download(&agent(), &f, &mut |_| {}).unwrap();
        assert_eq!(std::fs::read(&f.dest).unwrap(), data);
    }

    #[test]
    fn wrong_range_fails_and_drops_part() {
        let data = body();
        let srv = serve(data.clone(), Mode::WrongRange);
        let t = tempfile::tempdir().unwrap();
        // Unpinned: without the range check a body of the right length would be accepted.
        let f = file_for(t.path(), &srv.url, &data, false);
        std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
        std::fs::write(f.part_path(), &data[..100_000]).unwrap();
        let err = download(&agent(), &f, &mut |_| {}).unwrap_err();
        assert!(format!("{err:#}").contains("different range"), "{err:#}");
        assert!(!f.dest.exists());
        assert!(!f.part_path().exists());
    }

    #[test]
    fn checksum_mismatch_retries_once_then_fails() {
        let data = body();
        let srv = serve(data.clone(), Mode::Corrupt);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, true);
        let err = download(&agent(), &f, &mut |_| {}).unwrap_err();
        assert!(format!("{err:#}").contains("sha256 mismatch"), "{err:#}");
        assert!(!f.dest.exists());
        assert!(!f.part_path().exists());
        assert_eq!(srv.ranges.lock().unwrap().len(), 2, "one retry");
    }

    #[test]
    fn short_body_keeps_part_for_resume() {
        let data = body();
        let srv = serve(data.clone(), Mode::Truncate);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, false);
        assert!(download(&agent(), &f, &mut |_| {}).is_err());
        assert!(!f.dest.exists());
        let kept = std::fs::metadata(f.part_path())
            .map(|m| m.len())
            .unwrap_or(0);
        assert!(kept > 0 && kept < f.size, "partial bytes are kept: {kept}");
        // A later call resumes from the part file against a well-behaved server.
        let good = serve(data.clone(), Mode::Ranges);
        let f2 = ModelFile {
            url: good.url.clone(),
            ..f.clone()
        };
        download(&agent(), &f2, &mut |_| {}).unwrap();
        assert_eq!(std::fs::read(&f.dest).unwrap(), data);
        assert_eq!(
            good.ranges.lock().unwrap().as_slice(),
            &[Some(format!("bytes={kept}-"))]
        );
    }

    #[test]
    fn oversized_part_restarts() {
        let data = body();
        let srv = serve(data.clone(), Mode::Ranges);
        let t = tempfile::tempdir().unwrap();
        let f = file_for(t.path(), &srv.url, &data, true);
        std::fs::create_dir_all(f.dest.parent().unwrap()).unwrap();
        std::fs::write(f.part_path(), vec![1u8; data.len() + 10]).unwrap();
        download(&agent(), &f, &mut |_| {}).unwrap();
        assert_eq!(std::fs::read(&f.dest).unwrap(), data);
        assert_eq!(srv.ranges.lock().unwrap().as_slice(), &[None]);
    }

    #[test]
    fn present_files_are_not_fetched() {
        let t = tempfile::tempdir().unwrap();
        let f = ModelFile {
            label: "x".into(),
            // Nothing listens here; a request would fail the test.
            url: "http://127.0.0.1:9/never".into(),
            dest: t.path().join("x.bin"),
            size: 3,
            sha256: None,
        };
        std::fs::write(&f.dest, b"abc").unwrap();
        let mut calls = 0;
        prefetch_files(&[f], |p| {
            calls += 1;
            assert_eq!(p.done, 3);
        })
        .unwrap();
        assert_eq!(calls, 1);
    }

    fn settings(models_dir: Option<&Path>, embedder: &str) -> ModelSettings {
        ModelSettings {
            models_dir: models_dir.map(Path::to_path_buf),
            detector: "deim-dfine-s".into(),
            embedder: embedder.into(),
            ..ModelSettings::default()
        }
    }

    #[test]
    fn flat_layout_matches_the_loaders() {
        let d = Path::new("/m");
        let files = plan_in(&settings(Some(d), "siglip2-b16-256"), None).unwrap();
        let dests: Vec<_> = files.iter().map(|f| f.dest.clone()).collect();
        assert_eq!(
            dests,
            vec![
                d.join("deim/dfine-s-coco.onnx"),
                d.join("siglip2-b16-256/tokenizer.json"),
                d.join("siglip2-b16-256/tokenizer_config.json"),
                d.join("siglip2-b16-256/special_tokens_map.json"),
                d.join("siglip2-b16-256/text_model.onnx"),
                d.join("siglip2-b16-256/vision_model.onnx"),
            ]
        );
        assert!(files.iter().all(|f| f.sha256.is_some()));
        assert!(files[1..].iter().all(|f| {
            f.url
                .contains("/resolve/d1114256522a37ffa257a0a58017348ab0058db2/")
        }));
        // The so400m directory name is the one siglip_local uses.
        let so = plan_in(&settings(Some(d), "siglip2-so400m-384"), None).unwrap();
        assert_eq!(
            so.last().unwrap().dest,
            d.join("siglip2-so400m-patch16-384/vision_model.onnx")
        );
    }

    #[test]
    fn usls_layout_without_models_dir() {
        let c = Path::new("/cache/usls");
        let files = plan_in(&settings(None, "siglip2-b16-256"), Some(c)).unwrap();
        assert_eq!(
            files[0].dest,
            c.join("jamjamjon/assets/deim/dfine-s-coco.onnx")
        );
        assert_eq!(
            files[0].url,
            "https://github.com/jamjamjon/assets/releases/download/deim/dfine-s-coco.onnx"
        );
        assert_eq!(
            files.last().unwrap().dest,
            c.join("onnx-community/siglip2-base-patch16-256-ONNX/onnx/vision_model.onnx")
        );
        assert_eq!(
            files[1].dest,
            c.join("onnx-community/siglip2-base-patch16-256-ONNX/tokenizer.json")
        );
        let clip = plan_in(&settings(None, "clip-b16"), Some(c)).unwrap();
        assert_eq!(
            clip.last().unwrap().dest,
            c.join("jamjamjon/assets/clip/vit-b16-visual.onnx")
        );
    }

    #[test]
    fn custom_detector_and_sidecar_need_nothing() {
        let mut s = settings(Some(Path::new("/m")), "gemma2");
        s.detector = "/x/my.onnx".into();
        assert!(plan_in(&s, None).unwrap().is_empty());
        assert!(plan_in(&settings(Some(Path::new("/m")), "nope"), None).is_err());
    }

    #[test]
    fn gemma_native_files() {
        let d = Path::new("/m");
        let r = plan_in(&settings(Some(d), "gemma2-native:256"), None);
        if cfg!(feature = "gemma-native") {
            let files = r.unwrap();
            assert_eq!(
                files[1..]
                    .iter()
                    .map(|f| f.dest.clone())
                    .collect::<Vec<_>>(),
                vec![
                    d.join("embeddinggemma-2/embeddinggemma-2-Q8_0.gguf"),
                    d.join("embeddinggemma-2/mmproj-embeddinggemma-2-Q8_0.gguf"),
                ]
            );
        } else {
            assert!(r.is_err());
        }
    }

    #[test]
    fn sha256_of_known_input() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("abc");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
