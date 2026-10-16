//! Audio windows. ffmpeg decodes a source's audio to 16 kHz mono PCM; every `hop` seconds the last
//! `win` seconds are embedded with EmbeddingGemma 2's audio encoder (through the sidecar) and stored
//! in the same vector space as the crops. A text query such as "glass breaking", "a dog barking" or
//! "someone shouting" then returns audio windows next to the visual hits, with no fixed list of
//! sound classes. Only runs when the embedder reports audio support.

use crate::db::Db;
use crate::models::{Embedder, ModelSettings};
use anyhow::{Context, Result};
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const SAMPLE_RATE: usize = 16_000;

#[derive(Debug, Clone)]
pub struct AudioOpts {
    /// Resolved media URL or file path.
    pub url: String,
    pub cam: String,
    pub live: bool,
    /// Wall-clock start of a file source (ms); ignored for live sources.
    pub t_start: i64,
    pub db_path: std::path::PathBuf,
    pub models: ModelSettings,
    pub win_s: f32,
    pub hop_s: f32,
    /// Windows quieter than this RMS (full scale = 1.0) are skipped as silence.
    pub min_rms: f32,
}

/// True when ffprobe sees an audio stream.
pub fn has_audio(url: &str) -> bool {
    let mut c = std::process::Command::new("ffprobe");
    c.args(crate::ingest::user_agent_args(url));
    if crate::ingest::is_http(url) {
        // Fail the probe when the server sends nothing for 20 s, instead of blocking the ingest.
        c.args(["-rw_timeout", "20000000"]);
    }
    let out = c
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            url,
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains("audio"),
        Err(_) => false,
    }
}

/// Minimal PCM16 mono WAV container around raw samples.
pub fn wav_bytes(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + data_len as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&1u16.to_le_bytes()); // mono
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

/// Parses a PCM16 mono WAV (as written by `wav_bytes`) into float samples and its sample rate.
#[cfg_attr(not(feature = "gemma-native"), allow(dead_code))]
pub fn wav_to_f32(wav: &[u8]) -> Result<(u32, Vec<f32>)> {
    anyhow::ensure!(
        wav.len() >= 44 && &wav[..4] == b"RIFF" && &wav[8..12] == b"WAVE",
        "not a WAV"
    );
    let rate = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
    let mut i = 12;
    while i + 8 <= wav.len() {
        let id = &wav[i..i + 4];
        let len = u32::from_le_bytes([wav[i + 4], wav[i + 5], wav[i + 6], wav[i + 7]]) as usize;
        if id == b"data" {
            let end = (i + 8 + len).min(wav.len());
            let pcm = wav[i + 8..end]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
                .collect();
            return Ok((rate, pcm));
        }
        i += 8 + len + (len & 1);
    }
    anyhow::bail!("WAV without a data chunk")
}

fn rms(pcm: &[i16]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let sum: f64 = pcm.iter().map(|s| (*s as f64 / 32768.0).powi(2)).sum();
    (sum / pcm.len() as f64).sqrt() as f32
}

/// Decodes, windows and embeds until the stream ends or `stop` is set. Returns windows stored.
/// The ffmpeg child's pid is published through `pid` so the caller can end a live stream early.
pub fn run_audio(
    opts: &AudioOpts,
    stop: Arc<AtomicBool>,
    pid: Arc<std::sync::Mutex<Option<u32>>>,
) -> Result<usize> {
    let db = Db::open(&opts.db_path)?;
    let mut emb = Embedder::load(&opts.models)?;
    anyhow::ensure!(emb.audio, "embedder has no audio encoder");
    let cam_id = db
        .camera_by_name(&opts.cam)?
        .map(|c| c.id)
        .context("camera row missing")?;

    let mut c = std::process::Command::new("ffmpeg");
    c.args(["-nostdin", "-hide_banner", "-loglevel", "error"]);
    if opts.live {
        c.args(crate::ingest::live_input_args(&opts.url));
    }
    c.args(["-i", &opts.url])
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-f",
            "s16le",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    let mut child = c.spawn().context("spawn ffmpeg (audio)")?;
    if let Ok(mut p) = pid.lock() {
        *p = Some(child.id());
    }
    let mut out = child.stdout.take().context("ffmpeg audio stdout")?;

    let win = (opts.win_s * SAMPLE_RATE as f32) as usize;
    let hop = (opts.hop_s * SAMPLE_RATE as f32) as usize;
    let mut ring: Vec<i16> = Vec::with_capacity(win + hop);
    let mut chunk = vec![0u8; hop * 2];
    let mut samples_total: usize = 0;
    let mut anchor_ms: Option<i64> = None;
    let mut stored = 0usize;
    tracing::info!(cam = %opts.cam, win_s = opts.win_s, hop_s = opts.hop_s, "audio windows start");

    'outer: loop {
        // Fill one hop of samples (ffmpeg writes in arbitrary sizes).
        let mut got = 0usize;
        while got < chunk.len() {
            if stop.load(Ordering::Relaxed) {
                break 'outer;
            }
            match out.read(&mut chunk[got..]) {
                Ok(0) => break 'outer,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(cam = %opts.cam, "audio read: {e}");
                    break 'outer;
                }
            }
        }
        if got == 0 {
            break;
        }
        let now = chrono::Utc::now().timestamp_millis();
        let anchor = *anchor_ms.get_or_insert(if opts.live {
            now - (got / 2) as i64 * 1000 / SAMPLE_RATE as i64
        } else {
            opts.t_start
        });
        let pcm: Vec<i16> = chunk[..got]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes(*b))
            .collect();
        samples_total += pcm.len();
        ring.extend_from_slice(&pcm);
        if ring.len() > win {
            let drop = ring.len() - win;
            ring.drain(..drop);
        }
        if ring.len() < win / 2 {
            continue;
        }
        let t1 = anchor + (samples_total as i64) * 1000 / SAMPLE_RATE as i64;
        let t0 = t1 - (ring.len() as i64) * 1000 / SAMPLE_RATE as i64;
        let level = rms(&ring);
        if level < opts.min_rms {
            continue;
        }
        let wav = wav_bytes(&ring, SAMPLE_RATE as u32);
        match emb.embed_audio_wav(&wav) {
            Ok(v) if !v.is_empty() => {
                db.insert_audio_window(cam_id, t0, t1, level as f64, &v)?;
                stored += 1;
                if stored % 20 == 1 {
                    tracing::info!(cam = %opts.cam, windows = stored, rms = format!("{level:.3}"), "audio");
                }
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(cam = %opts.cam, "audio embed failed: {e:#}"),
        }
    }
    // Clear the published pid before the reap, so a caller cannot signal a pid that the kernel
    // has already given to another process.
    if let Ok(mut p) = pid.lock() {
        *p = None;
    }
    let _ = child.kill();
    let _ = child.wait();
    tracing::info!(cam = %opts.cam, windows = stored, "audio windows done");
    Ok(stored)
}
