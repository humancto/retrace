//! EmbeddingGemma 2 in-process: llama.cpp runs the text model, mtmd runs the image and audio
//! encoders, and the encoder rows go into the text model's batch between `<bos>` and `<eos>`, the
//! way llama-server builds mixed batches for chat. Mean-pooled, L2-normalised, Matryoshka-truncated.
//! Same framing DigUp validated against the mlx-vlm reference (cosine ≥ 0.999 on text and images).
//!
//! One `Engine` is one llama context: not `Sync`, use it from one thread (or behind a mutex).

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code
)]

mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use anyhow::{Context, Result, bail, ensure};
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::Once;

static BACKEND: Once = Once::new();

unsafe extern "C" fn quiet_log(
    level: ffi::ggml_log_level,
    text: *const c_char,
    _user: *mut c_void,
) {
    // Only errors reach stderr; llama.cpp is chatty at info level.
    if level == ffi::GGML_LOG_LEVEL_ERROR && !text.is_null() {
        let s = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        eprint!("[llama] {s}");
    }
}

/// How an input is framed. Text gets the model's retrieval prompts; media gets no prefix.
pub enum Input<'a> {
    /// Raw text (already prefixed by the caller, e.g. `task: search result | query: …`).
    Text(&'a str),
    /// RGB8 images, interleaved into one input: `nx`, `ny`, pixels (nx*ny*3 bytes).
    Images(Vec<(u32, u32, &'a [u8])>),
    /// Mono PCM in [-1, 1] at `sample_rate` Hz (resampled to the model's rate if needed).
    Audio {
        samples: &'a [f32],
        sample_rate: u32,
    },
}

pub struct Engine {
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
    media: *mut ffi::mtmd_context,
    batch: *mut ffi::llama_batch_ext,
    vocab: *const ffi::llama_vocab,
    n_embd_out: usize,
    n_embd_inp: usize,
    n_ctx: usize,
    dim: usize,
    audio_rate: u32,
    pub has_vision: bool,
    pub has_audio: bool,
}

impl Engine {
    /// Loads the text model (`text_gguf`) and, when given, the mmproj with the image/audio encoders.
    /// `dim` truncates the 768-d output (Matryoshka: 768 | 512 | 256 | 128).
    pub fn load(
        text_gguf: &str,
        mmproj_gguf: Option<&str>,
        dim: usize,
        n_threads: usize,
        use_gpu: bool,
    ) -> Result<Self> {
        ensure!(
            [768usize, 512, 256, 128].contains(&dim),
            "dim must be 768, 512, 256 or 128"
        );
        BACKEND.call_once(|| unsafe {
            ffi::llama_log_set(Some(quiet_log), std::ptr::null_mut());
            ffi::mtmd_helper_log_set(Some(quiet_log), std::ptr::null_mut());
            ffi::llama_backend_init();
        });
        let n_threads = n_threads.max(1) as i32;
        let n_ctx: u32 = 2048; // one input must fit one micro-batch (holds a 1120-token image or 30 s of audio)
        unsafe {
            let mut mp = ffi::llama_model_default_params();
            mp.n_gpu_layers = if use_gpu { 99 } else { 0 };
            let path = CString::new(text_gguf)?;
            let model = ffi::llama_model_load_from_file(path.as_ptr(), mp);
            ensure!(!model.is_null(), "could not load {text_gguf}");

            let mut cp = ffi::llama_context_default_params();
            cp.n_ctx = n_ctx;
            cp.n_batch = n_ctx;
            cp.n_ubatch = n_ctx;
            cp.n_seq_max = 1;
            cp.n_threads = n_threads;
            cp.n_threads_batch = n_threads;
            cp.embeddings = true;
            cp.pooling_type = ffi::LLAMA_POOLING_TYPE_MEAN;
            let ctx = ffi::llama_init_from_model(model, cp);
            if ctx.is_null() {
                ffi::llama_model_free(model);
                bail!("could not create a llama context");
            }

            let mut media = std::ptr::null_mut();
            let (mut has_vision, mut has_audio, mut audio_rate) = (false, false, 16_000u32);
            if let Some(mm) = mmproj_gguf {
                let mut p = ffi::mtmd_context_params_default();
                p.use_gpu = use_gpu;
                p.n_threads = n_threads;
                p.print_timings = false;
                p.warmup = false;
                let mpath = CString::new(mm)?;
                media = ffi::mtmd_init_from_file(mpath.as_ptr(), model, p);
                if media.is_null() {
                    ffi::llama_free(ctx);
                    ffi::llama_model_free(model);
                    bail!("could not load mmproj {mm}");
                }
                has_vision = ffi::mtmd_support_vision(media);
                has_audio = ffi::mtmd_support_audio(media);
                if has_audio {
                    let r = ffi::mtmd_get_audio_sample_rate(media);
                    if r > 0 {
                        audio_rate = r as u32;
                    }
                }
            }
            let batch = ffi::llama_batch_ext_init(ctx);
            let vocab = ffi::llama_model_get_vocab(model);
            Ok(Self {
                model,
                ctx,
                media,
                batch,
                vocab,
                n_embd_out: ffi::llama_model_n_embd_out(model) as usize,
                n_embd_inp: ffi::llama_model_n_embd_inp(model) as usize,
                n_ctx: n_ctx as usize,
                dim,
                audio_rate,
                has_vision,
                has_audio,
            })
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_embd_out(&self) -> usize {
        self.n_embd_out
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<ffi::llama_token>> {
        let bytes = text.as_bytes();
        let mut toks = vec![0 as ffi::llama_token; bytes.len() + 16];
        let n = unsafe {
            ffi::llama_tokenize(
                self.vocab,
                bytes.as_ptr() as *const c_char,
                bytes.len() as i32,
                toks.as_mut_ptr(),
                toks.len() as i32,
                add_special,
                true,
            )
        };
        let n = if n < 0 {
            toks.resize((-n) as usize, 0);
            let m = unsafe {
                ffi::llama_tokenize(
                    self.vocab,
                    bytes.as_ptr() as *const c_char,
                    bytes.len() as i32,
                    toks.as_mut_ptr(),
                    toks.len() as i32,
                    add_special,
                    true,
                )
            };
            ensure!(m >= 0, "tokenize failed");
            m as usize
        } else {
            n as usize
        };
        toks.truncate(n);
        // Keep room for <eos>: a too-long text is cut and still closed.
        if toks.len() > self.n_ctx - 1 {
            toks.truncate(self.n_ctx - 1);
            toks.push(unsafe { ffi::llama_vocab_eos(self.vocab) });
        }
        Ok(toks)
    }

    unsafe fn place(&self, idx: i32, pos: &mut i32) -> Result<()> {
        ensure!(
            idx >= 0,
            "batch full (input longer than {} positions)",
            self.n_ctx
        );
        let p: [ffi::llama_pos; 4] = [*pos, 0, 0, 0];
        unsafe {
            ensure!(
                ffi::llama_batch_ext_set_pos(self.batch, idx, p.as_ptr()),
                "set_pos"
            );
            ensure!(
                ffi::llama_batch_ext_set_output_embd(self.batch, idx, true),
                "set_output_embd"
            );
        }
        *pos += 1;
        Ok(())
    }

    unsafe fn add_tokens(&self, toks: &[ffi::llama_token], pos: &mut i32) -> Result<()> {
        for &t in toks {
            let idx = unsafe { ffi::llama_batch_ext_add_token(self.batch, 0, t) };
            unsafe { self.place(idx, pos)? };
        }
        Ok(())
    }

    unsafe fn add_rows(&self, rows: *const f32, n_rows: usize, pos: &mut i32) -> Result<()> {
        for r in 0..n_rows {
            let e = ffi::llama_embd {
                data: unsafe { rows.add(r * self.n_embd_inp) },
                n_rows: 1,
                n_embd: self.n_embd_inp,
            };
            let idx = unsafe { ffi::llama_batch_ext_add_embd(self.batch, 0, e) };
            unsafe { self.place(idx, pos)? };
        }
        Ok(())
    }

    /// Runs the batch through the text model and returns the pooled, normalised, truncated vector.
    unsafe fn run(&self) -> Result<Vec<f32>> {
        let status =
            unsafe { ffi::llama_process(self.ctx, ffi::LLAMA_PROCESS_TYPE_ENCODE, self.batch) };
        unsafe { ffi::llama_batch_ext_clear(self.batch) };
        ensure!(status == 0, "llama_process failed ({status})");
        let p = unsafe { ffi::llama_get_embeddings_seq(self.ctx, 0) };
        ensure!(!p.is_null(), "no pooled embedding");
        let full = unsafe { std::slice::from_raw_parts(p, self.n_embd_out) };
        let mut v: Vec<f32> = full[..self.dim.min(self.n_embd_out)].to_vec();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= n;
        }
        Ok(v)
    }

    /// Embeds one input. Text is framed `<bos> text <eos>`; media as `<bos> <|media> rows <media|> <eos>`.
    pub fn embed(&mut self, input: Input<'_>) -> Result<Vec<f32>> {
        unsafe { ffi::llama_batch_ext_clear(self.batch) };
        let mut pos: i32 = 0;
        match input {
            Input::Text(t) => {
                let toks = self.tokenize(t, true)?;
                unsafe { self.add_tokens(&toks, &mut pos)? };
            }
            Input::Images(imgs) => {
                ensure!(self.has_vision, "mmproj has no vision encoder");
                ensure!(!imgs.is_empty(), "no images");
                for (nx, ny, px) in imgs.iter() {
                    let need = *nx as usize * *ny as usize * 3;
                    ensure!(
                        px.len() == need,
                        "image buffer is {} bytes, expected {nx}x{ny}x3 = {need}",
                        px.len()
                    );
                }
                // SAFETY: every buffer is exactly nx*ny*3 RGB bytes (checked above); mtmd copies it.
                let bitmaps: Vec<*mut ffi::mtmd_bitmap> = imgs
                    .iter()
                    .map(|(nx, ny, px)| unsafe { ffi::mtmd_bitmap_init(*nx, *ny, px.as_ptr()) })
                    .collect();
                let r = unsafe { self.embed_media(&bitmaps, &mut pos) };
                for b in bitmaps {
                    unsafe { ffi::mtmd_bitmap_free(b) };
                }
                r?;
            }
            Input::Audio {
                samples,
                sample_rate,
            } => {
                ensure!(self.has_audio, "mmproj has no audio encoder");
                let pcm = resample(samples, sample_rate, self.audio_rate);
                let max = 30 * self.audio_rate as usize; // the reference cuts audio at 30 s
                let pcm = &pcm[..pcm.len().min(max)];
                ensure!(!pcm.is_empty(), "empty audio");
                let b = unsafe { ffi::mtmd_bitmap_init_from_audio(pcm.len(), pcm.as_ptr()) };
                let r = unsafe { self.embed_media(&[b], &mut pos) };
                unsafe { ffi::mtmd_bitmap_free(b) };
                r?;
            }
        }
        unsafe { self.run() }
    }

    /// Tokenises one media marker per bitmap (with special tokens), encodes media chunks with mtmd
    /// and adds text tokens / encoder rows to the batch in order.
    unsafe fn embed_media(&self, bitmaps: &[*mut ffi::mtmd_bitmap], pos: &mut i32) -> Result<()> {
        for b in bitmaps {
            ensure!(!b.is_null(), "could not build a media bitmap");
        }
        let marker = unsafe { CStr::from_ptr(ffi::mtmd_default_marker()) }
            .to_str()
            .unwrap_or("<__media__>");
        let prompt = vec![marker; bitmaps.len()].join(" ");
        let cprompt = CString::new(prompt)?;
        let text = ffi::mtmd_input_text {
            text: cprompt.as_ptr(),
            text_len: cprompt.as_bytes().len(),
            add_special: true,
            parse_special: true,
        };
        let chunks = unsafe { ffi::mtmd_input_chunks_init() };
        let status = unsafe {
            ffi::mtmd_tokenize(
                self.media,
                chunks,
                &text,
                bitmaps.as_ptr() as *const *const ffi::mtmd_bitmap,
                bitmaps.len(),
            )
        };
        let result = (|| -> Result<()> {
            ensure!(status == 0, "mtmd_tokenize failed ({status})");
            let n = unsafe { ffi::mtmd_input_chunks_size(chunks) };
            for i in 0..n {
                let chunk = unsafe { ffi::mtmd_input_chunks_get(chunks, i) };
                let ty = unsafe { ffi::mtmd_input_chunk_get_type(chunk) };
                if ty == ffi::MTMD_INPUT_CHUNK_TYPE_TEXT {
                    let mut count: usize = 0;
                    let toks = unsafe { ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut count) };
                    if !toks.is_null() && count > 0 {
                        let s = unsafe { std::slice::from_raw_parts(toks, count) };
                        unsafe { self.add_tokens(s, pos)? };
                    }
                } else {
                    let rc = unsafe { ffi::mtmd_encode_chunk(self.media, chunk) };
                    ensure!(rc == 0, "mtmd_encode_chunk failed ({rc})");
                    let rows = unsafe { ffi::mtmd_get_output_embd(self.media) };
                    ensure!(!rows.is_null(), "encoder produced no rows");
                    let n_tokens = unsafe { ffi::mtmd_input_chunk_get_n_tokens(chunk) };
                    unsafe { self.add_rows(rows, n_tokens, pos)? };
                }
            }
            Ok(())
        })();
        unsafe { ffi::mtmd_input_chunks_free(chunks) };
        result
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if !self.batch.is_null() {
                ffi::llama_batch_ext_free(self.batch);
            }
            if !self.media.is_null() {
                ffi::mtmd_free(self.media);
            }
            if !self.ctx.is_null() {
                ffi::llama_free(self.ctx);
            }
            if !self.model.is_null() {
                ffi::llama_model_free(self.model);
            }
        }
    }
}

// The engine owns raw llama.cpp handles; moving it between threads is fine, sharing is not.
unsafe impl Send for Engine {}

/// Linear resampling (good enough for speech-band audio between 16 and 24 kHz).
fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = ((x.len() as f64) / ratio) as usize;
    (0..n)
        .map(|i| {
            let s = i as f64 * ratio;
            let j = s.floor() as usize;
            let f = (s - j as f64) as f32;
            let a = x[j.min(x.len() - 1)];
            let b = x[(j + 1).min(x.len() - 1)];
            a + (b - a) * f
        })
        .collect()
}

/// Where the model files live by default and how to get them.
pub const MODEL_REPO: &str = "ggml-org/embeddinggemma-2-GGUF";

pub fn default_model_dir() -> std::path::PathBuf {
    std::env::var("RETRACE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("models"))
        .join("embeddinggemma-2")
}

/// Finds `*.gguf` text model and `mmproj*.gguf` in a directory.
pub fn find_model_files(dir: &std::path::Path) -> Result<(String, Option<String>)> {
    let mut text = None;
    let mut mmproj = None;
    ensure!(
        dir.is_dir(),
        "no EmbeddingGemma 2 model directory at {} — run scripts/get-gemma.sh (downloads {MODEL_REPO}, ~865 MB)",
        dir.display()
    );
    for e in std::fs::read_dir(dir)
        .with_context(|| format!("model dir {}", dir.display()))?
        .flatten()
    {
        let name = e.file_name().to_string_lossy().to_lowercase();
        if !name.ends_with(".gguf") {
            continue;
        }
        if name.starts_with("mmproj") {
            mmproj = Some(e.path().to_string_lossy().into_owned());
        } else {
            text = Some(e.path().to_string_lossy().into_owned());
        }
    }
    let text = text.with_context(|| {
        format!(
            "no text model .gguf in {} (download from https://huggingface.co/{MODEL_REPO})",
            dir.display()
        )
    })?;
    Ok((text, mmproj))
}
