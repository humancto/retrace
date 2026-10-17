//! Model loading (detector + embedder). The detector and the CLIP/SigLIP 2 embedders run in-process
//! through usls/onnxruntime; model files resolve as a local path, a `models_dir` override, or a
//! download (GitHub release URL, or Hugging Face for SigLIP 2). The `gemma2` embedder is
//! EmbeddingGemma 2 served by the local Python sidecar (`sidecar/embed_server.py`): one vector
//! space for text, images, image sets and audio, which is what makes audio search possible.

use anyhow::{Context, Result};
use base64::Engine;
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::time::Duration;
use usls::{
    Config, DType, Device, Image, Model, Runtime,
    models::{Clip, DEIM},
};

pub mod fetch;
pub use fetch::{ModelFile, Progress, prefetch};

pub const COCO_PERSON_VEHICLE: &[usize] = &[0, 1, 2, 3, 5, 7]; // person, bicycle, car, motorcycle, bus, truck

#[derive(Debug, Clone)]
pub struct ModelSettings {
    pub models_dir: Option<PathBuf>,
    pub detector: String, // "deim-dfine-s" | "deim-dfine-n" | path/URL to an ONNX file
    pub embedder: String, // "clip-b16" | "siglip2-b16-256" | "siglip2-so400m-384" | "gemma2"
    pub device: String,   // "cpu" | "coreml"
    pub conf: f32,
    pub threads: usize,
    /// Sidecar URL for the `gemma2` embedder.
    pub embed_url: String,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            models_dir: None,
            detector: "deim-dfine-s".into(),
            embedder: "clip-b16".into(),
            device: "cpu".into(),
            conf: 0.35,
            threads: 0,
            embed_url: "http://127.0.0.1:8082".into(),
        }
    }
}

fn device(s: &str) -> Result<Device> {
    // `cpu-only` is the documented spelling for "keep everything off the GPU" (it also controls
    // the in-process EmbeddingGemma path); for onnxruntime it is plain CPU.
    let s = if s.eq_ignore_ascii_case("cpu-only") {
        "cpu"
    } else {
        s
    };
    s.parse::<Device>()
        .map_err(|e| anyhow::anyhow!("bad device '{s}': {e:?}"))
}

/// Prefer a local file under models_dir, else the given remote spec.
fn resolve(models_dir: &Option<PathBuf>, local_rel: &str, remote: &str) -> String {
    if let Some(d) = models_dir {
        let p = d.join(local_rel);
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    if Path::new(remote).exists() {
        return remote.to_string();
    }
    remote.to_string()
}

const GH: &str = "https://github.com/jamjamjon/assets/releases/download";

pub struct Detector {
    rt: Runtime<DEIM>,
    pub name: String,
}

impl Detector {
    pub fn load(s: &ModelSettings) -> Result<Self> {
        let (name, file) = match s.detector.as_str() {
            "deim-dfine-s" => (
                "deim-dfine-s",
                resolve(
                    &s.models_dir,
                    "deim/dfine-s-coco.onnx",
                    &format!("{GH}/deim/dfine-s-coco.onnx"),
                ),
            ),
            "deim-dfine-n" => (
                "deim-dfine-n",
                resolve(
                    &s.models_dir,
                    "deim/dfine-n-coco.onnx",
                    &format!("{GH}/deim/dfine-n-coco.onnx"),
                ),
            ),
            "deim-dfine-m" => (
                "deim-dfine-m",
                resolve(
                    &s.models_dir,
                    "deim/dfine-m-coco.onnx",
                    &format!("{GH}/deim/dfine-m-coco.onnx"),
                ),
            ),
            other => ("custom", other.to_string()),
        };
        let mut cfg = Config::deim_dfine_s_coco()
            .with_model_file(&file)
            .with_model_device(device(&s.device)?)
            .with_model_dtype(DType::Fp32)
            // The detector keeps low-confidence boxes too: the tracker uses them only to extend
            // existing tracks (ByteTrack's second stage); new tracks need >= s.conf.
            .with_class_confs(&[(s.conf * 0.5).max(0.1)])
            .retain_classes(COCO_PERSON_VEHICLE)
            .with_num_dry_run_all(0);
        if s.threads > 0 {
            cfg = cfg.with_num_intra_threads_all(s.threads);
        }
        let cfg = cfg.commit().context("detector config")?;
        let rt = DEIM::new(cfg).context("load detector")?;
        tracing::info!(detector = name, file = %file, "detector ready");
        Ok(Self {
            rt,
            name: name.into(),
        })
    }

    /// Returns boxes (xmin, ymin, w, h, conf, class name) in pixel units of the given image.
    pub fn detect(&mut self, img: &Image) -> Result<Vec<usls::Hbb>> {
        let ys = self.rt.forward(std::slice::from_ref(img))?;
        Ok(ys.into_iter().next().map(|y| y.hbbs).unwrap_or_default())
    }
}

enum Inner {
    Onnx(Runtime<Clip>),
    Remote {
        url: String,
        agent: ureq::Agent,
    },
    #[cfg(feature = "gemma-native")]
    Native(llama_embed::Engine),
    /// The model was freed by [`Embedder::release`].
    Released,
}

pub struct Embedder {
    inner: Inner,
    pub name: String,
    pub dim: usize,
    /// True when the embedder can embed audio (EmbeddingGemma 2 with its audio encoder loaded).
    pub audio: bool,
}

/// One request item for the sidecar. `kind`: query | doc | image | images | audio.
#[derive(serde::Serialize)]
struct Item<'a> {
    kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    b64: Option<Vec<String>>,
}

fn jpeg_b64(img: &RgbImage) -> Result<String> {
    let mut buf = Vec::with_capacity(32 * 1024);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
    enc.encode_image(img)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf))
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= n;
    }
    v
}

impl Embedder {
    pub fn load(s: &ModelSettings) -> Result<Self> {
        if s.embedder == "gemma2" {
            return Self::load_remote(&s.embed_url);
        }
        if s.embedder == "gemma2-native" || s.embedder.starts_with("gemma2-native:") {
            return Self::load_native(s);
        }
        let dev = device(&s.device)?;
        let (name, cfg) = match s.embedder.as_str() {
            "clip-b16" => (
                "clip-vit-b16",
                Config::clip_vit_b16()
                    .with_visual_file(resolve(
                        &s.models_dir,
                        "clip/vit-b16-visual.onnx",
                        &format!("{GH}/clip/vit-b16-visual.onnx"),
                    ))
                    .with_textual_file(resolve(
                        &s.models_dir,
                        "clip/vit-b16-textual.onnx",
                        &format!("{GH}/clip/vit-b16-textual.onnx"),
                    ))
                    .with_tokenizer_file(resolve(
                        &s.models_dir,
                        "clip/tokenizer.json",
                        &format!("{GH}/clip/tokenizer.json"),
                    ))
                    .with_tokenizer_config_file(resolve(
                        &s.models_dir,
                        "clip/tokenizer_config.json",
                        &format!("{GH}/clip/tokenizer_config.json"),
                    ))
                    .with_special_tokens_map_file(resolve(
                        &s.models_dir,
                        "clip/special_tokens_map.json",
                        &format!("{GH}/clip/special_tokens_map.json"),
                    )),
            ),
            "siglip2-b16-256" => (
                "siglip2-b16-256",
                siglip_local(Config::siglip2_b16_256(), &s.models_dir, "siglip2-b16-256"),
            ),
            "siglip2-b16-384" => (
                "siglip2-b16-384",
                siglip_local(Config::siglip2_b16_384(), &s.models_dir, "siglip2-b16-384"),
            ),
            "siglip2-so400m-384" => (
                "siglip2-so400m-patch16-384",
                siglip_local(
                    Config::siglip2_so400m_patch16_384(),
                    &s.models_dir,
                    "siglip2-so400m-patch16-384",
                ),
            ),
            other => anyhow::bail!(
                "unknown embedder '{other}' (clip-b16 | siglip2-b16-256 | siglip2-b16-384 | siglip2-so400m-384 | gemma2 | gemma2-native[:dim])"
            ),
        };
        let mut cfg = cfg
            .with_visual_device(dev)
            .with_textual_device(dev)
            .with_visual_dtype(DType::Fp32)
            .with_textual_dtype(DType::Fp32)
            .with_num_dry_run_all(0);
        if s.threads > 0 {
            cfg = cfg.with_num_intra_threads_all(s.threads);
        }
        let cfg = cfg.commit().context("embedder config")?;
        let mut rt = Clip::new(cfg).context("load embedder")?;
        let probe = rt.encode_texts(&["a photo of a person"])?.embedding;
        let dim = *probe.shape().last().unwrap_or(&0);
        tracing::info!(embedder = name, dim, "embedder ready");
        Ok(Self {
            inner: Inner::Onnx(rt),
            name: name.into(),
            dim,
            audio: false,
        })
    }

    /// EmbeddingGemma 2 inside the process (llama.cpp + mtmd). Model files: `<models_dir>/embeddinggemma-2/`
    /// (`*.gguf` text model + `mmproj*.gguf`), from https://huggingface.co/ggml-org/embeddinggemma-2-GGUF.
    /// `gemma2-native:512` picks the Matryoshka dimension (default 256).
    #[cfg(feature = "gemma-native")]
    fn load_native(s: &ModelSettings) -> Result<Self> {
        let dim: usize = s
            .embedder
            .split_once(':')
            .and_then(|(_, d)| d.parse().ok())
            .unwrap_or(256);
        let dir = s
            .models_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("models"))
            .join("embeddinggemma-2");
        let (text, mmproj) = llama_embed::find_model_files(&dir)?;
        let threads = if s.threads > 0 {
            s.threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        };
        let use_gpu = !s.device.eq_ignore_ascii_case("cpu-only");
        let t = std::time::Instant::now();
        let eng = llama_embed::Engine::load(&text, mmproj.as_deref(), dim, threads, use_gpu)?;
        let audio = eng.has_audio;
        tracing::info!(
            embedder = "gemma2-native",
            dim,
            vision = eng.has_vision,
            audio,
            load_s = format!("{:.1}", t.elapsed().as_secs_f64()),
            "embedder ready (in-process llama.cpp)"
        );
        Ok(Self {
            inner: Inner::Native(eng),
            name: format!("gemma2-native:{dim}"),
            dim,
            audio,
        })
    }

    #[cfg(not(feature = "gemma-native"))]
    fn load_native(_s: &ModelSettings) -> Result<Self> {
        anyhow::bail!(
            "this build has no in-process EmbeddingGemma 2: rebuild with `--features gemma-native`, or use `--embedder gemma2` with the sidecar"
        )
    }

    /// EmbeddingGemma 2 through the sidecar: `GET /health` reports the model, dimension and modalities.
    fn load_remote(url: &str) -> Result<Self> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(300)))
            .build()
            .into();
        let url = url.trim_end_matches('/').to_string();
        let mut res = agent.get(format!("{url}/health")).call().with_context(|| {
            format!("embedding sidecar not reachable at {url} (start sidecar/embed_server.py)")
        })?;
        let h: serde_json::Value = res.body_mut().read_json().context("sidecar /health")?;
        let dim = h["dim"].as_u64().unwrap_or(0) as usize;
        anyhow::ensure!(dim > 0, "sidecar reported no dimension: {h}");
        let model = h["model"].as_str().unwrap_or("gemma2").to_string();
        let mods: Vec<String> = h["modalities"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let audio = mods.iter().any(|m| m == "audio");
        let name = format!(
            "gemma2:{}:{dim}",
            model.rsplit('/').next().unwrap_or("embeddinggemma-2")
        );
        tracing::info!(embedder = %name, dim, ?mods, "embedder ready (sidecar)");
        Ok(Self {
            inner: Inner::Remote { url, agent },
            name,
            dim,
            audio,
        })
    }

    /// Frees the model now (its weights and any GPU buffers); every later embed call fails. A
    /// process that loaded EmbeddingGemma natively must call this before it exits: llama.cpp's
    /// Metal device is a C++ static, and its destructor aborts if model buffers are still alive.
    pub fn release(&mut self) {
        self.inner = Inner::Released;
        self.audio = false;
    }

    fn remote(&self, items: &[Item]) -> Result<Vec<Vec<f32>>> {
        let Inner::Remote { url, agent } = &self.inner else {
            anyhow::bail!("not a remote embedder")
        };
        let mut res = agent
            .post(format!("{url}/embed"))
            .send_json(serde_json::json!({ "items": items }))
            .context("sidecar /embed")?;
        let v: serde_json::Value = res
            .body_mut()
            .read_json()
            .context("sidecar /embed response")?;
        if let Some(e) = v["error"].as_str() {
            anyhow::bail!("sidecar: {e}");
        }
        let vecs = v["vectors"]
            .as_array()
            .context("sidecar response has no vectors")?
            .iter()
            .map(|row| {
                row.as_array()
                    .map(|a| {
                        a.iter()
                            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                            .collect::<Vec<f32>>()
                    })
                    .unwrap_or_default()
            })
            .map(normalized)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            vecs.len() == items.len(),
            "sidecar returned {} vectors for {} items",
            vecs.len(),
            items.len()
        );
        Ok(vecs)
    }

    /// Embeds a search query. CLIP-style models get the "a photo of …" rewrite; EmbeddingGemma 2 gets
    /// its own retrieval prompt inside the sidecar. Returns the text that was actually embedded.
    pub fn embed_query(&mut self, q: &str) -> Result<(String, Vec<f32>)> {
        match &mut self.inner {
            Inner::Onnx(_) => {
                let r = crate::search::rewrite(q);
                let v = self.embed_text(&r)?;
                Ok((r, v))
            }
            Inner::Remote { .. } => {
                let v = self.remote(&[Item {
                    kind: "query",
                    text: Some(q),
                    title: None,
                    b64: None,
                }])?;
                Ok((q.to_string(), v.into_iter().next().unwrap_or_default()))
            }
            #[cfg(feature = "gemma-native")]
            Inner::Native(eng) => {
                // EmbeddingGemma's retrieval prompt for queries; documents use `title: … | text: …`.
                let prompt = format!("task: search result | query: {q}");
                let v = eng.embed(llama_embed::Input::Text(&prompt))?;
                Ok((prompt, v))
            }
            Inner::Released => anyhow::bail!("the embedder was released"),
        }
    }

    /// Embeds raw text (no rewriting).
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        match &mut self.inner {
            Inner::Onnx(rt) => {
                let y = rt.encode_texts(&[text])?.embedding;
                let n = y.norm_l2_keepdim(-1)?;
                let y = y / n;
                Ok(y.iter().cloned().collect())
            }
            Inner::Remote { .. } => Ok(self
                .remote(&[Item {
                    kind: "query",
                    text: Some(text),
                    title: None,
                    b64: None,
                }])?
                .into_iter()
                .next()
                .unwrap_or_default()),
            #[cfg(feature = "gemma-native")]
            Inner::Native(eng) => {
                let prompt = format!("task: search result | query: {text}");
                eng.embed(llama_embed::Input::Text(&prompt))
            }
            Inner::Released => anyhow::bail!("the embedder was released"),
        }
    }

    /// One L2-normalized vector per image.
    pub fn embed_images(&mut self, imgs: &[RgbImage]) -> Result<Vec<Vec<f32>>> {
        match &mut self.inner {
            Inner::Onnx(rt) => {
                let mut out = Vec::with_capacity(imgs.len());
                for img in imgs {
                    let img = Image::from(img.clone());
                    let y = rt.encode_images(std::slice::from_ref(&img))?.embedding;
                    let n = y.norm_l2_keepdim(-1)?;
                    let y = y / n;
                    out.push(y.iter().cloned().collect());
                }
                Ok(out)
            }
            Inner::Remote { .. } => {
                let items = imgs
                    .iter()
                    .map(|i| {
                        Ok(Item {
                            kind: "image",
                            text: None,
                            title: None,
                            b64: Some(vec![jpeg_b64(i)?]),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.remote(&items)
            }
            #[cfg(feature = "gemma-native")]
            Inner::Native(eng) => imgs
                .iter()
                .map(|i| {
                    eng.embed(llama_embed::Input::Images(vec![(
                        i.width(),
                        i.height(),
                        i.as_raw().as_slice(),
                    )]))
                })
                .collect(),
            Inner::Released => anyhow::bail!("the embedder was released"),
        }
    }

    /// One vector for a set of views of the same object (a tracklet's best crops). CLIP averages the
    /// per-image vectors; EmbeddingGemma 2 embeds the images as one interleaved input.
    pub fn embed_image_set(&mut self, imgs: &[RgbImage]) -> Result<Vec<f32>> {
        match &mut self.inner {
            Inner::Onnx(_) => Ok(mean_normalized(&self.embed_images(imgs)?)),
            Inner::Remote { .. } => {
                let b64 = imgs.iter().map(jpeg_b64).collect::<Result<Vec<_>>>()?;
                Ok(self
                    .remote(&[Item {
                        kind: "images",
                        text: None,
                        title: None,
                        b64: Some(b64),
                    }])?
                    .into_iter()
                    .next()
                    .unwrap_or_default())
            }
            #[cfg(feature = "gemma-native")]
            Inner::Native(eng) => {
                let views: Vec<(u32, u32, &[u8])> = imgs
                    .iter()
                    .map(|i| (i.width(), i.height(), i.as_raw().as_slice()))
                    .collect();
                eng.embed(llama_embed::Input::Images(views))
            }
            Inner::Released => anyhow::bail!("the embedder was released"),
        }
    }

    /// Embeds a 16 kHz mono WAV (EmbeddingGemma 2 only).
    pub fn embed_audio_wav(&mut self, wav: &[u8]) -> Result<Vec<f32>> {
        anyhow::ensure!(self.audio, "embedder '{}' cannot embed audio", self.name);
        match &mut self.inner {
            #[cfg(feature = "gemma-native")]
            Inner::Native(eng) => {
                let (rate, pcm) = crate::audio::wav_to_f32(wav)?;
                eng.embed(llama_embed::Input::Audio {
                    samples: &pcm,
                    sample_rate: rate,
                })
            }
            _ => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(wav);
                Ok(self
                    .remote(&[Item {
                        kind: "audio",
                        text: None,
                        title: None,
                        b64: Some(vec![b64]),
                    }])?
                    .into_iter()
                    .next()
                    .unwrap_or_default())
            }
        }
    }
}

/// SigLIP 2 files live on Hugging Face (onnx-community). If a models_dir holds a mirrored copy, use it.
fn siglip_local(cfg: Config, models_dir: &Option<PathBuf>, sub: &str) -> Config {
    if let Some(d) = models_dir {
        let base = d.join(sub);
        if base.join("vision_model.onnx").exists() {
            return cfg
                .with_visual_file(base.join("vision_model.onnx").to_string_lossy())
                .with_textual_file(base.join("text_model.onnx").to_string_lossy())
                .with_tokenizer_file(base.join("tokenizer.json").to_string_lossy())
                .with_tokenizer_config_file(base.join("tokenizer_config.json").to_string_lossy())
                .with_special_tokens_map_file(
                    base.join("special_tokens_map.json").to_string_lossy(),
                );
        }
    }
    cfg
}

pub fn mean_normalized(vs: &[Vec<f32>]) -> Vec<f32> {
    if vs.is_empty() {
        return vec![];
    }
    let d = vs[0].len();
    let mut m = vec![0f32; d];
    for v in vs {
        for i in 0..d {
            m[i] += v[i];
        }
    }
    let n = (m.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-9);
    m.iter().map(|x| x / n).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_embedder_fails_cleanly() {
        let mut e = Embedder {
            inner: Inner::Remote {
                url: "http://127.0.0.1:9".into(),
                agent: ureq::Agent::new_with_defaults(),
            },
            name: "test".into(),
            dim: 4,
            audio: true,
        };
        e.release();
        assert!(!e.audio);
        let err = e.embed_text("a person").unwrap_err().to_string();
        assert!(err.contains("released"), "{err}");
        assert!(e.embed_query("a person").is_err());
        assert!(e.embed_images(&[]).is_err());
        assert!(e.embed_image_set(&[]).is_err());
        assert!(e.embed_audio_wav(&[]).is_err());
    }

    #[test]
    fn resolve_prefers_models_dir_file() {
        let t = tempfile::tempdir().unwrap();
        let d = Some(t.path().to_path_buf());
        let remote = format!("{GH}/deim/dfine-s-coco.onnx");
        // Missing local file: the remote spec passes through to usls.
        assert_eq!(resolve(&d, "deim/dfine-s-coco.onnx", &remote), remote);
        std::fs::create_dir_all(t.path().join("deim")).unwrap();
        std::fs::write(t.path().join("deim/dfine-s-coco.onnx"), b"x").unwrap();
        assert_eq!(
            resolve(&d, "deim/dfine-s-coco.onnx", &remote),
            t.path().join("deim/dfine-s-coco.onnx").to_string_lossy()
        );
        assert_eq!(resolve(&None, "deim/dfine-s-coco.onnx", &remote), remote);
    }

    #[test]
    fn siglip_local_uses_models_dir_once_vision_exists() {
        let t = tempfile::tempdir().unwrap();
        let d = Some(t.path().to_path_buf());
        let base = t.path().join("siglip2-b16-256");
        let vision = base
            .join("vision_model.onnx")
            .to_string_lossy()
            .into_owned();
        let cfg = format!(
            "{:?}",
            siglip_local(Config::siglip2_b16_256(), &d, "siglip2-b16-256")
        );
        assert!(!cfg.contains(&vision), "no local copy yet");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("vision_model.onnx"), b"x").unwrap();
        let cfg = format!(
            "{:?}",
            siglip_local(Config::siglip2_b16_256(), &d, "siglip2-b16-256")
        );
        assert!(cfg.contains(&vision), "{cfg}");
        assert!(cfg.contains(&*base.join("text_model.onnx").to_string_lossy()));
    }

    #[cfg(feature = "gemma-native")]
    #[test]
    fn load_native_reads_models_dir() {
        let t = tempfile::tempdir().unwrap();
        let s = ModelSettings {
            models_dir: Some(t.path().to_path_buf()),
            embedder: "gemma2-native:256".into(),
            ..ModelSettings::default()
        };
        let err = Embedder::load(&s).err().expect("no gguf files");
        let want = t.path().join("embeddinggemma-2");
        assert!(
            format!("{err:#}").contains(&*want.to_string_lossy()),
            "{err:#}"
        );
    }
}
