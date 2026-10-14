<p align="center">
  <img src="docs/screenshots/search-red-jacket.jpg" alt="Retrace answering 'a person in a red jacket' with ranked, playable clips" width="100%">
</p>

<h1 align="center">Retrace</h1>

<p align="center"><strong>Ask your cameras what happened. Then retrace their steps.</strong><br>
A fully local video investigator. Nothing leaves the machine.</p>

<p align="center">
  <a href="https://github.com/humancto/retrace/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/humancto/retrace/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3fc1b9.svg"></a>
  <img alt="Rust 1.88+" src="https://img.shields.io/badge/rust-1.88%2B-e29a45.svg">
  <img alt="macOS · Linux" src="https://img.shields.io/badge/platform-macOS%20%C2%B7%20Linux-95a3b3.svg">
  <img alt="No cloud" src="https://img.shields.io/badge/cloud%20calls-0-e06b69.svg">
</p>

---

Point Retrace at a video file, an RTSP camera, an HLS stream or a YouTube Live feed. It turns the
footage into **tracklets** (one record per person or vehicle per visit), embeds their best crops with
an open CLIP-style model, indexes them in a single SQLite file, and answers questions like

> _"a person in a red jacket"_ · _"two people walking together"_ · _"a white van"_ · _"someone loitering by the door for 45 s"_

with ranked, playable clips, a drawn trajectory, and "more like this". Rules are standing queries
that fire on new tracklets while the person is still in frame. An optional local VLM adds captions.
Pull the network cable and it still works.

It is the same hybrid design the big cloud camera vendors run (edge crops → vector index →
filter-then-exact-search → per-query threshold), collapsed onto one laptop, with open weights.

## Sixty seconds to a first answer

```bash
brew install ffmpeg rust                      # rustup is fine too; Rust 1.88+
git clone https://github.com/humancto/retrace && cd retrace
cargo build --release                         # first build takes a few minutes
scripts/e2e.sh                                # public sample video → index → search → clip → eval
```

Then index your own footage and open the UI:

```bash
./target/release/retrace --embedder siglip2-b16-256 ingest --source ~/Movies/driveway.mp4 --cam driveway
./target/release/retrace --embedder siglip2-b16-256 serve
open http://127.0.0.1:8080/
```

Models download once on first run (DEIM from GitHub releases, SigLIP 2 from Hugging Face) into
the local cache; after that, no network is needed. Use `--models-dir` for an air-gapped machine.

## What it looks like

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/tracklet-detail.jpg" alt="Clip playback with the trajectory drawn over the context frame"><br><sub><b>Open a hit.</b> The clip plays from four seconds before the visit; the path is drawn over the frame, green dot in, red dot out.</sub></td>
    <td width="50%"><img src="docs/screenshots/more-like-this.jpg" alt="More like this: the same person's other visits by crop similarity"><br><sub><b>More like this.</b> Crop similarity links the same person's other visits, even when the tracker split them.</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/rules-backtest.jpg" alt="A standing rule with its backtest over indexed footage"><br><sub><b>Rules with a backtest.</b> "person in a red jacket, score ≥ 0.13" would have fired three times on this footage. New matches raise an alert in the live feed.</sub></td>
    <td><img src="docs/screenshots/search-red-jacket.jpg" alt="Search with filters, timeline lanes and result cards"><br><sub><b>Search.</b> Filters for camera, class, direction, entry edge, dwell and time; timeline lanes per camera; the Plan drawer shows the rewritten query and the threshold.</sub></td>
  </tr>
</table>

Screenshots are from OpenCV's public `vtest.avi` sample (80 s, outdoor courtyard, about ten people).

## Measured, not promised

Every number below comes from `scripts/e2e.sh` and `retrace eval` on `vtest.avi`, 2026-10-13.

|                      | Apple M4 Max (CPU provider)                                    | 2-core Linux container                       |
| -------------------- | -------------------------------------------------------------- | -------------------------------------------- |
| Analysis             | 318 frames at 4 fps, **35 tracklets**, 3,121 detections        | same tracklets                               |
| Ingest wall time     | **33 s** for 79 s of video (2.4× real time)                    | slower than real time, backpressure keeps up |
| Search, end to end | **p50 14 ms · p95 16 ms** (`/api/search` on an idle machine; UI round trip 35 ms) | 50–65 ms |
| Clip cut             | under a second from file or live recording                     | 0.8–1 s                                      |
| Eval, 7 queries, k=5 | **recall@5 0.95 · mAP 0.95** | — |

The golden set is [`eval/golden.vtest.json`](eval/golden.vtest.json). Every hit was checked by eye
against the crop, and tracklet ids are reproducible run to run (tracks are persisted in id order), so
the set stays valid. It is deliberately small. The miss is "a white van": three white vans are in the
footage, one of them was detected as `car`, and the query's inferred class filter (`truck`) hides it.
Vehicle class votes on low-resolution crops are the weak spot; `--conf` and the caption path help.

```
$ retrace --embedder siglip2-b16-256 eval --golden eval/golden.vtest.json --k 5
a person in a red jacket                           recall@5=1.00 AP=1.00 top=[21, 30, 6, 26, 31]
a white van                                        recall@5=0.67 AP=0.67 top=[5, 16]
two people walking together                        recall@5=1.00 AP=1.00 top=[7, 31, 28, 26, 23]
a bicycle                                          recall@5=1.00 AP=1.00 top=[35]
person in a white hooded jacket                    recall@5=1.00 AP=1.00 top=[11, 6, 21, 31, 14]
a man in a blue jacket                             recall@5=1.00 AP=1.00 top=[31, 10, 24, 28, 6]
person in dark clothes                             recall@5=1.00 AP=0.96 top=[17, 25, 3, 18, 8]

queries=7 recall@5=0.952 mAP=0.946 p50=248.8ms p95=414.0ms embedder=siglip2-b16-256
```

The latency column of that run was captured on a machine under heavy load from other work; the idle figure in the table above (14 ms) is the one to expect.

## How it works

```
cameras / phones / files / YouTube Live ──ffmpeg──▶ frames (4 fps, 640 px)      ──ffmpeg -c copy──▶ data/rec/<cam>/<epoch>.mp4
   └▶ DEIM-D-FINE (ONNX) ─▶ sparse-fps tracker ─▶ tracklet close ─▶ best 3 crops + context frame
        └▶ CLIP / SigLIP 2 image tower ─▶ mean vector ─▶ sqlite-vec (vec0, cosine)   ◀── rules evaluate here
        └▶ (async) VLM caption (JSON schema) ─▶ FTS5                                   ◀── hybrid search (RRF)
        └▶ annotated frame (boxes, ids, trails) ─▶ /live.mjpeg
   └▶ (gemma2) audio 16 kHz ─▶ 10 s windows every 5 s ─▶ EmbeddingGemma 2 audio encoder ─▶ vec_audio   ◀── same query vector
axum API + UI: /api/search  /api/tracklets  /clip  /thumb  /ctx  /api/live (SSE)  /live.mjpeg  /api/rules  /api/events  /api/similar  /api/stats
```

**Query path.** Rewrite ("a photo of …", class inferred from the words) → text tower → filtered exact
cosine KNN → BM25 over captions → reciprocal-rank fusion → dynamic threshold
(max(top − 0.06, μ + σ), at least five) → clip by time range.

**Tracking at 2–8 fps.** Retrace samples video sparsely on purpose; detection at a few fps is the
cheap part of the recipe. Between two analysed frames a walking person moves most of a box width,
and ByteTrack's defaults (IoU association, one-frame confirmation) drop such tracks; on real footage
they produced 136 fragments where there were about 35 visits. [`src/track.rs`](src/track.rs) is a
small tracker made for this regime: constant-velocity prediction, an IoU-or-centre-distance gate,
ByteTrack-style second-stage matching with low-confidence detections, tentative tracks that live
three frames, confirmation after two hits. Unconfirmed detections are drawn thin in the live view.

**Live rules.** Tracks that dwell 8 s or more are written provisionally and re-evaluated every 10 s
of track time, so a "loitering for 45 s" rule fires at 45 s while the person is still in frame, once
per person; the tracklet is searchable before it closes.

## Live cameras

Anything ffmpeg can open is a source: `rtsp://` cameras, `http(s)://` HLS or MJPEG, `srt://`, and
YouTube URLs (live streams are resolved with `yt-dlp` and re-resolved on every reconnect; a YouTube
_video_ is downloaded once to `data/cache/` and replayed as a file).

```bash
# ingest while serving; repeat --source for each camera
./target/release/retrace --embedder siglip2-b16-256 serve \
  --source driveway=rtsp://user:pass@192.168.1.40:554/h264Preview_01_sub \
  --source phone1=rtsp://127.0.0.1:8554/phone1

# any public live stream, e.g. a street camera on YouTube Live (brew install yt-dlp)
./target/release/retrace --embedder siglip2-b16-256 serve \
  --source times_square="https://www.youtube.com/watch?v=<live video id>"
```

While a live source runs:

- **Live view** shows the analysed frames with boxes, track ids, dwell and trails, so you watch
  exactly what the index sees (`/live.mjpeg?cam=`, `/frame.jpg?cam=`, `/api/live-cams`).
- **Recorder**: the stream is copied without re-encoding into `data/rec/<cam>/<epoch>.mp4` as
  fragmented 10 s segments, kept for `--retain-hours` (48). Result clips are cut from these, so
  "play the clip" works for live cameras with no extra services. `--record false` turns it off.
- **Backpressure**: if detection cannot keep up with `--fps`, Retrace analyses every n-th frame
  instead of letting the pipe fill (the live view shows `1/n`); the stream never drifts behind real
  time. On an M-series Mac it is 1/1.
- Ctrl-C stops the ingest loops and the recorders.

[MediaMTX](https://github.com/bluenviron/mediamtx) remains optional as a hub for many cameras and
phones (`--mediamtx http://127.0.0.1:9996` makes clips redirect to its playback endpoint).

## EmbeddingGemma 2: one space for sight, sound and words

`--embedder gemma2` swaps CLIP for Google's **EmbeddingGemma 2** (Apache-2.0, 740M): text, images,
image sets and **audio** in one 768-d space (Matryoshka-truncated to 256-d by default, a 3× smaller
index at almost the same quality). It runs in a small local sidecar (PyTorch on Apple Silicon's
MPS, CUDA or CPU) that the Rust core talks to over localhost.

```bash
pip install -U "sentence-transformers>=6.1" "transformers>=5" torch pillow soundfile
python3 sidecar/embed_server.py --dim 256 --selftest      # downloads the model once (~1.5 GB)
python3 sidecar/embed_server.py --dim 256 &               # http://127.0.0.1:8082
./target/release/retrace --embedder gemma2 --db data/gemma.db serve --source door=rtsp://...
```

What changes with it:

- **Open-vocabulary audio search.** Sources with sound are also decoded to 16 kHz mono; every 5 s
  the last 10 s window is embedded and stored next to the tracklets. "a dog barking", "glass
  breaking", "someone shouting", "a car alarm" return audio windows in the same results, with a
  player cut from the recording. No fixed list of sound classes.
- **A tracklet is embedded as one input** (its best crops interleaved), not as the mean of per-crop
  vectors.
- **Queries in 100+ languages** without translation, and the model's retrieval prompts replace the
  "a photo of …" rewrite.
- Recordings and clips keep their audio (AAC) so audio hits play back.

### In-process, no Python: `--embedder gemma2-native`

llama.cpp's server cannot reach the model's image and audio encoders, which is why the sidecar
exists. The in-process path does what [DigUp](https://github.com/ARahim3/DigUp) does, in Rust:
[`crates/llama-embed`](crates/llama-embed) builds llama.cpp at a pinned tag (b11461) with the
`mtmd` multimodal library as static libraries (Metal on macOS, with the one-line "no logits"
patch), and feeds the image and audio encoder rows into the text model's batch between `<bos>`
and `<eos>`, mean-pooled, normalised, truncated. One binary, about 1 GB of RAM with the full model,
nothing listening on a port.

```bash
scripts/get-gemma.sh                                  # Q8_0 text + mmproj (~865 MB) → models/embeddinggemma-2/
cargo build --release --features gemma-native         # cmake + bindgen; first build compiles llama.cpp
./target/release/retrace --embedder gemma2-native:256 --db data/gemma.db serve --source door=rtsp://...
```

Needs cmake, git and a C++ toolchain (Xcode Command Line Tools on a Mac); bindgen needs libclang
(`export LIBCLANG_PATH=/Library/Developer/CommandLineTools/usr/lib` if it is not found).
`LLAMA_CPP_DIR` points the build at an existing llama.cpp checkout; `--device cpu-only` keeps the
model off the GPU. An index is tied to one embedder: use a separate `--db` per embedder.

## Captions with a local VLM (optional)

```bash
llama-server -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3-VL-8B-Instruct-F16.gguf -ngl 99 -c 8192 --port 8081 --jinja
./target/release/retrace annotate --endpoint http://127.0.0.1:8081/v1
```

`annotate` sends each tracklet's crops to any OpenAI-compatible endpoint (llama-server, mlx-vlm,
Ollama) with a grammar-enforced JSON schema; captions join search through BM25 and reciprocal-rank
fusion. This is the fix for attribute queries that zero-shot CLIP gets wrong on small crops.

## Options

| Flag                                    | Default                        | Notes                                                                                                                                                                  |
| --------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--embedder`                            | `clip-b16`                     | `siglip2-b16-256` (recommended on a Mac), `siglip2-b16-384`, `siglip2-so400m-384` (slower, better), `gemma2`, `gemma2-native[:dim]`. An index is tied to one embedder. |
| `--embed-url` / `--audio`               | `http://127.0.0.1:8082` / true | sidecar address for `gemma2`; audio windows on/off                                                                                                                     |
| `--detector`                            | `deim-dfine-s`                 | `deim-dfine-n` for small machines, `deim-dfine-m` for quality, or any D-FINE/DEIM ONNX path or URL                                                                     |
| `--device`                              | `cpu`                          | `coreml` with `--features coreml` (experimental for transformer detectors)                                                                                             |
| `--models-dir`                          | –                              | pre-downloaded models (layout below); env `RETRACE_MODELS`                                                                                                             |
| `--db` / `--crops`                      | `data/index.db` / `data/crops` | one SQLite file + JPEG crops                                                                                                                                           |
| `ingest --fps` / `--width`              | 4 / 640                        | analysis rate and width                                                                                                                                                |
| `ingest --start`                        | now − duration                 | wall-clock start of a file (epoch ms or RFC3339)                                                                                                                       |
| `--record` / `--rec` / `--retain-hours` | true / `data/rec` / 48         | built-in recorder for live sources                                                                                                                                     |
| `--conf`                                | 0.35                           | detection confidence; detections down to half of it extend existing tracks only                                                                                        |
| `serve --bind`                          | `127.0.0.1:8080`               | there is no auth; see [SECURITY.md](SECURITY.md) before binding wider                                                                                                  |

### Offline model directory

`--models-dir DIR` is checked first:

```
DIR/deim/dfine-s-coco.onnx
DIR/clip/{vit-b16-visual.onnx, vit-b16-textual.onnx, tokenizer.json, tokenizer_config.json, special_tokens_map.json}
DIR/siglip2-b16-256/{vision_model.onnx, text_model.onnx, tokenizer.json, tokenizer_config.json, special_tokens_map.json}
DIR/embeddinggemma-2/{embeddinggemma-2-Q8_0.gguf, mmproj-embeddinggemma-2-Q8_0.gguf}
```

DEIM and CLIP files come from `github.com/jamjamjon/assets/releases`; SigLIP 2 from
`huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX`; EmbeddingGemma 2 from
`huggingface.co/ggml-org/embeddinggemma-2-GGUF`.

### Where the onnxruntime CDN is unreachable

`cargo build --no-default-features --features dynamic-ort` and set `ORT_DYLIB_PATH` to a
`libonnxruntime` (for example the one inside `pip install onnxruntime>=1.28`).

## API

Everything the UI does is a JSON call you can script.

- `GET /api/search?q=&cam=&class=&direction=&entry=&exit=&min_dwell_s=&t0=&t1=&k=&mode=all`
- `GET /api/tracklets?cam=&t0=&t1=&limit=` · `GET /api/tracklets/{id}` · `GET /api/similar/{id}`
- `GET /clip?cam=&t0=&t1=&pre=4000` (MP4) · `GET /thumb/{id}?k=0` · `GET /ctx/{id}`
- `GET /api/live` (SSE: `tracklet`, `alert`) · `GET /api/stats`
- `GET /live.mjpeg?cam=` · `GET /frame.jpg?cam=` · `GET /api/live-cams`
- `POST /api/rules {text, min_score, class, cam, min_dwell_s}` → rule + backtest · `GET /api/rules` · `DELETE /api/rules/{id}` · `GET /api/alerts?since=`
- `POST /api/events {cam, ts, source, type, attrs}` · `GET /api/events?t0=&t1=`

## What touches the network, exactly

- Model weights, once, on first use of each model (GitHub releases, Hugging Face).
- The onnxruntime binary, at **build** time, from `cdn.pyke.io` (or none with `dynamic-ort`).
- `yt-dlp`, only when you give it a YouTube URL.
- The sources you point it at.

No telemetry, no update checks, no accounts. [SECURITY.md](SECURITY.md) has the threat model.

## Known limitations

- Tracks can still swap identities when two people cross closely (no appearance term in the
  tracker yet); "more like this" links fragments by crop similarity.
- Zero-shot CLIP attribute queries are coarse on low-resolution crops; captions from the VLM are
  the fix.
- Live clips come from the built-in recorder (or MediaMTX); the first ~15 s after a stream starts
  may not be recorded yet.
- YouTube media URLs expire after a few hours; the source is re-resolved automatically, with a
  short gap.
- Colour queries fail silently on infrared footage until the caption path marks frames monochrome.

## Roadmap

A spatial-temporal query DSL compiled by a local LLM (zones, sequences, "then / within"),
appearance re-ID and cross-camera journeys on a floor plan, rule actions (TTS talk-down), an
investigation agent with signed evidence packages, edge-device detections over MQTT (reCamera, Pi
AI Camera), a memory-mapped slab index as a second `VectorIndex` engine, and an eval dashboard.
Issues tagged `good first issue` are the way in.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). The short version: one change per PR, run
`scripts/e2e.sh` before you push, and paste the `retrace eval` line for anything that touches
search or tracking.

## License

MIT, see [LICENSE](LICENSE). Model weights carry their own licenses: DEIM / D-FINE Apache-2.0,
CLIP MIT, SigLIP 2 Apache-2.0, EmbeddingGemma 2 Apache-2.0. The sample footage in the screenshots
is OpenCV's `vtest.avi` (Apache-2.0).
