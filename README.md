<p align="center">
  <img src="docs/screenshots/search-red-jacket.jpg" alt="Retrace answers 'a person in a red jacket' with ranked clips" width="100%">
</p>

<h1 align="center">Retrace</h1>

<p align="center"><strong>Ask your cameras what happened. Then retrace their steps.</strong><br>
A video investigator that runs on one computer. No data leaves the machine.</p>

<p align="center">
  <a href="https://github.com/humancto/retrace/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/humancto/retrace/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-3fc1b9.svg"></a>
  <img alt="Rust 1.88+" src="https://img.shields.io/badge/rust-1.88%2B-e29a45.svg">
  <img alt="macOS · Linux" src="https://img.shields.io/badge/platform-macOS%20%C2%B7%20Linux-95a3b3.svg">
  <img alt="No cloud" src="https://img.shields.io/badge/cloud%20calls-0-e06b69.svg">
</p>

---

Retrace reads video from a file, an RTSP camera, an HLS stream, or a YouTube Live stream. It
makes one record for each person or vehicle for each visit, and calls it a **tracklet**. It
embeds the best crops of each tracklet with an open image model and stores them in one SQLite
file. Then you ask questions in plain language:

> _"a person in a red jacket"_ · _"two people walking together"_ · _"a white van"_ · _"a person who stays near the door for 45 s"_

The answer is a ranked list of clips that play in the browser. Retrace draws the path of the
person on the frame and finds other visits of the same person. A **rule** is a question that
Retrace asks again for each new tracklet, so it can send an alert while the person is still in
the frame. Everything runs on your computer with open weights.

## Desktop app

Retrace Desktop is the same engine in a native macOS window. You do not need a terminal,
Homebrew, Rust or Python.

1. Download `Retrace_0.5.0_aarch64.dmg` from the
   [latest GitHub release](https://github.com/humancto/retrace/releases/latest). The build is for
   Apple Silicon and macOS 14 or later.
2. Open the DMG and drag Retrace to Applications.
3. Open Retrace. The build has no Apple signature, so macOS blocks it the first time. On macOS 14,
   right-click the app and select **Open**. On macOS 15 and later, open System Settings > Privacy &
   Security and click **Open Anyway**. Alternative:
   `xattr -dr com.apple.quarantine /Applications/Retrace.app`.

On the first run, the app downloads ffmpeg, ffprobe, yt-dlp, the detector and the weights of the
selected embedder. A window shows the progress of each file. With the default embedder, SigLIP 2,
the download is about 1.7 GB (1,671,494,078 bytes in our test). Then the app shows the search UI.

To index a video, select **File > Open Video…** or drag the video into the window. The window
title shows the indexing progress.

The app keeps all its data in `~/Library/Application Support/com.humancto.retrace/`: the tools,
the weights, one index for each embedder, the logs and `settings.json`. It does not use the
`usls` cache of the CLI.

The app contains both embedders:

- `siglip2-b16-256` (the default).
- `gemma2-native`, EmbeddingGemma 2 in the process. To use it, set `"embedder": "gemma2-native"`
  in `settings.json` and start the app again. The app then downloads the GGUF files (865 MB).

Read [docs/DESKTOP.md](docs/DESKTOP.md) for the architecture, the build and the signing steps.

## Quick start

You need macOS (Apple Silicon) or Linux, Rust 1.88 or later, and ffmpeg.

```bash
brew install ffmpeg rust
git clone https://github.com/humancto/retrace && cd retrace
cargo build --release                         # the first build takes some minutes
scripts/e2e.sh                                # sample video → index → search → clip → eval
```

Index your video and open the UI:

```bash
./target/release/retrace --embedder siglip2-b16-256 ingest --source ~/Movies/driveway.mp4 --cam driveway
./target/release/retrace --embedder siglip2-b16-256 serve
open http://127.0.0.1:8080/
```

The first run downloads the models to the local cache. After that, Retrace does not use the
network.

## What runs on your machine

```
retrace                      one Rust binary, 49 MB, one process
├─ ffmpeg                    one child process for each source: decodes to 640 px frames at 4 fps
├─ ffmpeg                    one more child for each live source: records the stream (video copied, audio to AAC)
├─ DEIM-D-FINE S             detector, ONNX, 40 MB on disk     cache: ~/Library/Caches/usls (macOS), ~/.cache/usls (Linux)
├─ SigLIP 2 B/16             image tower + text tower, ONNX, 1.4 GB on disk   same cache
├─ tracker                   src/track.rs, in the process
├─ data/index.db             SQLite: tracklets, vectors (sqlite-vec), captions (FTS5), rules, alerts, events
├─ data/crops/<cam>/         JPEG crops, 3 for each tracklet, plus one context frame
├─ data/rec/<cam>/           MP4 segments of 10 s from live sources, kept for 48 h
└─ 127.0.0.1:8080            the HTTP server: the JSON API and the UI (ui/index.html, one file)

optional
├─ EmbeddingGemma 2          a Python sidecar on 127.0.0.1:8082, or in the process with --features gemma-native
└─ a VLM server              llama-server or Ollama on 127.0.0.1:8081, for captions
```

The pipeline, from frame to answer:

```
frame ─▶ detector ─▶ tracker ─▶ tracklet (path, dwell, entry and exit edge, direction, best crops)
                                   └─▶ image tower ─▶ one vector ─▶ sqlite-vec     ◀── rules run here
question ─▶ rewrite ─▶ text tower ─▶ filtered exact cosine search ─▶ BM25 on captions ─▶ rank fusion ─▶ threshold ─▶ clips
```

- **Retrace sets the threshold for each query**: max(top − 0.06, μ + σ), with at least 5 results.
  On the sample video, "a bicycle" returns 1 result and "a person in a red jacket" returns 5 of 30 candidates.
- **The tracker samples at 4 fps on purpose.** Between two frames, a person who walks moves
  almost one box width. The tracker predicts with constant velocity and confirms a track after 2
  hits. The author measured 35 visits on real video where ByteTrack's defaults made 136 fragments.
- **Rules run while the track is open.** Retrace writes a track with a dwell of 8 s or more as
  provisional and checks it again after each 10 s. A rule "dwell 45 s or more" fires at 45 s.

## Measured results

From `scripts/e2e.sh` and `retrace eval` on OpenCV's public sample `vtest.avi` (80 s, a
courtyard, about 10 people), on 2026-10-13.

| | Apple M4 Max (CPU), measured 2026-10-13 | Linux container, 2 cores, reported by the author |
| -------------------- | --------------------------------------------------- | ------------------------------------------------ |
| Analysis             | 318 frames at 4 fps, 35 tracklets, 3,121 detections | the same tracklets                               |
| Ingest time          | 33 s for 79 s of video                              | slower than real time; backpressure skips frames |
| Search, end to end   | p50 14 ms, p95 15 ms                                | 50 to 65 ms                                      |
| Clip cut             | less than 1 s                                       | 0.8 to 1 s                                       |
| Eval, 7 queries, k=5 | recall@5 0.95, mAP 0.95                             | —                                                |

```
$ retrace --embedder siglip2-b16-256 eval --golden eval/golden.vtest.json --k 5
a person in a red jacket                           recall@5=1.00 AP=1.00 top=[21, 30, 6, 26, 31]
a white van                                        recall@5=0.67 AP=0.67 top=[5, 16]
two people walking together                        recall@5=1.00 AP=1.00 top=[7, 31, 28, 26, 23]
a bicycle                                          recall@5=1.00 AP=1.00 top=[35]
person in a white hooded jacket                    recall@5=1.00 AP=1.00 top=[11, 6, 21, 31, 14]
a man in a blue jacket                             recall@5=1.00 AP=1.00 top=[31, 10, 24, 28, 6]
person in dark clothes                             recall@5=1.00 AP=0.96 top=[17, 25, 3, 18, 8]

queries=7 recall@5=0.952 mAP=0.946 p50=14.3ms p95=15.3ms embedder=siglip2-b16-256
```

We checked each hit in [`eval/golden.vtest.json`](eval/golden.vtest.json) against the crop by
eye. The one miss is a white van that the detector classed as `car`; the query infers `truck`
and hides it. Tracklet ids are the same for each run on one machine. A different ffmpeg or
onnxruntime build can detect one track more or less, which moves the ids; `scripts/e2e.sh` then
prints the eval without the recall check.

## Screens

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/tracklet-detail.jpg" alt="The detail panel: clip playback with the path drawn on the frame"><br><sub><b>Open a result.</b> The clip starts 4 s before the visit. The green point is the entry, the red point is the exit. Pin it to the case, find more like it, download the clip, or copy a link.</sub></td>
    <td width="50%"><img src="docs/screenshots/timeline-brush.jpg" alt="A time range selected on the timeline"><br><sub><b>Timeline.</b> Each camera has a lane with the tracklet density. Drag on a lane to limit the search to that time. The hits are marked in white.</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/more-like-this.jpg" alt="More like this"><br><sub><b>More like this.</b> Other visits of the same person, by crop similarity, also when the tracker split a visit.</sub></td>
    <td><img src="docs/screenshots/rules-backtest.jpg" alt="A rule with its backtest, and the case panel"><br><sub><b>Rules and the case.</b> "a person in a red jacket, score ≥ 0.13" has 3 matches in the indexed video. The case panel keeps the results that you pin and exports them as JSON.</sub></td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/screenshots/live-view.jpg" alt="An analysed frame from a YouTube Live street camera, with boxes, ids, dwell and trails" width="100%"><br><sub><b>Live view.</b> An analysed frame from the EarthCam New Orleans street camera on YouTube Live, 2026-10-14 00:22 local time: 2 confirmed persons with their dwell and trails, 1.3 fps, every third frame analysed because the machine was under load.</sub></td>
  </tr>
</table>

## The search console

The UI is one HTML file with no build step. It has:

- A command bar with the recent queries. The query and the filters are in the URL, so a search is
  a link that you can send.
- A timeline with the tracklet density for each camera. Drag on a lane to select a time range.
  Click a bar to open the tracklet at that time.
- Result cards that show the context frame, the box, and the best crop. Move the pointer over a
  card to see its 3 crops.
- Keyboard control: `/` focuses the search, `j` and `k` move the selection, `Enter` opens the
  result, `p` pins it, `Esc` closes the panel, `?` shows the list.
- A detail panel with the clip, the path on the frame, the crops, "more like this", a clip
  download, and a link to copy.
- A case panel. Pin results to it, then export the case as JSON with the clip URLs. The case
  stays in your browser.
- Audio results with a player, when the embedder supports audio.
- The live view, the live feed, and the rules with their backtest.

## Live cameras

A source is each input that ffmpeg can open: `rtsp://`, `http(s)://` HLS or MJPEG, `srt://`,
and YouTube URLs (`yt-dlp` resolves them; Retrace downloads a YouTube video one time and plays it as
a file).

```bash
./target/release/retrace --embedder siglip2-b16-256 serve \
  --source driveway=rtsp://user:pass@192.168.1.40:554/h264Preview_01_sub \
  --source street="https://www.youtube.com/watch?v=<live video id>"
```

Caution: Retrace stores a URL with a password in the database, and the process list shows it. Read
[SECURITY.md](SECURITY.md).

With a live source, Retrace shows the analysed frames in the UI (`/live.mjpeg?cam=`), records
the stream into 10 s segments for clip playback, and analyses each n-th frame when detection is
slower than the stream, so the analysis stays at real time. When a stream sends no frame for 30 s,
Retrace stops ffmpeg (SIGTERM, then SIGKILL) and restarts the source. For a YouTube URL it
resolves the media URL again. Ctrl-C stops everything.

**YouTube Live.** On 2026-10-14 we pointed Retrace at the EarthCam New Orleans street camera. The
stream delivers about 30 s of segments, then YouTube answers HTTP 403 to the next segment request.
The cause is on the YouTube side: yt-dlp reports no proof-of-origin token provider on this
machine, and YouTube limits clients without one. Retrace detects the stall and restarts the
source; the cycle is about 30 s of video for each 60 s. In the first 25 s of one cycle Retrace made
23 person tracklets, "a person in a white shirt" returned 3 people in white shirts first, and "a
person in a red shirt" returned a person in a red top first (we checked the crops by eye). RTSP
cameras, HLS from your own devices, SRT and files do not have this limit. A yt-dlp proof-of-origin
token plugin removes it for YouTube.

## Embedders

| `--embedder`          | Model                                       | Notes                                                                         |
| --------------------- | ------------------------------------------- | ----------------------------------------------------------------------------- |
| `siglip2-b16-256`     | SigLIP 2 B/16, 256 px, ONNX                 | recommended; `siglip2-b16-384` and `siglip2-so400m-384` are slower and better |
| `clip-b16`            | CLIP ViT-B/16, ONNX                         | the default; English only in practice                                         |
| `gemma2`              | EmbeddingGemma 2 in a Python sidecar        | text, images, and audio in one space; queries in more than 100 languages      |
| `gemma2-native[:dim]` | EmbeddingGemma 2 in the process (llama.cpp) | no Python, no open port; `cargo build --features gemma-native`                |

An index belongs to one embedder. Use a different `--db` for each embedder.

With EmbeddingGemma 2, Retrace also embeds 10 s audio windows every 5 s, so "a dog barking" or
"glass breaking" return audio results with a player. Setup:

```bash
pip install -U "sentence-transformers>=6.1" "transformers>=5" torch pillow soundfile
python3 sidecar/embed_server.py --dim 256 &               # downloads the model one time (~1.5 GB)
./target/release/retrace --embedder gemma2 --db data/gemma.db serve --source door=rtsp://...
```

For the in-process path: `scripts/get-gemma.sh` downloads the GGUF files (865 MB), then
`cargo build --release --features gemma-native` (needs cmake, git, a C++ toolchain, and libclang
for bindgen). The build compiles llama.cpp with Metal on macOS.

Measured on 2026-10-14 (Apple M4 Max): the in-process path indexed `vtest.avi` in 2 min 15 s,
model load included (the binary is 57 MB). After the first query, a search takes 9 to 21 ms.
"a person in a red jacket", "una persona con chaqueta roja" (Spanish) and
"ఎర్ర జాకెట్ వేసుకున్న వ్యక్తి" (Telugu) all return the two people in red jackets in the first
3 results, with no translation step. "a bicycle" returns the one bicycle. Audio windows need a
source with sound; the sample video has none.

Captions from a local VLM are optional. Start `llama-server` with a vision model, then run
`retrace annotate --endpoint http://127.0.0.1:8081/v1`. The captions join the search through BM25.

## Options

| Flag                                    | Default                        | Notes                                                                                            |
| --------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------ |
| `--detector`                            | `deim-dfine-s`                 | `deim-dfine-n` for small machines, `deim-dfine-m` for quality, or a D-FINE/DEIM ONNX path or URL |
| `--device`                              | `cpu`                          | `coreml` with `--features coreml` (experimental); `cpu-only` keeps EmbeddingGemma off the GPU    |
| `--models-dir`                          | –                              | a directory with downloaded models, for computers with no network; env `RETRACE_MODELS`          |
| `--db` / `--crops`                      | `data/index.db` / `data/crops` | the SQLite file and the JPEG crops                                                               |
| `--conf`                                | 0.35                           | the detection confidence; detections down to half of it only extend tracks                       |
| `ingest --fps` / `--width`              | 4 / 640                        | the analysis rate and width                                                                      |
| `ingest --start`                        | now − duration                 | the wall-clock start of a file (epoch ms or RFC 3339)                                            |
| `--record` / `--rec` / `--retain-hours` | true / `data/rec` / 48         | the recorder for live sources                                                                    |
| `--embed-url` / `--audio`               | `127.0.0.1:8082` / true        | the sidecar address; audio windows on or off                                                     |
| `serve --bind`                          | `127.0.0.1:8080`               | no authentication: read [SECURITY.md](SECURITY.md) before you bind to another address            |

Model files for `--models-dir`:

```
DIR/deim/dfine-s-coco.onnx
DIR/clip/{vit-b16-visual.onnx, vit-b16-textual.onnx, tokenizer.json, tokenizer_config.json, special_tokens_map.json}
DIR/siglip2-b16-256/{vision_model.onnx, text_model.onnx, tokenizer.json, tokenizer_config.json, special_tokens_map.json}
DIR/embeddinggemma-2/{embeddinggemma-2-Q8_0.gguf, mmproj-embeddinggemma-2-Q8_0.gguf}
```

If `cdn.pyke.io` is not reachable at build time, run
`cargo build --no-default-features --features dynamic-ort` and set `ORT_DYLIB_PATH` to a
`libonnxruntime` file.

## API

Each function of the UI is a JSON call:

- `GET /api/search?q=&cam=&class=&direction=&entry=&exit=&min_dwell_s=&t0=&t1=&k=`
- `GET /api/tracklets?cam=&t0=&t1=&limit=` · `GET /api/tracklets/{id}` · `GET /api/similar/{id}`
- `GET /clip?cam=&t0=&t1=&pre=4000` (MP4) · `GET /thumb/{id}?k=0` · `GET /ctx/{id}`
- `GET /api/live` (SSE) · `GET /live.mjpeg?cam=` · `GET /frame.jpg?cam=` · `GET /api/live-cams` · `GET /api/stats`
- `POST /api/rules {text, min_score, class, cam, min_dwell_s}` · `GET /api/rules` · `DELETE /api/rules/{id}` · `GET /api/alerts?since=`
- `POST /api/events {cam, ts, source, type, attrs}` · `GET /api/events?t0=&t1=`

## Network use

Retrace uses the network only for the model downloads (one time), the onnxruntime binary at build
time (`cdn.pyke.io`), `yt-dlp` for YouTube URLs, and the sources that you give. Retrace has no
telemetry, no update check, and no account.

## Known limits

- The tracker can swap two people who cross near each other. "More like this" connects the
  fragments.
- Attribute queries are coarse on small crops. VLM captions correct this.
- The first 15 s after a live stream starts can be missing from the recording.
- Colour queries fail on infrared video with no error.

## Roadmap

A query language for zones and sequences compiled by a local LLM, appearance re-identification
across cameras, rule actions, signed evidence packages, detections from edge devices over MQTT,
and an eval dashboard. Issues with the label `good first issue` are the entry point.

## Contributing and license

Read [CONTRIBUTING.md](CONTRIBUTING.md): one change for each pull request, run `scripts/e2e.sh`
before you push, add the `retrace eval` line for search and tracker changes.

MIT, see [LICENSE](LICENSE). The model weights have their own licenses: DEIM / D-FINE
Apache-2.0, CLIP MIT, SigLIP 2 Apache-2.0, EmbeddingGemma 2 Apache-2.0. The sample video is
OpenCV's `vtest.avi` (Apache-2.0). This document follows ASD-STE100 Simplified Technical English.
