# Changelog

All notable changes to Retrace are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- A Windows installer.
- Developer ID signing and notarization in the release build (today a person does it; see
  `docs/DESKTOP.md`).
- Automatic updates for the desktop app.
- A live indicator in the menu bar.
- One shared embedder for ingest, audio and search, to use less memory.
- Bundle the licence notices of the compiled Rust crates (`cargo about`) in the app and show them
  in Help > Third-Party Software.
- Unique camera names for files with the same name in different folders (today `/a/clip.mp4` and
  `/b/clip.mp4` both index as camera `clip`).
- A marker file with the pinned sha256 for each installed tool, so the app checks more than the
  file size and the file type.
- Wait for an ingest that still runs its own EmbeddingGemma 2 embedder after the 3 s quit deadline,
  so its llama.cpp buffers are freed before exit.

## [0.5.0] — 2026-10-17

### Added
- **Retrace Desktop** (`desktop/src-tauri/`, Tauri v2): the engine and both embedders (SigLIP 2
  and EmbeddingGemma 2 in the process) in a native macOS window. Download
  `Retrace_0.5.0_aarch64.dmg` from the GitHub release. You do not need a terminal, Homebrew, Rust or
  Python.
- First-run setup with a progress window: the app downloads fixed versions of ffmpeg and ffprobe
  9.0.2 (martin-riedl.de GPL build), yt-dlp 2026.08.19, the detector and the weights of the
  selected embedder. Each file has a SHA-256 check, and a stopped download continues. With SigLIP 2,
  a cold first run downloads 9 files, 1,671,494,078 bytes. Setup checks that ffmpeg has libx264.
  **Try again** and **Use the installed tools** are on the setup page.
- **File > Open Video…**, drag and drop, Finder "Open With" and `--open <file>` add a video to a
  queue that indexes one file at a time. The window title shows the indexing progress.
- **File > Add Camera…** adds a live source. The stream check runs off the main thread, so the
  window does not stop. If the address is a video and not a live stream (for example a YouTube
  video that is not live), the app indexes it one time on the file queue. Two live cameras cannot
  have the same name. **Help** opens the data folder, the GitHub page and the third-party
  software list.
- All app data is in `~/Library/Application Support/com.humancto.retrace/`, with one index and one
  crops folder for each embedder. `RETRACE_APP_DATA`, `RETRACE_SETUP`, `RETRACE_MODELS_SEED`,
  `RETRACE_DOWNLOAD_DIR` and `RETRACE_LOG_STDERR` control unattended runs.
- A quit stops the ingest, the child processes and the server, then frees the embedder. The app
  exits with code 0 and leaves no ffmpeg or yt-dlp process.
- CI workflow `desktop.yml`: macOS arm64 (required) and Linux AppImage and deb (best effort). On a
  `v*` tag it attaches the bundles to the release.
- `docs/DESKTOP.md`: architecture, data layout, build, CI and signing.

### Changed
- The engine is now a library (`src/lib.rs`) and a thin CLI (`src/main.rs`). The CLI behaviour does
  not change: `scripts/e2e.sh` passes with recall@5 0.952 and mAP 0.946.
- A tools resolver (`src/tools.rs`) finds ffmpeg, ffprobe and yt-dlp in this order: the app tools
  folder, the folder of the executable, `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`. Each call to
  these tools goes through it. For the CLI, the new parts are the folder of the executable before
  `PATH` and the two Homebrew folders after it. In the desktop app, yt-dlp also gets
  `--ffmpeg-location`; the CLI does not add it.
- `serve` logs the address it bound, not the address string it was given.
- `api::bind` binds the port first and returns a handle that stops the server cleanly. `serve`
  uses it.
- The engine records the frame-pipe ffmpeg, the audio ffmpeg, yt-dlp and the clip ffmpeg as child
  processes, so a stop kills all of them.
- `Embedder::release` frees the model before exit. This prevents an abort in llama.cpp at exit
  with EmbeddingGemma 2.
- The model downloads for the flat `--models-dir` layout are in `src/models/fetch.rs`.
- With tracklets in the index and no query, the UI tells how many tracklets and cameras are
  indexed, not "No index yet". With an empty index, the desktop app tells you how to add a
  video (Cmd+O, or drop the file on the window).

### Known limitations
- The builds have an ad-hoc signature only. Apple does not notarize them. On macOS 14, right-click
  the app and select **Open**. On macOS 15 and later, use System Settings > Privacy & Security >
  **Open Anyway**.
- Today the only release artifact is for macOS on Apple Silicon (macOS 14 or later). The Linux
  build is best effort.
- The UI page still loads its fonts from Google Fonts.
- One indexed video uses much memory: peak RSS was about 6.6 GiB with SigLIP 2 and about 2.4 GiB
  with EmbeddingGemma 2 for `vtest.avi`.
- The app has no menu item to change the embedder. Change `"embedder"` in `settings.json` and open
  the app again.

## [0.4.1] — 2026-10-16

### Fixed
- Live sources: an error in the ingest loop now ends the ffmpeg frame pipe, the recorder, the
  stall watchdog and the audio thread the same way as a normal end, and reaps each child. Before,
  a persistent error (the source restarts every 1 to 30 s) left zombie processes, a second audio
  thread on the same camera (duplicate audio windows), and one User-Agent entry per restart.

## [0.4.0] — 2026-10-14

### Added
- A new search UI (`ui/index.html`, still one file with no build step): a command bar with
  recent queries and deep links (the query and the filters live in the URL hash), a timeline
  with tracklet density for each camera that you can drag to select a time range, result cards
  that scrub through their crops on hover, keyboard navigation (`/`, `j`, `k`, `Enter`, `p`,
  `Esc`), a slide-in detail panel with the clip, the path, the crops, "more like this", a clip
  download and a copy-link action, a **case** panel that keeps pinned results in the browser
  and exports them as JSON, audio results inline, and error toasts for every API error.
- A stall watchdog for live sources: when no frame arrives for 30 s, Retrace kills ffmpeg
  (SIGTERM, then SIGKILL) and restarts the source, which also resolves a YouTube URL again.
- HTTP inputs get `-rw_timeout 20 s`; YouTube media requests carry the User-Agent that yt-dlp
  used.
- `/ste`: the docs follow ASD-STE100 Simplified Technical English, checked by a bundled script.
- The in-process EmbeddingGemma 2 path (`--embedder gemma2-native`) builds and runs on Apple
  Silicon: `vtest.avi` indexed in 2 min 15 s with the model load, searches in 9 to 21 ms after the
  first query, and Spanish and Telugu queries return the same people as the English query.

### Changed
- README, CONTRIBUTING, SECURITY, the landing page and the issue templates are rewritten in
  Simplified Technical English with measured numbers only.
- `crates/llama-embed/build.rs` finds `cmake`, `git` and `ninja` in the usual package-manager
  directories when they are not on `PATH`, and links the llama.cpp archives whole and unbundled.

### Known limitation
- YouTube Live: without a proof-of-origin token provider for yt-dlp, YouTube serves about 30 s
  of segments and then answers 403. Retrace restarts the source each time (about one restart
  per minute). RTSP, HLS from your own cameras, SRT and files are not affected.


## [0.3.0] — 2026-10-13

First public release.

### Added

- `ingest`: ffmpeg → DEIM-D-FINE (ONNX) → sparse-fps tracker → tracklets with path, entry/exit
  edge, direction, dwell, best crops and a context frame → CLIP ViT-B/16 or SigLIP 2 embeddings →
  `sqlite-vec`. File replay, RTSP, HLS, SRT and YouTube Live sources with reconnect, a built-in
  fragmented-MP4 recorder, and backpressure (frames are skipped, never queued).
- `serve`: axum API and a single-file UI with filters, timeline lanes, result cards with
  bounding boxes, clip playback, trajectory overlay, "more like this", live feed (SSE), live
  view (`/live.mjpeg`), rules with backtest, and events.
- `search`, `annotate` (VLM captions through any OpenAI-compatible endpoint, grammar-enforced
  JSON, joined via BM25 + RRF), and `eval` (recall@k, mAP, latency on a golden set).
- EmbeddingGemma 2 as an embedder: via a local Python sidecar (`--embedder gemma2`) or in-process
  through llama.cpp + mtmd (`--embedder gemma2-native`, `--features gemma-native`). One space for
  text, images and audio; open-vocabulary audio search over 10 s windows; queries in 100+
  languages.
- Live rules: tracks that dwell 8 s or more are written provisionally and re-evaluated every 10 s
  of track time, so a "loitering for 45 s" rule fires while the person is still in frame.
- `eval/golden.vtest.json`: an honest golden set over OpenCV's public `vtest.avi`, and
  `scripts/e2e.sh`, which indexes that video into a throwaway database and checks search, clips,
  rules and eval end to end.
- Open-source scaffolding: CI (fmt, clippy, tests, release build on Linux and macOS), an on-demand
  end-to-end workflow, issue and PR templates, contributing guide, code of conduct, security
  policy.

### Fixed
- Tracklet ids were not reproducible between runs of the same video: tracks that closed on the
  same frame were persisted in `HashMap` iteration order. They are now written in id order, so a
  golden eval set keyed by id is stable (two consecutive `scripts/e2e.sh` runs agree exactly).
- Logs now go to stderr; stdout carries only machine-readable output (ingest stats JSON, eval lines).
- Review fixes before the first public release: an unknown `cam` in a search, a rule or a posted event is now an
  error instead of silently widening to every camera; `/clip` clamps `dur` to 1–60 s and `/api/similar` clamps `limit`
  to 1–500; the `/api/live` SSE task exits when the client disconnects instead of polling the
  database forever; the clip concat list uses a private temp file; score sorts are NaN-safe
  (`total_cmp`); the in-process EmbeddingGemma path checks image buffer sizes before handing them
  to llama.cpp; `--device cpu-only` is accepted by the ONNX side as documented; MediaMTX redirects
  percent-encode the camera name; redirected logs carry no ANSI codes.
- `scripts/mac-setup.sh` ran `rustup-init` even when Rust was already installed (operator
  precedence in `||`/`&&`).
- `scripts/demo-file.sh` wiped `./data` without asking; it now asks, or takes `FRESH=1`.
- The UI requested a favicon that did not exist (one console error per page load).
- Clippy clean-up across the crate (collapsed `if`s, needless borrows, `is_multiple_of`).

[Unreleased]: https://github.com/humancto/retrace/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/humancto/retrace/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/humancto/retrace/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/humancto/retrace/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/humancto/retrace/releases/tag/v0.3.0
