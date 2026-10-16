# Changelog

All notable changes to Retrace are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/humancto/retrace/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/humancto/retrace/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/humancto/retrace/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/humancto/retrace/releases/tag/v0.3.0
