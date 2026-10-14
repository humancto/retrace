# Contributing to Retrace

This page tells you how to get a working build, what a good change looks like, and what happens
after you open a pull request. This document follows ASD-STE100 Simplified Technical English.

## Rules

- **No data leaves the machine.** Each feature must work with no network connection after the
  model weights are in the cache. Retrace has no telemetry and no cloud call.
- **Open weights only.** A new model must have a permissive license (Apache-2.0, MIT, or
  similar) and a download that each person can repeat.
- **Measure.** A claim about speed or recall must come with the command that produced the number.

## Get a build

```bash
brew install ffmpeg rust        # or rustup; Rust 1.88+
git clone https://github.com/humancto/retrace && cd retrace
cargo build --release
scripts/e2e.sh                  # downloads a public sample video and the weights, then runs the full pipeline
```

The first build compiles the onnxruntime bindings, `tokenizers`, and `image`. It takes some
minutes. `scripts/e2e.sh` is the test that you run before you open a pull request. It uses a
temporary database. It does not touch `./data`.

## Project layout

| Path                      | Content                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `src/ingest.rs`           | ffmpeg → detector → tracker → tracklet close → embed → store                  |
| `src/track.rs`            | the sparse-fps tracker and its unit tests                                     |
| `src/models.rs`           | detector and embedder loading, model download, cache                          |
| `src/db.rs`               | the SQLite schema, the `sqlite-vec` table, and the FTS5 table                 |
| `src/search.rs`           | query rewrite, KNN and BM25, reciprocal rank fusion, the dynamic threshold    |
| `src/api.rs`              | the axum routes and the clip cutter                                           |
| `src/rules.rs`            | standing queries and their backtest                                           |
| `src/audio.rs`            | 16 kHz audio windows for the EmbeddingGemma 2 path                            |
| `src/overlay.rs`          | annotated live frames (`/live.mjpeg`)                                         |
| `ui/index.html`           | the full UI in one file, with no build step                                   |
| `sidecar/embed_server.py` | EmbeddingGemma 2 over localhost (text, image, audio)                          |
| `crates/llama-embed`      | EmbeddingGemma 2 in the process through llama.cpp (`--features gemma-native`) |
| `eval/`                   | golden query sets for `retrace eval`                                          |
| `scripts/`                | setup, demo, and the end-to-end test                                          |

## Make a change

1. Open an issue first for each change that is larger than a bug fix. Then we agree on the shape.
2. Make a branch from `main`: `git checkout -b feat/short-name` (or `fix/`, `docs/`, `chore/`).
3. Make small commits. Use [Conventional Commits](https://www.conventionalcommits.org/), for
   example `feat(track): add an appearance term to the association`.
4. Before you push, run these commands:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::large_enum_variant
   cargo test
   scripts/e2e.sh
   ```
5. Open the pull request against `main` and complete the template. CI runs fmt, clippy, the
   tests, and a release build on Linux and macOS. The end-to-end job runs on request and each
   week.

### A change that is easy to merge

- It does one thing. A tracker change and a UI change are two pull requests.
- It has numbers. For a search or tracker change, add the `retrace eval` line before and after
  the change.
- It keeps the UI in one file with no build step.
- It respects the embedder boundary. An index belongs to one embedder. Code that mixes vectors
  from two embedders must stop with an error.
- It has no `unsafe` code outside `crates/llama-embed`, where the FFI lives.

## Add a model

Detectors: each D-FINE, DEIM, or RT-DETR ONNX export that `usls` can load works with
`--detector path-or-url`. To add a preset with a name, extend the match in `src/models.rs` and
add the download specification.

Embedders: implement the `Embedder` surface (text tower, image tower, dimension, name) in
`src/models.rs`. Add an eval run on `eval/golden.vtest.json` to the pull request, so we can
compare.

## Report a bug

Use the bug template. The most useful attachments are the ingest log line (`ingest done … stats`),
the `/api/stats` JSON, and the search `plan` from the Plan panel in the UI.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

When you contribute, you agree that your contribution is under the [MIT License](LICENSE).
