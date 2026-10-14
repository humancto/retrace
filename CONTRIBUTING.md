# Contributing to Retrace

Thanks for looking under the hood. Retrace is small enough that one person can hold the whole
pipeline in their head, and we would like to keep it that way. This page tells you how to get a
working build, what a good change looks like, and what happens after you open a pull request.

## Ground rules

- **Nothing leaves the machine.** Every feature must work with the network cable pulled once the
  model weights are cached. No telemetry, no cloud calls, no "optional" phone-home.
- **Open weights only.** New models need a permissive license (Apache-2.0, MIT, or similar) and a
  reproducible download.
- **Measure, don't assert.** Claims about speed or recall come with the command that produced them.

## Getting a build

```bash
brew install ffmpeg rust        # or rustup; Rust 1.88+
git clone https://github.com/humancto/retrace && cd retrace
cargo build --release
scripts/e2e.sh                  # downloads a public sample video + weights, runs the whole pipeline
```

The first build compiles onnxruntime bindings, `tokenizers`, and `image`; expect a few minutes.
`scripts/e2e.sh` is the test you run before opening a PR. It uses a temporary database and never
touches `./data`.

## Project layout

| Path                      | What lives there                                                      |
| ------------------------- | --------------------------------------------------------------------- |
| `src/ingest.rs`           | ffmpeg → detector → tracker → tracklet close → embed → store          |
| `src/track.rs`            | the sparse-fps tracker (unit tests live here)                         |
| `src/models.rs`           | detector / embedder loading, model download + cache                   |
| `src/db.rs`               | SQLite schema, `sqlite-vec` and FTS5 tables                           |
| `src/search.rs`           | query rewrite, KNN + BM25, reciprocal-rank fusion, dynamic threshold  |
| `src/api.rs`              | axum routes and the clip cutter                                       |
| `src/rules.rs`            | standing queries and their backtest                                   |
| `src/audio.rs`            | 16 kHz audio windows for the EmbeddingGemma 2 path                    |
| `src/overlay.rs`          | annotated live frames (`/live.mjpeg`)                                 |
| `ui/index.html`           | the whole UI, one file, no build step                                 |
| `sidecar/embed_server.py` | EmbeddingGemma 2 over localhost (text + image + audio)                |
| `crates/llama-embed`      | EmbeddingGemma 2 in-process via llama.cpp (`--features gemma-native`) |
| `eval/`                   | golden query sets for `retrace eval`                                  |
| `scripts/`                | setup, demo, end-to-end test                                          |

## Making a change

1. Open an issue first for anything bigger than a bug fix, so we can agree on the shape.
2. Branch from `main`: `git checkout -b feat/short-name` (or `fix/`, `docs/`, `chore/`).
3. Keep commits small and use [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(track): add appearance term to association`.
4. Before pushing:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::large_enum_variant
   cargo test
   scripts/e2e.sh
   ```
5. Open the PR against `main` and fill in the template. CI runs fmt, clippy, tests and a release
   build on Linux and macOS; the end-to-end job runs on demand and weekly.

### What makes a change easy to merge

- It does one thing. Tracker changes and UI changes are two PRs.
- It is measured. For search or tracking changes, paste the `retrace eval` line before and after.
- It keeps the UI a single file with no build step.
- It respects the embedder boundary: an index is tied to one embedder, and code that mixes
  vectors from two must refuse rather than silently return garbage.
- It avoids `unsafe` unless it is in `crates/llama-embed`, where the FFI lives.

## Adding a model

Detectors: any D-FINE / DEIM / RT-DETR ONNX export that `usls` can load works via `--detector
path-or-url`. To add a named preset, extend the match in `src/models.rs` and add its download spec.

Embedders: implement the `Embedder` surface (text tower, image tower, dimension, name) in
`src/models.rs`. Add an eval run on `eval/golden.vtest.json` to the PR so we can compare.

## Reporting bugs

Use the bug template. The most useful attachments are the ingest log line (`ingest done … stats`),
the `/api/stats` JSON, and the search `plan` from the UI's Plan drawer.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind; assume good faith.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
