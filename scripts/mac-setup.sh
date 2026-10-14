#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: toolchain, ffmpeg, MediaMTX (optional hub), release build.
set -euo pipefail
command -v brew >/dev/null || { echo "install Homebrew first: https://brew.sh"; exit 1; }
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg
brew list mediamtx >/dev/null 2>&1 || brew install mediamtx
if ! command -v cargo >/dev/null; then
  brew install rustup-init
  rustup-init -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
cd "$(dirname "$0")/.."
cargo build --release
echo
echo "Built: ./target/release/retrace"
echo "Try:   ./target/release/retrace --embedder siglip2-b16-256 ingest --source /path/to/video.mp4 --cam demo"
echo "       ./target/release/retrace --embedder siglip2-b16-256 serve   # then open http://127.0.0.1:8080/"
