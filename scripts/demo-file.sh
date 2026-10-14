#!/usr/bin/env bash
# Replay a video file into an index and serve the UI.
#
#   scripts/demo-file.sh video.mp4 [camname] [embedder]
#   FRESH=1 scripts/demo-file.sh video.mp4      # wipe ./data first (asks unless FRESH=1 is set)
set -euo pipefail
cd "$(dirname "$0")/.."
VIDEO=${1:?usage: scripts/demo-file.sh video.mp4 [camname] [embedder]}
CAM=${2:-demo}
EMB=${3:-siglip2-b16-256}
BIN=./target/release/retrace; [ -x "$BIN" ] || BIN=./target/debug/retrace
[ -x "$BIN" ] || { echo "no binary: run cargo build --release first"; exit 1; }
if [ -d data ] && [ "${FRESH:-0}" != "1" ]; then
  read -r -p "./data already holds an index. Wipe it and start fresh? [y/N] " yn || yn=n
  case "$yn" in [Yy]*) FRESH=1 ;; *) echo "keeping ./data (new tracklets are appended)";; esac
fi
[ "${FRESH:-0}" = "1" ] && rm -rf data
"$BIN" --embedder "$EMB" ingest --source "$VIDEO" --cam "$CAM"
exec "$BIN" --embedder "$EMB" serve
