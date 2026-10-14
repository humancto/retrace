#!/usr/bin/env bash
# Starts the EmbeddingGemma 2 sidecar for `retrace --embedder gemma2` (first run downloads the model).
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python3 -c "import sentence_transformers" 2>/dev/null; then
  echo "installing sidecar dependencies (sentence-transformers, torch, pillow, soundfile)…"
  python3 -m pip install -U "sentence-transformers>=6.1" "transformers>=5" torch pillow soundfile
fi
exec python3 sidecar/embed_server.py --dim "${RETRACE_EMBED_DIM:-256}" --port "${RETRACE_EMBED_PORT:-8082}" "$@"
