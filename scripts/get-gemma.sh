#!/usr/bin/env bash
# Downloads EmbeddingGemma 2 (ggml-org's GGUF build: Q8_0 text model + Q8_0 mmproj with the image and
# audio encoders, ~865 MB) into $RETRACE_MODELS/embeddinggemma-2 (default ./models/embeddinggemma-2)
# for `retrace --embedder gemma2-native`. Apache-2.0 weights; nothing else is fetched.
set -euo pipefail
DIR="${RETRACE_MODELS:-models}/embeddinggemma-2"
REPO="https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main"
VARIANT="${1:-Q8_0}"   # Q8_0 (recommended) or BF16
mkdir -p "$DIR"
for f in "embeddinggemma-2-${VARIANT}.gguf" "mmproj-embeddinggemma-2-${VARIANT}.gguf"; do
  if [[ -s "$DIR/$f" ]]; then echo "have $DIR/$f"; continue; fi
  echo "downloading $f …"
  curl -L --fail --progress-bar -o "$DIR/$f.part" "$REPO/$f" && mv "$DIR/$f.part" "$DIR/$f"
done
ls -la "$DIR"
echo "ready: retrace --embedder gemma2-native --models-dir ${RETRACE_MODELS:-models} serve …"
