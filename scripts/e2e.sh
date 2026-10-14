#!/usr/bin/env bash
# End-to-end smoke test: index a public sample video into a throwaway database, serve it,
# and check the API answers (stats, search, clip, eval). Nothing touches ./data.
#
#   scripts/e2e.sh                       # downloads OpenCV's vtest.avi (7.8 MB) into a temp dir
#   VIDEO=~/Movies/driveway.mp4 scripts/e2e.sh
#   EMB=clip-b16 scripts/e2e.sh          # any --embedder value
#
# First run downloads the detector and embedder weights (a few hundred MB); later runs use the cache.
set -euo pipefail
cd "$(dirname "$0")/.."

EMB=${EMB:-siglip2-b16-256}
PORT=${PORT:-18080}
BIN=${BIN:-./target/release/retrace}
[ -x "$BIN" ] || BIN=./target/debug/retrace
[ -x "$BIN" ] || { echo "no binary: run cargo build --release first"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/retrace-e2e.XXXXXX")
SRV=""
cleanup() { [ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT

VIDEO=${VIDEO:-$TMP/vtest.avi}
if [ ! -s "$VIDEO" ]; then
  echo "▸ downloading OpenCV vtest.avi (80 s courtyard, ~10 people)"
  curl -sSL --fail -o "$VIDEO" https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
fi

echo "▸ ingest ($EMB)"
"$BIN" --embedder "$EMB" --db "$TMP/index.db" --crops "$TMP/crops" \
  ingest --source "$VIDEO" --cam e2e > "$TMP/ingest.json" 2> "$TMP/ingest.log"
# stdout ends with the pretty-printed stats object; take the last top-level {...} block.
TRACKLETS=$(python3 - "$TMP/ingest.json" <<'EOF'
import json, sys
text = open(sys.argv[1]).read()
start = text.rfind("\n{\n")
print(json.loads(text[start:] if start >= 0 else text)["tracklets"])
EOF
)
echo "  tracklets=$TRACKLETS"
[ "$TRACKLETS" -gt 0 ] || { echo "FAIL: ingest produced no tracklets"; cat "$TMP/ingest.log"; exit 1; }

echo "▸ serve on 127.0.0.1:$PORT"
"$BIN" --embedder "$EMB" --db "$TMP/index.db" --crops "$TMP/crops" \
  serve --bind "127.0.0.1:$PORT" > "$TMP/serve.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/api/stats" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "FAIL: server exited"; cat "$TMP/serve.log"; exit 1; }
  sleep 0.5
done

python3 - "$PORT" "$TMP" <<'EOF'
import json, sys, urllib.request, urllib.parse
port, tmp = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
def get(path):
    return urllib.request.urlopen(base + path, timeout=60)
stats = json.load(get("/api/stats"))
assert stats["tracklets"] > 0, stats
print(f"  stats: {stats['tracklets']} tracklets, classes {dict(stats['by_class'])}")
html = get("/").read().decode()
assert "<title>Retrace</title>" in html, "UI did not render"
r = json.load(get("/api/search?k=5&q=" + urllib.parse.quote("a person in a red jacket")))
assert r["hits"], "search returned no hits"
top = r["hits"][0]["tracklet"]
print(f"  search: {len(r['hits'])} hits in {r['timing_ms']['total']:.1f} ms, top id={top['id']} class={top['class']} score={r['hits'][0]['score']:.3f}")
clip = get(f"/clip?cam={urllib.parse.quote(top['cam_name'])}&t0={top['t0']}&t1={top['t1']}")
data = clip.read()
assert clip.headers.get("content-type", "").startswith("video/"), clip.headers.get("content-type")
assert len(data) > 10_000, f"clip too small: {len(data)} bytes"
print(f"  clip: {len(data)//1024} KB {clip.headers.get('content-type')}")
thumb = get(f"/thumb/{top['id']}").read()
assert thumb[:2] == b"\xff\xd8", "thumb is not a JPEG"
sim = json.load(get(f"/api/similar/{top['id']}?limit=3"))
assert "hits" in sim
req = urllib.request.Request(base + "/api/rules", data=json.dumps({"text": "person in a red jacket", "min_score": 0.13, "min_dwell_s": 0, "class": None}).encode(), headers={"content-type": "application/json"})
rule = json.load(urllib.request.urlopen(req, timeout=60))
print(f"  rule backtest: {rule['backtest']['matches']} matches")
EOF

if [ -f eval/golden.vtest.json ] && [ "${VIDEO##*/}" = "vtest.avi" ]; then
  echo "▸ eval (recall@5 on eval/golden.vtest.json)"
  "$BIN" --embedder "$EMB" --db "$TMP/index.db" --crops "$TMP/crops" \
    eval --golden eval/golden.vtest.json --k 5 2>/dev/null | tee "$TMP/eval.txt" | sed 's/^/  /'
  REC=$(grep -o 'recall@5=[0-9.]*' "$TMP/eval.txt" | tail -1 | cut -d= -f2)
  python3 -c "import sys; r=float(sys.argv[1]); sys.exit(0 if r >= 0.5 else 1)" "$REC" \
    || { echo "FAIL: recall@5=$REC below 0.5"; exit 1; }
fi

echo "✓ e2e passed"
