#!/usr/bin/env python3
"""Retrace embedding sidecar: EmbeddingGemma 2 (text + images + audio) over localhost.

One embedding space for queries, crops, image sets and 16 kHz audio windows, so Retrace can
answer "a dog barking" with audio windows next to "a person in a red jacket" with crops.
Everything stays on this machine. Runs on Apple Silicon (MPS), CUDA or CPU.

    pip install -U "sentence-transformers>=6.1" "transformers>=5" torch pillow soundfile
    python3 sidecar/embed_server.py --dim 256            # http://127.0.0.1:8082
    retrace --embedder gemma2 serve --source door=rtsp://...

GET  /health -> {"model": "...", "dim": 256, "modalities": ["text", "image", "audio"]}
POST /embed  {"items": [
                {"kind": "query",  "text": "a dog barking"},
                {"kind": "doc",    "text": "...", "title": "..."},
                {"kind": "image",  "b64": ["<jpeg>"]},
                {"kind": "images", "b64": ["<jpeg>", "<jpeg>"]},        # one vector for a set of views
                {"kind": "audio",  "b64": ["<wav, 16 kHz mono>"]}]}
          -> {"dim": 256, "vectors": [[...], ...]}

Flags: --model (default google/embeddinggemma-2), --dim 768|512|256|128 (Matryoshka truncation),
--no-audio / --no-vision (skip an encoder, smaller model), --mock (no model; deterministic
vectors for plumbing tests), --selftest.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DEFAULT = "google/embeddinggemma-2"


class MockEmbedder:
    """Deterministic pseudo-embeddings so the Rust side can be exercised without the model."""

    def __init__(self, dim, modalities):
        self.dim = dim
        self.modalities = modalities
        self.model_name = "mock"

    def _vec(self, seed: bytes):
        h = hashlib.sha256(seed).digest()
        out = []
        i = 0
        while len(out) < self.dim:
            h = hashlib.sha256(h + bytes([i])).digest()
            out.extend(u / 2**31 - 1.0 for u in struct.unpack("<8I", h[:32]))
            i += 1
        out = out[: self.dim]
        n = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / n for x in out]

    def embed(self, items):
        vecs = []
        for it in items:
            kind = it.get("kind", "query")
            if kind in ("query", "doc"):
                words = " ".join(sorted((it.get("text") or "").lower().split()))
                vecs.append(self._vec(("text:" + words).encode()))
            else:
                blob = b"".join(base64.b64decode(b) for b in it.get("b64", []))
                vecs.append(self._vec(kind.encode() + hashlib.sha256(blob).digest()))
        return vecs


class GemmaEmbedder:
    def __init__(self, model_name, dim, audio=True, vision=True, device=None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        cfg = {}
        if not audio:
            cfg["audio_config"] = None
        if not vision:
            cfg["vision_config"] = None
        # bf16 where native (CUDA); fp32 elsewhere — never fp16 (NaNs, per the model card).
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        t = time.time()
        self.model = SentenceTransformer(
            model_name,
            device=device,
            config_kwargs=cfg,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model_name = model_name
        self.dim = dim
        self.device = device
        self.modalities = ["text"] + (["image"] if vision else []) + (["audio"] if audio else [])
        self.lock = threading.Lock()
        print(f"[sidecar] loaded {model_name} on {device} ({dtype}), dim {dim}, modalities {self.modalities}, {time.time()-t:.1f}s",
              file=sys.stderr, flush=True)

    def _encode(self, inp, prompt_name=None):
        kw = {"truncate_dim": self.dim, "normalize_embeddings": True, "convert_to_numpy": True}
        if prompt_name:
            kw["prompt_name"] = prompt_name
        with self.lock:
            v = self.model.encode(inp, **kw)
        return [float(x) for x in v.reshape(-1)[: self.dim]]

    def embed(self, items):
        from PIL import Image

        out = []
        for it in items:
            kind = it.get("kind", "query")
            if kind == "query":
                out.append(self._encode(it.get("text") or "", prompt_name="SearchQuery"))
            elif kind == "doc":
                title = it.get("title") or "none"
                out.append(self._encode(f"title: {title} | text: {it.get('text') or ''}"))
            elif kind in ("image", "images"):
                if "image" not in self.modalities:
                    raise ValueError("vision encoder not loaded")
                imgs = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in it.get("b64", [])]
                if not imgs:
                    raise ValueError("no image data")
                text = " ".join("<|image|>" for _ in imgs)
                out.append(self._encode({"text": text, "image": imgs}))
            elif kind == "audio":
                if "audio" not in self.modalities:
                    raise ValueError("audio encoder not loaded")
                wav = base64.b64decode(it.get("b64", [""])[0])
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav)
                    path = f.name
                try:
                    out.append(self._encode({"text": "<|audio|>", "audio": [path]}))
                finally:
                    os.unlink(path)
            else:
                raise ValueError(f"unknown kind {kind}")
        return out


def make_handler(embedder):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._json(200, {"model": embedder.model_name, "dim": embedder.dim,
                                 "modalities": embedder.modalities})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/embed"):
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                items = req.get("items") or []
                t = time.time()
                vecs = embedder.embed(items)
                self._json(200, {"dim": embedder.dim, "vectors": vecs, "ms": round((time.time() - t) * 1000, 1)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def selftest(embedder):
    from PIL import Image

    img = Image.new("RGB", (224, 224), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    items = [{"kind": "query", "text": "a red square"}, {"kind": "image", "b64": [base64.b64encode(buf.getvalue()).decode()]}]
    if "audio" in embedder.modalities:
        import math
        pcm = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000))) for i in range(16000))
        wav = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm
        items.append({"kind": "audio", "b64": [base64.b64encode(wav).decode()]})
    t = time.time()
    vecs = embedder.embed(items)
    dot = sum(a * b for a, b in zip(vecs[0], vecs[1]))
    print(f"selftest ok: {len(vecs)} vectors of dim {len(vecs[0])} in {time.time()-t:.2f}s; text·image cosine = {dot:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--dim", type=int, default=256, choices=[768, 512, 256, 128])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--mock", action="store_true", help="no model: deterministic vectors for plumbing tests")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.mock:
        mods = ["text"] + ([] if a.no_vision else ["image"]) + ([] if a.no_audio else ["audio"])
        emb = MockEmbedder(a.dim, mods)
    else:
        emb = GemmaEmbedder(a.model, a.dim, audio=not a.no_audio, vision=not a.no_vision, device=a.device)
    if a.selftest:
        selftest(emb)
        return
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(emb))
    print(f"[sidecar] serving on http://{a.host}:{a.port}  (model={emb.model_name}, dim={emb.dim}, modalities={emb.modalities})",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
