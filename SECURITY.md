# Security Policy

Retrace processes camera footage, which is about as sensitive as data gets. We take reports
seriously and will respond quickly.

## Supported versions

| Version        | Supported |
| -------------- | --------- |
| 0.3.x (`main`) | ✅        |
| < 0.3          | ❌        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Use GitHub's private reporting: go to the repository's **Security** tab and choose
**Report a vulnerability**. You will get an acknowledgement within 72 hours and a fix or a
mitigation plan within 14 days for anything confirmed.

Useful details: the command line you ran, the source type (file, RTSP, HLS, YouTube), the
embedder, and a minimal reproduction if you have one.

## Threat model, in short

- **The server binds to `127.0.0.1:8080` by default** and has no authentication. If you pass
  `--bind 0.0.0.0:…`, put it behind something that authenticates (Tailscale, a reverse proxy
  with auth, an SSH tunnel). Anyone who can reach the port can search your footage and play
  clips.
- **Stream URLs can carry credentials** (`rtsp://user:pass@…`). They are stored in the SQLite
  database (`cameras.source`) and returned by `/api/stats`. Treat `data/` as sensitive.
- **Outbound network calls are limited to model downloads** on first run (GitHub releases for
  DEIM and CLIP, Hugging Face for SigLIP 2 and EmbeddingGemma 2, `cdn.pyke.io` for the
  onnxruntime binary at build time) plus whatever sources you point it at. `yt-dlp` resolves
  YouTube URLs when you use them. There is no telemetry. Use `--models-dir` to run fully
  offline.
- **ffmpeg parses untrusted media.** Keep it current (`brew upgrade ffmpeg`).
- **The optional VLM annotator** (`retrace annotate`) talks to an OpenAI-compatible endpoint
  you choose. Point it only at a server you control.

## Scope

In scope: anything in this repository (the Rust binary, the UI, the Python sidecar, the native
crate, the scripts). Out of scope: vulnerabilities in upstream models, ffmpeg, onnxruntime or
llama.cpp themselves (report those upstream, but tell us too so we can bump the pin).
