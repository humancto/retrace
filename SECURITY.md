# Security policy

Retrace processes camera video. This data is sensitive. We answer reports quickly. This document
follows ASD-STE100 Simplified Technical English.

## Supported versions

| Version        | Supported |
| -------------- | --------- |
| 0.3.x (`main`) | yes       |
| before 0.3     | no        |

## Report a vulnerability

Do not open a public issue for a security problem.

Use the private report function of GitHub. Open the **Security** tab of the repository and select
**Report a vulnerability**. You get an acknowledgement in 72 hours. You get a fix or a mitigation
plan in 14 days for each confirmed problem.

Useful details: the command line that you ran, the source type (file, RTSP, HLS, YouTube), the
embedder, and a small reproduction if you have one.

## Threat model

The server binds to `127.0.0.1:8080` by default. It has no authentication. If you set
`--bind 0.0.0.0:…`, put the server behind a component that authenticates, for example Tailscale,
a reverse proxy with authentication, or an SSH tunnel. Each person who can reach the port can
search your video and play clips.

A stream URL can contain a password (`rtsp://user:pass@…`). Retrace stores the URL in the SQLite
database (`cameras.source`) and returns it from `/api/stats`. The URL is also visible in the
process list. Treat `data/` as sensitive.

Retrace sends data to the network only for these items: the model downloads on the first run
(GitHub releases for DEIM and CLIP, Hugging Face for SigLIP 2 and EmbeddingGemma 2, `cdn.pyke.io`
for the onnxruntime binary at build time), the sources that you give, and `yt-dlp` for YouTube
URLs. Retrace has no telemetry. Use `--models-dir` on a computer with no network.

ffmpeg parses media that you do not control. Keep ffmpeg current (`brew upgrade ffmpeg`).

The optional VLM annotator (`retrace annotate`) sends crops to the OpenAI-compatible endpoint that
you select. Point it only at a server that you control.

## Scope

In scope: each item in this repository (the Rust binary, the UI, the Python sidecar, the native
crate, the scripts). Out of scope: vulnerabilities in the models, ffmpeg, onnxruntime, or
llama.cpp. Report those upstream. Also tell us, so we can update the pin.
