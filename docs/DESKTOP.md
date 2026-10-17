# Retrace Desktop

Status: design for v0.5.0. The app does not exist yet. This document gives the design and the build steps.

## Goal

Retrace Desktop is one file that you download and open. It analyses video on your machine. You do not need a
terminal, Homebrew or Python.

The app contains the Retrace engine and both embedders:

- SigLIP 2 (the default).
- EmbeddingGemma 2, which runs in the same process.

On the first run, the app downloads ffmpeg, ffprobe, yt-dlp and the weights of the embedder that you select. A
window shows the progress of each download. After setup, the app shows the Retrace UI in a native window.

Only these first-run downloads leave the machine. Video URLs and camera URLs that you enter also use the network.
The UI uses system fonts, so it does not load fonts from the internet.

## Architecture

The app is a Tauri v2 application in `desktop/src-tauri/`. It is a workspace member, but it is not a default
member. A root `cargo build` does not build it.

The engine runs in the app process. The app does not start a second process.

1. The app opens the setup window. This window loads the local page `setup.html`.
2. A background task downloads the missing tools and models. It sends progress events to the setup window.
3. The task opens the database, loads the embedder and starts the HTTP engine on `127.0.0.1` with a free port.
4. The same window then goes to `http://127.0.0.1:<port>/`. The engine UI has no access to Tauri IPC.

Engine changes outside `desktop/`:

- `src/lib.rs` makes the engine a library. `src/main.rs` is a thin CLI over this library. `src/main.rs` declares
  no modules of its own, so each module compiles only one time.
- `api::bind` binds the port before the server starts and returns a handle that stops the server cleanly.
- `tools` finds ffmpeg, ffprobe and yt-dlp. The search order is: the app tools folder, the folder of the
  executable, `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`. Every call to these tools goes through this module.
  yt-dlp always gets `--ffmpeg-location`.
- The engine records each child process (ffmpeg, yt-dlp, the recorder). Thus the app can stop all of them when it quits.
- The UI uses a system font stack.

Menu actions:

- **File > Open Video…** adds the file to the ingest queue.
- **File > Add Camera…** opens a small local window. You enter a name and a stream URL.
- **Help** opens the data folder, the GitHub page and the third-party software list.

You can also drag a video into the window, use Finder "Open With", or start the app with `--open <file>`.

One worker thread does the ingest work, one file at a time. This limits memory use. Only one app instance runs.
A second launch sends its files to the first instance.

When you quit, the app stops the ingest, waits up to 3 seconds for the threads, stops all child processes and
stops the server. Then it kills any child process that is still running.

## Data layout

The app keeps all data in one folder:

`~/Library/Application Support/com.humancto.retrace/`

The environment variable `RETRACE_APP_DATA` changes this folder. Tests use it.

| Path                  | Contents                                        |
| --------------------- | ----------------------------------------------- |
| `bin/`                | ffmpeg, ffprobe, yt-dlp and their licence files |
| `models/`             | model weights (see below)                       |
| `cache/`              | downloaded web videos                           |
| `logs/`               | `retrace.log`, a new file each day              |
| `rec/`                | live camera recordings                          |
| `index-<embedder>.db` | the search index for one embedder               |
| `crops-<embedder>/`   | the image crops for one embedder                |
| `settings.json`       | embedder, ingest settings, tool paths           |

Each embedder has its own index and crops folder. The app never mixes vectors from two embedders. In names,
`:` and `/` change to `-`. For example, `gemma2-native:256` gives `index-gemma2-native-256.db`.

All paths are absolute. A macOS app starts in the folder `/`, so relative paths are not safe.

To change the embedder, the app stops the server, opens the other index, starts a new server and loads it in the
window.

### Models

The app puts the weights in a flat layout in `models/`. The engine reads this layout directly and does not
download anything itself.

| Item                           | Files                                                             | Size         |
| ------------------------------ | ----------------------------------------------------------------- | ------------ |
| Detector (always)              | `deim/dfine-s-coco.onnx`                                          | small        |
| SigLIP 2 (default)             | `siglip2-b16-256/` vision model, text model and 3 tokenizer files | about 1.4 GB |
| EmbeddingGemma 2 (if selected) | `embeddinggemma-2/` model GGUF and mmproj GGUF                    | about 0.9 GB |

The app downloads only the embedder that you select.

The downloader:

- writes to a `.part` file and continues a stopped download,
- checks the SHA-256 hash of each file,
- renames the file only after the check passes,
- uses fixed Hugging Face commit URLs, not `main`.

If the hash is wrong, the downloader deletes the file and tries one more time. Then it shows the error.

### Tools

- ffmpeg and ffprobe: a fixed GPL build from martin-riedl.de. Retrace uses libx264 to make clips, so setup checks
  that the build has the libx264 encoder.
- yt-dlp: a fixed release from GitHub.

The app downloads these tools on the first run. The DMG does not contain them. You can select tools that are
already installed. Then the app keeps their absolute paths in `settings.json`.

## Build

Requirements: Rust, `cargo install tauri-cli --version "^2" --locked`, and cmake for the EmbeddingGemma build.

```sh
cd desktop/src-tauri
MACOSX_DEPLOYMENT_TARGET=14.0 CARGO_TARGET_DIR=../../target-desktop cargo tauri build
```

Output:

- `target-desktop/release/bundle/macos/Retrace.app`
- `target-desktop/release/bundle/dmg/Retrace_0.5.0_aarch64.dmg`

The minimum macOS version is 14.0. The binary contains ONNX Runtime and llama.cpp, so the bundle
contains no extra libraries.

For CI release builds, set `LLAMA_NATIVE=0`. This makes llama.cpp use generic CPU instructions, so the build runs
on older CPUs.

The root CLI build does not change: `cargo build --release` makes the `retrace` binary, and `scripts/e2e.sh`
passes.

## Signing

The build uses an ad-hoc signature (`signingIdentity "-"`). Apple does not notarize it.

To open an ad-hoc app the first time:

1. Open Retrace. macOS blocks it.
2. Open System Settings > Privacy & Security.
3. Click **Open Anyway**.

On macOS 15 and later, control-click > Open does not work for this app.

Alternative: run this command in Terminal.

```sh
xattr -dr com.apple.quarantine /Applications/Retrace.app
```

### Developer ID signing (human step)

A person with an Apple Developer ID must do this step. The build does not do it.

```sh
codesign --force --deep --options runtime --timestamp \
  --sign "Developer ID Application: <NAME> (<TEAMID>)" Retrace.app
ditto -c -k --keepParent Retrace.app Retrace.zip
xcrun notarytool submit Retrace.zip --apple-id <APPLE_ID> --team-id <TEAMID> --password <APP_PASSWORD> --wait
xcrun stapler staple Retrace.app
```

Do not put Apple credentials in the repository or in CI logs.

## Third-party software

| Software                       | Licence                         |
| ------------------------------ | ------------------------------- |
| ffmpeg (martin-riedl.de build) | GPL-3.0                         |
| yt-dlp                         | Unlicense                       |
| D-FINE detector                | Apache-2.0                      |
| SigLIP 2 weights               | as published by the model owner |
| EmbeddingGemma weights         | as published by the model owner |
| Tauri                          | MIT or Apache-2.0               |
