# Retrace Desktop

Status: built in v0.5.0. The release artifact is `Retrace_0.5.0_aarch64.dmg` for macOS 14 or later on Apple
Silicon. The Linux build is best effort.

## Goal

Retrace Desktop is one file that you download and open. It analyses video on your machine. You do not need a
terminal, Homebrew, Rust or Python.

The app contains the Retrace engine and both embedders:

- SigLIP 2 (`siglip2-b16-256`, the default).
- EmbeddingGemma 2 (`gemma2-native`), which runs in the same process through llama.cpp.

On the first run, the app downloads ffmpeg, ffprobe, yt-dlp, the detector and the weights of the selected
embedder. A window shows the progress of each download. After setup, the app shows the Retrace UI in a native
window.

These first-run downloads use the network. Video URLs and camera URLs that you enter also use the network.
The UI uses the system fonts and loads nothing from the network. The engine does not send video, queries or
results to the network.

## Install

1. Download `Retrace_0.5.0_aarch64.dmg` from <https://github.com/humancto/retrace/releases/latest>.
2. Open the DMG and drag Retrace to Applications.
3. Open Retrace. See [Signing](#signing) for the first open of an unsigned build.

## Architecture

The app is a Tauri v2 application in `desktop/src-tauri/` (package `retrace-desktop`, binary `Retrace`). It is a
workspace member, but it is not a default member. A root `cargo build` does not build it.

The engine runs in the app process. The app does not start a second engine process.

1. The app opens the setup window. This window loads the local page `desktop/setup-ui/setup.html`.
2. A background task downloads the missing tools and models. It sends progress events to the setup window.
3. The task opens the database, loads the embedder and starts the HTTP engine on `127.0.0.1` with a free port.
4. The same window then goes to `http://127.0.0.1:<port>/`. The engine UI has no access to Tauri IPC.

| File                   | Function                                                               |
| ---------------------- | ---------------------------------------------------------------------- |
| `src/main.rs`          | entry point, arguments, single instance, windows, quit                 |
| `src/paths.rs`         | the data layout; all paths are absolute                                |
| `src/settings.rs`      | `settings.json` and the `RETRACE_SETUP` override                       |
| `src/setup.rs`         | the first-run plan, progress events, model seed                        |
| `src/tools.rs`         | ffmpeg, ffprobe and yt-dlp downloads, the libx264 check, licence texts |
| `src/engine.rs`        | loads the index and the embedder, binds the server, opens the UI       |
| `src/worker.rs`        | the ingest queue (one file at a time)                                  |
| `src/menu.rs`          | the app menu                                                           |
| `src/nav.rs`           | the navigation policy and the download handler                         |
| `src/logging.rs`       | the log file                                                           |
| `setup-ui/setup.html`  | the setup page: progress, **Try again**, **Use the installed tools**   |
| `setup-ui/camera.html` | the **Add Camera…** form                                               |

Engine changes outside `desktop/`:

- `src/lib.rs` makes the engine a library. `src/main.rs` is a thin CLI over this library. `src/main.rs` declares
  no modules of its own, so each module compiles only one time.
- `api::bind` binds the port before the server starts and returns a handle that stops the server cleanly. The
  CLI `serve` command uses the same function.
- `src/tools.rs` finds ffmpeg, ffprobe and yt-dlp. The search order is: the app tools folder, the folder of the
  executable, `PATH`, `/opt/homebrew/bin`, `/usr/local/bin`. Every call to these tools goes through this module.
  yt-dlp always gets `--ffmpeg-location`, because an app that Finder starts has no Homebrew folder on `PATH`.
- `src/models/fetch.rs` downloads the model files into a flat folder (see [Models](#models)).
- The engine records each child process (the ffmpeg frame pipe, the audio ffmpeg, yt-dlp, the recorder and the
  clip ffmpeg). Thus the app can stop all of them when it quits.
- `Embedder::release` frees the model before the process exits.

Menu actions:

- **File > Open Video…** adds the file to the ingest queue.
- **File > Add Camera…** opens a small local window. You enter a name and a stream URL.
- **Help** opens the data folder, the GitHub page and the third-party software list.

You can also drag a video into the window, use Finder "Open With", or start the app with `--open <file>`.

One worker thread does the ingest work, one file at a time. This limits memory use. Only one app instance runs.
A second launch sends its files to the first instance.

While the app indexes a file, the window title shows `Indexing <file>…`. When the ingest ends, the title
shows the number of tracklets that it added, for example `Retrace — 36 tracklets indexed from vtest.avi`.

A download link in the UI (for example **Download clip**) saves the file to your Downloads folder. The app opens
links to other sites in your default browser.

### Quit

When you quit, the app does these steps:

1. It stops the ingest and waits up to 3 seconds for the threads.
2. It stops all registered child processes and stops the server.
3. It frees the embedder.
4. It kills each child process that is still running.

The embedder must be free before the process exits. EmbeddingGemma keeps llama.cpp Metal buffers, and llama.cpp
stops the process with an error at exit if these buffers are still allocated. An ingest that a quit stops is
logged as "stopped before the end"; it is not counted as done.

## Data layout

The app keeps all data in one folder:

`~/Library/Application Support/com.humancto.retrace/`

| Path                  | Contents                                                   |
| --------------------- | ---------------------------------------------------------- |
| `bin/`                | ffmpeg, ffprobe, yt-dlp and their licence files            |
| `models/`             | model weights (see below)                                  |
| `cache/`              | downloaded web videos                                      |
| `logs/`               | `retrace.log`, a new file each day                         |
| `rec/`                | live camera recordings                                     |
| `index-<embedder>.db` | the search index for one embedder                          |
| `crops-<embedder>/`   | the image crops for one embedder                           |
| `settings.json`       | embedder, ingest settings, tool source and tool paths      |
| `THIRD-PARTY.txt`     | the third-party software list that the **Help** menu opens |

Each embedder has its own index and crops folder. The app never mixes vectors from two embedders. In names,
`:` and `/` change to `-`. For example, `gemma2-native:256` gives `index-gemma2-native-256.db`.

All paths are absolute. A macOS app starts in the folder `/`, so relative paths are not safe.

Keep this folder on a local disk. Do not put the SQLite index in a folder that a sync service changes.

### Change the embedder

The app has no menu item for the embedder. To change it:

1. Quit Retrace.
2. In `settings.json`, set `"embedder"` to `"gemma2-native"` (or `"siglip2-b16-256"`).
3. Open Retrace. If the weights of that embedder are missing, the setup window downloads them first.

The default ingest settings are the same as `scripts/e2e.sh`: detector `deim-dfine-s`, 4 fps, width 640, minimum
crop height 40 px.

### Models

The app puts the weights in a flat layout in `models/`. The engine reads this layout directly. It does not use the
`usls` cache of the CLI.

| Item                           | Files                                                                                 | Size         |
| ------------------------------ | ------------------------------------------------------------------------------------- | ------------ |
| Detector (always)              | `deim/dfine-s-coco.onnx`                                                              | 42 MB        |
| SigLIP 2 (default)             | `siglip2-b16-256/` vision model, text model and 3 tokenizer files                     | about 1.5 GB |
| EmbeddingGemma 2 (if selected) | `embeddinggemma-2/embeddinggemma-2-Q8_0.gguf` and `mmproj-embeddinggemma-2-Q8_0.gguf` | 865 MB       |

The app downloads only the embedder that you select. A cold first run with the default embedder downloads 9
files, 1,671,494,078 bytes in total (tools, detector and SigLIP 2). In our test on 2026-10-16, the engine was
ready about 5 minutes after the launch.

The downloader:

- writes to a `.part` file and continues a stopped download,
- checks the SHA-256 hash of each file,
- renames the file only after the check passes,
- renames `vision_model.onnx` last, because the engine uses this file to find a complete SigLIP folder,
- uses fixed Hugging Face commit URLs, not `main` (SigLIP 2 commit `d1114256`).

If the hash is wrong, the downloader deletes the file and tries one more time. Then it shows the error with
**Try again**.

### Tools

| Tool              | Source                                                 | macOS arm64 size |
| ----------------- | ------------------------------------------------------ | ---------------- |
| ffmpeg 9.0.2      | fixed GPL build from <https://ffmpeg.martin-riedl.de/> | 28 MB (zip)      |
| ffprobe 9.0.2     | the same build                                         | 28 MB (zip)      |
| yt-dlp 2026.08.19 | fixed release from GitHub (`yt-dlp_macos`)             | 37 MB            |

Retrace uses libx264 to make clips, so setup checks that the ffmpeg build has the libx264 encoder.

The app downloads these tools on the first run. The DMG does not contain them. To use tools that are already
installed, click **Use the installed tools** in the setup window. Then the app keeps their absolute paths in
`settings.json`. If the installed ffmpeg has no libx264, setup shows a warning: clip export does not work.

### Environment variables

| Variable               | Effect                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| `RETRACE_APP_DATA`     | use this folder for all data, not the default folder                                        |
| `RETRACE_SETUP`        | `download` or `system`: the tool source, without the question in the setup page             |
| `RETRACE_MODELS_SEED`  | copy model files from this folder (same layout as `models/`) after a size and SHA-256 check |
| `RETRACE_DOWNLOAD_DIR` | save UI downloads to this folder, not to Downloads                                          |
| `RETRACE_LOG_STDERR`   | also write the log to stderr in a release build                                             |

Example of an unattended test run with an empty data folder:

```sh
RETRACE_APP_DATA=/tmp/retrace-test RETRACE_SETUP=download \
  /Applications/Retrace.app/Contents/MacOS/Retrace --open ~/Movies/vtest.avi
```

The log in `/tmp/retrace-test/logs/` shows the download progress and the line
`engine listening on http://127.0.0.1:<port>/`.

## Build

Requirements:

- Rust (see `rust-version` in `Cargo.toml`) and the Xcode command line tools.
- `cargo install tauri-cli --version "^2" --locked`.
- cmake, git and libclang for the EmbeddingGemma build (llama.cpp).

cmake must be on `PATH` and must work. If the cmake on `PATH` does not work, put a working cmake first. On the
development Mac, the working cmake is in a Python virtual environment:

```sh
export PATH="$HOME/.venvs/cmake/bin:$PATH"
```

Build the app from `desktop/src-tauri`:

```sh
cd desktop/src-tauri
PATH="$HOME/.venvs/cmake/bin:$PATH" MACOSX_DEPLOYMENT_TARGET=14.0 CARGO_TARGET_DIR=target-desktop \
  cargo tauri build
```

Output:

- `desktop/src-tauri/target-desktop/release/bundle/macos/Retrace.app`
- `desktop/src-tauri/target-desktop/release/bundle/dmg/Retrace_0.5.0_aarch64.dmg`

Lint and test the desktop crate from the repository root:

```sh
PATH="$HOME/.venvs/cmake/bin:$PATH" CARGO_TARGET_DIR=target-desktop \
  cargo clippy -p retrace-desktop --all-targets -- -D warnings \
  -A clippy::too_many_arguments -A clippy::large_enum_variant
PATH="$HOME/.venvs/cmake/bin:$PATH" CARGO_TARGET_DIR=target-desktop cargo test -p retrace-desktop
```

The minimum macOS version is 14.0. The binary contains ONNX Runtime and llama.cpp, so the bundle contains no
extra libraries (`otool -L` shows system libraries only).

On macOS, `crates/llama-embed/build.rs` always compiles llama.cpp with Metal and with generic CPU code
(`GGML_NATIVE=OFF`), so the app runs on all Apple Silicon Macs. On Linux it compiles for the CPU of the build
machine (`GGML_NATIVE=ON`). CI sets `LLAMA_NATIVE=0`, but `build.rs` does not read this variable yet.

The root CLI build does not change: `cargo build --release` makes the `retrace` binary, and `scripts/e2e.sh`
passes.

## CI

The workflow `.github/workflows/desktop.yml` builds the app. It runs on a push to `main`, on a `v*` tag, on a pull
request that changes `desktop/`, `src/`, `crates/`, `ui/`, `Cargo.toml` or `Cargo.lock`, and on a manual start.

| Job          | Runner         | Bundles           | Failure                           |
| ------------ | -------------- | ----------------- | --------------------------------- |
| macOS arm64  | `macos-14`     | `app`, `dmg`      | fails the workflow                |
| Linux x86_64 | `ubuntu-22.04` | `appimage`, `deb` | best effort (`continue-on-error`) |

Each job runs `cargo fmt --check`, the desktop clippy command above, `cargo test -p retrace-desktop` and
`cargo tauri build`. It uploads the bundles as the artifact `retrace-desktop-<OS>`. The macOS job also uploads
`Retrace_macos_aarch64.app.zip`. On a `v*` tag, a separate `release` job downloads the artifacts of both jobs
and attaches them to the GitHub release of that tag. Only that job gets a token that can write to the
repository; the other jobs have read access only.

The other CI jobs build only the root crate, so they do not compile llama.cpp.

## Memory

Peak resident memory (RSS) for one `--open` of the sample `vtest.avi` (macOS arm64, measured with
`/usr/bin/time -l` on 2026-10-16):

| Embedder          | Peak RSS                            |
| ----------------- | ----------------------------------- |
| `siglip2-b16-256` | about 6.6 GiB (7,120,666,624 bytes) |
| `gemma2-native`   | about 2.4 GiB (2,583,642,112 bytes) |

The engine loads one embedder, and each ingest loads its own detector and embedder. For this reason, one worker
indexes one file at a time.

## Manual check: download handler (owner step)

Automated tests cover the download policy only. A person must do this check one time per release, with a real
click. Do not use synthetic mouse or keyboard events.

1. Start the app with a temporary download folder:
   `RETRACE_DOWNLOAD_DIR=/tmp/retrace-dl Retrace.app/Contents/MacOS/Retrace`.
2. Search for a result, open a tracklet and click **Download clip**.
3. Make sure that the clip is in `/tmp/retrace-dl` and that `logs/retrace.log.*` contains
   `download finished`.

## Signing

The build uses an ad-hoc signature (`signingIdentity "-"`). Apple does not notarize it.

To open an ad-hoc app the first time:

- On macOS 14: right-click Retrace in Applications and select **Open**. Then click **Open** in the dialog.
- On macOS 15 and later: open Retrace one time (macOS blocks it). Then open System Settings > Privacy & Security
  and click **Open Anyway**. On these versions, right-click > Open does not work for this app.

Alternative: run this command in Terminal.

```sh
xattr -dr com.apple.quarantine /Applications/Retrace.app
```

To check the signature of a build:

```sh
codesign -dv Retrace.app                    # Signature=adhoc
codesign --verify --deep --strict Retrace.app
```

### Developer ID signing and notarization (human step)

A person with an Apple Developer ID must do this step. The build and CI do not do it. Do not put Apple
credentials in the repository or in CI logs.

1. Find the signing identity:

   ```sh
   security find-identity -v -p codesigning
   ```

2. Sign the app:

   ```sh
   codesign --force --deep --options runtime --timestamp \
     --sign "Developer ID Application: <NAME> (<TEAMID>)" Retrace.app
   ```

3. Send the app to Apple for notarization and attach the ticket:

   ```sh
   ditto -c -k --keepParent Retrace.app Retrace.zip
   xcrun notarytool submit Retrace.zip --apple-id <APPLE_ID> --team-id <TEAMID> \
     --password <APP_PASSWORD> --wait
   xcrun stapler staple Retrace.app
   ```

4. Make a new DMG from the stapled app, then notarize and staple the DMG:

   ```sh
   hdiutil create -volname Retrace -srcfolder Retrace.app -ov -format UDZO Retrace_0.5.0_aarch64.dmg
   xcrun notarytool submit Retrace_0.5.0_aarch64.dmg --apple-id <APPLE_ID> --team-id <TEAMID> \
     --password <APP_PASSWORD> --wait
   xcrun stapler staple Retrace_0.5.0_aarch64.dmg
   ```

5. Check the result:

   ```sh
   spctl -a -vv Retrace.app                  # source=Notarized Developer ID
   xcrun stapler validate Retrace_0.5.0_aarch64.dmg
   ```

## Third-party software

**Help > Third-Party Software** opens `THIRD-PARTY.txt` with the versions, licences and source links.

| Software                       | Licence            |
| ------------------------------ | ------------------ |
| ffmpeg (martin-riedl.de build) | GPL-3.0-or-later   |
| yt-dlp                         | The Unlicense      |
| D-FINE detector                | Apache-2.0         |
| SigLIP 2 weights               | Apache-2.0         |
| EmbeddingGemma 2 weights       | Gemma Terms of Use |
| Tauri                          | MIT or Apache-2.0  |
