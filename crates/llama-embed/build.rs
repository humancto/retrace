//! Builds llama.cpp (pinned tag) with the mtmd multimodal library as static libraries and generates
//! Rust bindings for `llama.h` + `mtmd.h`. This is DigUp's `scripts/build-llama.sh` as a cargo build
//! script: libraries only (no tools, no server, nothing downloaded at run time), Metal embedded on
//! macOS, CPU elsewhere, plus the one-line "no logits" patch (an embedding model never produces
//! logits, and without the patch every input token reserves 1 MiB for Gemma's 262k vocabulary).
//!
//! Environment:
//!   LLAMA_CPP_DIR   use an existing llama.cpp checkout instead of cloning into OUT_DIR
//!   LLAMA_CPP_TAG   tag to clone (default b11461, the first tags with EmbeddingGemma 2 are b11452+)
//!   LIBCLANG_PATH   where bindgen finds libclang if it cannot on its own

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_TAG: &str = "b11461";

fn run(cmd: &mut Command, what: &str) {
    let status = cmd.status().unwrap_or_else(|e| panic!("{what}: {e}"));
    assert!(status.success(), "{what} failed ({status})");
}

fn patch_no_logits(src: &Path) {
    let f = src.join("src/llama-context.cpp");
    let Ok(text) = std::fs::read_to_string(&f) else {
        return;
    };
    let from = "    bool has_logits     = true;";
    let to = "    bool has_logits     = !(cparams.embeddings && !cparams.causal_attn); // retrace: no logits for embedding contexts";
    if text.contains(from) {
        std::fs::write(&f, text.replace(from, to)).expect("write patched llama-context.cpp");
    }
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_TAG");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");

    // 1. Source.
    let src = match env::var("LLAMA_CPP_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let tag = env::var("LLAMA_CPP_TAG").unwrap_or_else(|_| DEFAULT_TAG.into());
            let dir = out.join(format!("llama.cpp-{tag}"));
            if !dir.join("CMakeLists.txt").exists() {
                run(
                    Command::new("git")
                        .args([
                            "clone",
                            "--quiet",
                            "--depth",
                            "1",
                            "--branch",
                            &tag,
                            "https://github.com/ggml-org/llama.cpp",
                        ])
                        .arg(&dir),
                    "git clone llama.cpp (set LLAMA_CPP_DIR to use a local checkout)",
                );
            }
            dir
        }
    };
    patch_no_logits(&src);

    // 2. Configure + build the libraries.
    let build = out.join("build");
    std::fs::create_dir_all(&build).unwrap();
    let mut cfg = Command::new("cmake");
    cfg.arg("-S").arg(&src).arg("-B").arg(&build);
    if Command::new("ninja").arg("--version").output().is_ok() {
        cfg.args(["-G", "Ninja"]);
    }
    cfg.args([
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_MTMD=ON",
        "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DMTMD_VIDEO=OFF",
        "-DGGML_OPENMP=OFF",
    ]);
    if target_os == "macos" {
        cfg.args([
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0",
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_NATIVE=OFF",
        ]);
    } else {
        cfg.args(["-DGGML_METAL=OFF", "-DGGML_NATIVE=ON"]);
    }
    run(&mut cfg, "cmake configure");
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .to_string();
    run(
        Command::new("cmake").arg("--build").arg(&build).args([
            "--config", "Release", "-j", &jobs, "--target", "llama", "mtmd",
        ]),
        "cmake build",
    );

    // 3. Link: every static library the build produced, in dependency order.
    let mut libs: Vec<(PathBuf, String)> = Vec::new();
    for entry in walkdir(&build) {
        let name = entry.file_name().unwrap().to_string_lossy().into_owned();
        if let Some(stem) = name.strip_prefix("lib").and_then(|n| n.strip_suffix(".a")) {
            libs.push((entry.parent().unwrap().to_path_buf(), stem.to_string()));
        }
    }
    let order = |n: &str| -> i32 {
        match n {
            "mtmd" => 0,
            "llama" => 1,
            "vendor-hash" => 2,
            "ggml" => 3,
            "ggml-base" => 9,
            _ => 5, // backends: ggml-cpu, ggml-metal, ggml-blas
        }
    };
    libs.sort_by_key(|(_, n)| order(n));
    let mut seen = std::collections::HashSet::new();
    for (dir, name) in &libs {
        println!("cargo:rustc-link-search=native={}", dir.display());
        if seen.insert(name.clone()) {
            println!("cargo:rustc-link-lib=static={name}");
        }
    }
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
        for fw in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // 4. Bindings.
    let include_dirs = [
        src.join("include"),
        src.join("ggml/include"),
        src.join("tools/mtmd"),
    ];
    let mut b = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("llama_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_function("ggml_backend_load_all.*")
        .allowlist_type("llama_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("MTMD_.*")
        .prepend_enum_name(false)
        .derive_default(true)
        .layout_tests(false);
    for d in &include_dirs {
        b = b.clang_arg(format!("-I{}", d.display()));
    }
    let bindings = b
        .generate()
        .expect("bindgen (needs libclang; set LIBCLANG_PATH, e.g. /Library/Developer/CommandLineTools/usr/lib on macOS)");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("write bindings.rs");
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walkdir(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}
