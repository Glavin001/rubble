//! Builds `batch/slang_batch.cpp` (several entries of one Slang source in one process,
//! `compile_many`) against the Slang SDK beside slangc, when there is one. Without it
//! (no SDK headers or library, a Windows host, `RUBBLE_SLANG_BATCH=0`) every compile is
//! a slangc of its own, as before: the outputs are the same either way.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=batch/slang_batch.cpp");
    for v in ["SLANG_DIR", "SLANGC", "RUBBLE_SLANG_BATCH", "PATH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let built = build().unwrap_or_default();
    println!("cargo:rustc-env=ENGINE_SLANG_BATCH={built}");
}

fn build() -> Option<String> {
    if std::env::var("RUBBLE_SLANG_BATCH").is_ok_and(|v| v == "0") {
        return None;
    }
    if std::env::var("CARGO_CFG_WINDOWS").is_ok() {
        return None;
    }
    let sdk = sdk_dir()?;
    let include = sdk.join("include");
    let lib = sdk.join("lib");
    if !include.join("slang.h").is_file() {
        return None;
    }
    // The compiler library's name: `slang-compiler` (2026 releases), `slang` before.
    let name = ["slang-compiler", "slang"].into_iter().find(|n| {
        ["dylib", "so"]
            .iter()
            .any(|ext| lib.join(format!("lib{n}.{ext}")).is_file())
    })?;
    let out = PathBuf::from(std::env::var_os("OUT_DIR")?).join("slang_batch");
    let compiler = cc::Build::new()
        .cpp(true)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .try_get_compiler()
        .ok()?;
    let status = compiler
        .to_command()
        .args(["-std=c++17", "-O2"])
        .arg("-I")
        .arg(&include)
        .arg("batch/slang_batch.cpp")
        .arg("-L")
        .arg(&lib)
        .arg(format!("-l{name}"))
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .arg("-o")
        .arg(&out)
        .status()
        .ok()?;
    if !status.success() {
        println!("cargo:warning=slang_batch did not build (every kernel compiles with slangc alone)");
        return None;
    }
    Some(out.display().to_string())
}

/// The Slang SDK: `SLANG_DIR`, else the directory above a slangc's `bin` that holds
/// `include/slang.h` (slangc as engine-slang-build finds it: `SLANGC`, `PATH`,
/// `~/.local`; a `SLANGC` that is a wrapper script has no SDK beside it, so the next
/// one is tried). The batch tool is used only when its Slang is the slangc's in use.
fn sdk_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("SLANG_DIR") {
        return Some(PathBuf::from(d));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("SLANGC") {
        candidates.push(PathBuf::from(p));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join("slangc")));
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        candidates.push(home.join(".local/bin/slangc"));
        candidates.push(home.join(".local/opt/slang-2026.19/bin/slangc"));
    }
    candidates
        .into_iter()
        .filter(|p| p.is_file())
        .filter_map(|p| p.canonicalize().ok())
        .filter_map(|real| real.parent().and_then(Path::parent).map(Path::to_path_buf))
        .find(|sdk| sdk.join("include/slang.h").is_file())
}
