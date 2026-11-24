use anyhow::{Context, Result};
use engine_slang_build::{compile, compile_entries_to, compile_many, compile_with_defines, find_slangc, par_map, Compile, SlangTarget};
use std::env;
use std::path::{Path, PathBuf};

include!("src/wgsl_loop_cap.rs");
include!("src/explicit_post.rs");
include!("src/kernel_build.rs");

fn main() -> Result<()> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let slang_root = manifest.join("../../slang");
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    println!("cargo::rustc-check-cfg=cfg(kernels_at_runtime)");
    // The compile code's key (the runtime store's directories are keyed by it too).
    let mut code = vec![
        manifest.join("src/kernel_build.rs"),
        manifest.join("src/explicit_post.rs"),
        manifest.join("src/wgsl_loop_cap.rs"),
    ];
    list_files(&manifest.join("../engine-kernel-manifest/src"), &mut code)?;
    list_files(&manifest.join("../engine-slang-build/src"), &mut code)?;
    code.sort();
    {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for f in &code {
            println!("cargo:rerun-if-changed={}", f.display());
            std::fs::read(f)?.hash(&mut h);
        }
        println!("cargo:rustc-env=KERNEL_BUILD_KEY={:016x}", h.finish());
    }
    // The kernels at run time: the fast profile on a native target (the iteration loop:
    // fast-check, the scenario matrix, tests run with --profile fast), unless
    // RUBBLE_KERNELS=embed. They are compiled on first use into a store keyed by the
    // Slang sources (src/kernel_store.rs), so this script does not watch them and a
    // kernel edit compiles no Rust. Everything else embeds them (release, the C library,
    // wasm): the same compile, the same bytes.
    println!("cargo:rerun-if-env-changed=RUBBLE_KERNELS");
    let fast = out_dir.components().any(|c| c.as_os_str() == "fast");
    let wasm = env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "wasm32");
    // (On wasm only when asked, RUBBLE_KERNELS=runtime: the fast web build's world wasm,
    // whose page module installs the store's reader; see kernel_store::set_loader.)
    let runtime = match env::var("RUBBLE_KERNELS").as_deref() {
        Ok("embed") => false,
        Ok("runtime") => true,
        _ => fast && !wasm,
    };
    if runtime {
        println!("cargo:rustc-cfg=kernels_at_runtime");
        std::fs::write(out_dir.join("direct_wgsl.rs"), "")?;
        return Ok(());
    }
    engine_slang_build::report_slangc_for_cargo();
    println!("cargo:rerun-if-changed={}", slang_root.display());
    // (Not `rerun-if-env-changed=RUBBLE_SLANG_CACHE`: the outputs are the same with or
    // without that cache, so switching between build-web.sh, which sets it, and plain
    // cargo must not rerun this script and recompile every crate above this one.)
    println!("cargo:rerun-if-env-changed=RUBBLE_EB_REF");
    println!("cargo:rerun-if-env-changed=RUBBLE_EB_SRC");
    println!("cargo:rerun-if-env-changed=RUBBLE_EB_REF_UNCAPPED");
    if let Ok(f) = std::env::var("RUBBLE_EB_SRC") {
        println!("cargo:rerun-if-changed={f}");
    }
    let _ = find_slangc()?;

    // `RUBBLE_SLANG_CACHE`: a directory keeping these outputs between builds, keyed by
    // everything they depend on (see `cache_key`). The SPIR-V and WGSL do not depend on
    // the cargo target, so the host and wasm32 builds share an entry.
    let cache = match env::var_os("RUBBLE_SLANG_CACHE") {
        Some(dir) => Some(PathBuf::from(dir).join(cache_key(&manifest, &slang_root)?)),
        None => None,
    };
    let compiled = match cache.as_deref().filter(|d| d.join("wgsl_ok.txt").is_file()) {
        Some(hit) => {
            for sub in ["spirv", "wgsl"] {
                copy_dir(&hit.join(sub), &out_dir.join(sub))?;
            }
            std::fs::copy(
                hit.join("spirv_entries.txt"),
                out_dir.join("spirv_entries.txt"),
            )?;
            Compiled {
                warnings: read_records(&hit.join("warnings.txt"))?,
                wgsl_ok: read_records(&hit.join("wgsl_ok.txt"))?,
            }
        }
        None => {
            let compiled = compile_all(&slang_root, &out_dir)?;
            if let Some(dir) = &cache {
                store(dir, &out_dir, &compiled)?;
            }
            compiled
        }
    };
    for w in &compiled.warnings {
        println!("cargo:warning={w}");
    }
    let wgsl_dir = out_dir.join("wgsl");
    let mut table = String::from("pub static DIRECT_WGSL: &[(&str, &str)] = &[\n");
    for entry in &compiled.wgsl_ok {
        table.push_str(&format!(
            "    ({entry:?}, include_str!({:?})),\n",
            wgsl_dir.join(format!("{entry}.wgsl")).display().to_string()
        ));
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("direct_wgsl.rs"), table)?;
    Ok(())
}

/// A hash of everything the outputs depend on: slangc's version, every file under
/// `slang/` (path and bytes), this script and engine-slang-build (which pass the
/// arguments). Equal keys mean slangc gets identical inputs, so a hit is the same bytes.
fn cache_key(manifest: &Path, slang_root: &Path) -> Result<String> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    engine_slang_build::slangc_version()?.hash(&mut h);
    let mut files = Vec::new();
    list_files(slang_root, &mut files)?;
    files.sort();
    for f in &files {
        (f.strip_prefix(slang_root)?, std::fs::read(f)?).hash(&mut h);
    }
    // The kernel lists (engine-kernel-manifest: which modules, entries and defines).
    let mut lists = Vec::new();
    list_files(&manifest.join("../engine-kernel-manifest/src"), &mut lists)?;
    lists.sort();
    for f in ["build.rs", "src/wgsl_loop_cap.rs", "../engine-slang-build/src/lib.rs"]
        .iter()
        .map(|f| manifest.join(f))
        .chain(lists)
    {
        std::fs::read(f)?.hash(&mut h);
    }
    Ok(format!("engine-webgpu-{:016x}", h.finish()))
}

fn list_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            list_files(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        std::fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

/// Records are separated by U+001E, since a warning can span lines.
fn read_records(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .split('\u{1e}')
        .filter(|r| !r.is_empty())
        .map(String::from)
        .collect())
}

/// Write the entry under a temporary name and rename it, so a build that stops
/// half way never leaves an entry that looks complete.
fn store(dir: &Path, out_dir: &Path, compiled: &Compiled) -> Result<()> {
    let tmp = dir.with_extension(format!("tmp{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    for sub in ["spirv", "wgsl"] {
        copy_dir(&out_dir.join(sub), &tmp.join(sub))?;
    }
    std::fs::copy(
        out_dir.join("spirv_entries.txt"),
        tmp.join("spirv_entries.txt"),
    )?;
    std::fs::write(tmp.join("warnings.txt"), compiled.warnings.join("\u{1e}"))?;
    std::fs::write(tmp.join("wgsl_ok.txt"), compiled.wgsl_ok.join("\u{1e}"))?;
    if std::fs::rename(&tmp, dir).is_err() {
        // Another build stored the same key first.
        std::fs::remove_dir_all(&tmp)?;
    }
    // Keep only this key: entries for older slang/ trees would only grow the cache.
    if let (Some(parent), Some(name)) = (dir.parent(), dir.file_name()) {
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let other = entry.file_name();
            if other != name && other.to_string_lossy().starts_with("engine-webgpu-") {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    Ok(())
}
