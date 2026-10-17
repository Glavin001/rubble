//! Offline `slangc` driver for build scripts and host tools.

use anyhow::{anyhow, bail, Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlangTarget {
    Metal,
    Spirv,
    Cuda,
    Wgsl,
}

impl SlangTarget {
    pub fn flag(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Spirv => "spirv",
            Self::Cuda => "cuda",
            Self::Wgsl => "wgsl",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Spirv => "spv",
            Self::Cuda => "cu",
            Self::Wgsl => "wgsl",
        }
    }
}

pub fn find_slangc() -> Result<PathBuf> {
    if let Ok(p) = env::var("SLANGC") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Ok(path) = which("slangc") {
        return Ok(path);
    }
    let home = env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let candidates = [
        home.join(".local/bin/slangc"),
        home.join(".local/opt/slang-2026.19/bin/slangc"),
        PathBuf::from("/opt/homebrew/bin/slangc"),
    ];
    for c in candidates {
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!("slangc not found; install 2026.19 and set SLANGC or PATH (see engine/TOOLCHAIN.md)")
}

fn which(bin: &str) -> Result<PathBuf> {
    let out = Command::new("which").arg(bin).output()?;
    if !out.status.success() {
        bail!("which {bin} failed");
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        bail!("empty which");
    }
    Ok(PathBuf::from(s))
}

pub fn compile(
    source: &Path,
    entry: &str,
    target: SlangTarget,
    out: &Path,
    include_dirs: &[&Path],
) -> Result<()> {
    compile_with_defines(source, entry, target, out, include_dirs, &[])
}

/// Like [`compile`], passing `-D NAME=VALUE` for each define (build-time target switches).
pub fn compile_with_defines(
    source: &Path,
    entry: &str,
    target: SlangTarget,
    out: &Path,
    include_dirs: &[&Path],
    defines: &[(&str, &str)],
) -> Result<()> {
    let slangc = find_slangc()?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut cmd = Command::new(&slangc);
    cmd.arg("-target")
        .arg(target.flag())
        .arg("-stage")
        .arg("compute")
        .arg("-entry")
        .arg(entry)
        .arg("-o")
        .arg(out);
    for dir in include_dirs {
        cmd.arg("-I").arg(dir);
    }
    for (name, value) in defines {
        cmd.arg(format!("-D{name}={value}"));
    }
    cmd.arg(source);
    let output = cmd
        .output()
        .with_context(|| format!("failed to spawn {}", slangc.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "slangc failed for {} entry {} target {}:\n{}",
            source.display(),
            entry,
            target.flag(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

/// Compile every `(source_rel, entry)` pair into `out_dir/<entry>.<ext>`.
pub fn compile_entries(
    slang_root: &Path,
    entries: &[(&str, &str)],
    target: SlangTarget,
    out_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let physics = slang_root.join("physics");
    let kernels = slang_root.join("kernels");
    let includes = [slang_root, physics.as_path(), kernels.as_path()];
    let mut outs = Vec::new();
    for (rel, entry) in entries {
        let source = slang_root.join(rel);
        let out = out_dir.join(format!("{entry}.{}", target.extension()));
        compile(&source, entry, target, &out, &includes)?;
        println!("cargo:rerun-if-changed={}", source.display());
        outs.push(out);
    }
    Ok(outs)
}

pub fn report_slangc_for_cargo() {
    if let Ok(p) = find_slangc() {
        println!("cargo:rustc-env=ENGINE_SLANGC={}", p.display());
        println!("cargo:warning=using slangc at {}", p.display());
    }
}
