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
    compile_stage(source, entry, "compute", target, out, include_dirs, defines)
}

/// Like [`compile_with_defines`] for any pipeline stage (`compute`, `vertex`, `fragment`).
pub fn compile_stage(
    source: &Path,
    entry: &str,
    stage: &str,
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
        .arg(stage)
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

/// Slang → MSL with the browser's `ENGINE_WGSL` buffer layout, so one host
/// buffer set and one command list serve WebGPU and native Metal. Slang assigns
/// Metal `[[buffer(n)]]` per entry; [`msl_layout::msl_buffers`] maps each back
/// to the shared `BIND(slot)` numbering from [`msl_layout::bind_slots`].
pub mod msl_layout {
    use anyhow::{anyhow, bail, Context, Result};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    /// `BIND(n) <type> gName;` → slot n, from the kernel sources.
    pub fn bind_slots(sources: &[PathBuf]) -> Result<HashMap<String, u32>> {
        let mut map = HashMap::new();
        for path in sources {
            for line in std::fs::read_to_string(path)?.lines() {
                let Some(rest) = line.trim_start().strip_prefix("BIND(") else {
                    continue;
                };
                let Some((n, decl)) = rest.split_once(')') else {
                    continue;
                };
                let Some(decl) = decl.split(';').next() else {
                    continue;
                };
                let Some(name) = decl.split_whitespace().last() else {
                    continue;
                };
                let slot: u32 = n.trim().parse()?;
                if let Some(old) = map.insert(name.to_string(), slot) {
                    if old != slot {
                        bail!("{name} is bound to both {old} and {slot}");
                    }
                }
            }
        }
        Ok(map)
    }

    /// `(global name, Metal buffer index)` pairs from the `[[kernel]]` / stage signature.
    pub fn msl_buffers(msl: &str, entry: &str, stage: &str) -> Result<Vec<(String, u32)>> {
        let sig = msl
            .lines()
            .find(|l| l.starts_with(stage) && l.contains(&format!(" {entry}(")))
            .ok_or_else(|| anyhow!("no {stage} signature for {entry}"))?;
        let mut out = Vec::new();
        for part in sig.split(", ") {
            let Some((decl, rest)) = part.split_once(" [[buffer(") else {
                continue;
            };
            let index: u32 = rest.split(')').next().unwrap().parse()?;
            let var = decl.split_whitespace().last().unwrap();
            // Slang suffixes globals with `_<n>`.
            let name = var.rsplit_once('_').map_or(var, |(base, _)| base);
            out.push((name.to_string(), index));
        }
        Ok(out)
    }

    /// Compile one compute entry to MSL (`-DENGINE_WGSL=1`) with reflection;
    /// returns (msl, thread-group size).
    pub fn compile_kernel(
        slangc: &Path,
        root: &Path,
        source: &Path,
        entry: &str,
        dest: &Path,
    ) -> Result<(String, [u32; 3])> {
        let reflection = dest.with_extension("json");
        let output = std::process::Command::new(slangc)
            .args([
                "-target",
                "metal",
                "-stage",
                "compute",
                "-entry",
                entry,
                "-DENGINE_WGSL=1",
                "-I",
            ])
            .arg(root.join("physics"))
            .arg("-I")
            .arg(root.join("kernels"))
            .arg("-o")
            .arg(dest)
            .arg("-reflection-json")
            .arg(&reflection)
            .arg(source)
            .output()
            .with_context(|| format!("spawn {}", slangc.display()))?;
        if !output.status.success() {
            bail!(
                "slangc {entry}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let json = std::fs::read_to_string(&reflection)?;
        Ok((
            std::fs::read_to_string(dest)?,
            reflected_threads(&json, entry)?,
        ))
    }

    /// `threadGroupSize` of `entry` in Slang's reflection JSON.
    fn reflected_threads(json: &str, entry: &str) -> Result<[u32; 3]> {
        let missing = || anyhow!("{entry}: no reflected thread group size");
        let at = json
            .find(&format!("\"name\": \"{entry}\""))
            .ok_or_else(missing)?;
        let rest = &json[at..];
        let key = rest.find("\"threadGroupSize\"").ok_or_else(missing)?;
        let open = rest[key..].find('[').ok_or_else(missing)? + key;
        let close = rest[open..].find(']').ok_or_else(missing)? + open;
        let v: Vec<u32> = rest[open + 1..close]
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<std::result::Result<_, _>>()?;
        match v[..] {
            [x, y, z] => Ok([x, y, z]),
            _ => bail!("{entry}: thread group size {v:?}"),
        }
    }
}
