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

/// `nvcc` for Slang-emitted CUDA → PTX: `NVCC`, `PATH`, then the toolkit's
/// default prefix. Only build hosts need it; the PTX runs on any driver.
pub fn find_nvcc() -> Result<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Ok(path) = which("nvcc") {
        return Ok(path);
    }
    let fallback = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if fallback.is_file() {
        return Ok(fallback);
    }
    bail!("nvcc not found; install the CUDA toolkit's nvcc and set NVCC or PATH (see engine/TOOLCHAIN.md)")
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

/// `f` over `items` on every core, results in input order. Build scripts compile
/// hundreds of entries at about a second of `slangc` each; one at a time that is minutes.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len().max(1));
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i) else { break };
                let r = f(item);
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect()
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
    pub fn reflected_threads(json: &str, entry: &str) -> Result<[u32; 3]> {
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

/// Slang → CUDA C++ → PTX for the shared rigid plan's kernels, in the
/// browser's `ENGINE_WGSL` buffer layout. Slang gathers every global a CUDA
/// entry touches into one `__constant__ SLANG_globalParams` struct; the
/// reflection gives each field's byte offset, which the host fills with device
/// pointers from the shared `BIND(slot)` numbering.
pub mod cuda_layout {
    use anyhow::{anyhow, bail, Context, Result};
    use std::path::Path;

    /// One field of `SLANG_globalParams`: a structured buffer (16 bytes:
    /// pointer, element count) or a uniform block pointer (8 bytes).
    #[derive(Clone, Debug)]
    pub struct GlobalField {
        pub name: String,
        pub offset: u32,
        pub size: u32,
    }

    #[derive(Clone, Debug)]
    pub struct CudaKernel {
        pub threads: [u32; 3],
        pub fields: Vec<GlobalField>,
    }

    /// Compile one compute entry to CUDA C++ (`-DENGINE_WGSL=1`) with reflection.
    pub fn compile_kernel(
        slangc: &Path,
        root: &Path,
        source: &Path,
        entry: &str,
        dest: &Path,
    ) -> Result<CudaKernel> {
        let reflection = dest.with_extension("json");
        let output = std::process::Command::new(slangc)
            .args([
                "-target",
                "cuda",
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
        Ok(CudaKernel {
            threads: super::msl_layout::reflected_threads(&json, entry)?,
            fields: global_fields(&json, entry)?,
        })
    }

    /// Uniform-bound globals (`binding.kind == "uniform"`) with offset and size.
    pub fn global_fields(json: &str, entry: &str) -> Result<Vec<GlobalField>> {
        let v: serde_json::Value =
            serde_json::from_str(json).with_context(|| format!("{entry}: reflection json"))?;
        let params = v["parameters"]
            .as_array()
            .ok_or_else(|| anyhow!("{entry}: reflection has no parameters"))?;
        let mut out = Vec::new();
        for p in params {
            let b = &p["binding"];
            if b["kind"] != "uniform" {
                bail!(
                    "{entry}: global {} is bound as {}, not uniform",
                    p["name"],
                    b["kind"]
                );
            }
            let num = |k: &str| {
                b[k].as_u64()
                    .map(|x| x as u32)
                    .ok_or_else(|| anyhow!("{entry}: global {} has no {k}", p["name"]))
            };
            let size = num("size")?;
            if size != 8 && size != 16 {
                bail!(
                    "{entry}: global {} is {size} bytes (expected a pointer or buffer)",
                    p["name"]
                );
            }
            out.push(GlobalField {
                name: p["name"]
                    .as_str()
                    .ok_or_else(|| anyhow!("{entry}: unnamed global"))?
                    .to_string(),
                offset: num("offset")?,
                size,
            });
        }
        Ok(out)
    }

    /// Slang's CUDA emitter can keep `uint v = gBuf[i];` as a pointer
    /// (`T * _Sk = (&(globalParams_0->gBuf_0)[i]);`) and dereference it only
    /// where `v` is used. If the kernel stores to that buffer in between, the
    /// load sees the new value: a silent miscompile Metal and WGSL do not have
    /// (it broke prefix_color_tiles_parallel). Lists every such read.
    pub fn deferred_load_hazards(cuda: &str) -> Vec<String> {
        // pointer temp -> (buffer, a store to the buffer happened since)
        let mut live: Vec<(String, String, bool)> = Vec::new();
        let mut out = Vec::new();
        let buffer_in = |text: &str| -> Option<String> {
            let at = text.find("globalParams_0->")? + "globalParams_0->".len();
            let name: String = text[at..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            Some(name)
        };
        let ptr_named = |text: &str| -> Option<String> {
            let at = text.find("_S")?;
            let name: String = text[at..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (name.len() > 2).then_some(name)
        };
        for (n, line) in cuda.lines().enumerate() {
            if line.starts_with('}') {
                live.clear();
                continue;
            }
            let t = line.trim();
            // `T * _Sk = (&(globalParams_0->gBuf_0)[...]);`
            if let Some((lhs, rhs)) = t.split_once(" = ") {
                if lhs.contains("* _S") && rhs.starts_with("(&(globalParams_0->") {
                    if let (Some(p), Some(b)) = (ptr_named(lhs), buffer_in(rhs)) {
                        live.retain(|(q, _, _)| *q != p);
                        live.push((p, b, false));
                    }
                    continue;
                }
            }
            // Stores: an assignment whose target names a buffer or a pointer
            // temp, or any atomic on a buffer.
            let mut stored: Option<(String, Option<String>)> = None;
            if t.contains("atomic") {
                stored = buffer_in(t).map(|b| (b, None));
            }
            if stored.is_none() {
                if let Some((lhs, _)) = t.split_once(" = ") {
                    if !lhs.contains(' ') || lhs.starts_with('*') || lhs.starts_with('(') {
                        if let Some(b) = buffer_in(lhs) {
                            stored = Some((b, None));
                        } else if let Some(p) = ptr_named(lhs) {
                            if let Some((_, b, _)) = live.iter().find(|(q, _, _)| *q == p) {
                                stored = Some((b.clone(), Some(p)));
                            }
                        }
                    }
                }
            }
            if let Some((buffer, via)) = stored {
                for (p, b, dirty) in live.iter_mut() {
                    if *b == buffer && via.as_deref() != Some(p.as_str()) {
                        *dirty = true;
                    }
                }
                continue;
            }
            for (p, b, dirty) in live.iter_mut() {
                if *dirty && (t.contains(&format!("*{p}")) || t.contains(&format!("{p}->"))) {
                    out.push(format!(
                        "line {}: {p} ({b}) read after a store to {b}",
                        n + 1
                    ));
                    *dirty = false;
                }
            }
        }
        out
    }

    /// CUDA C++ → fatbin: SASS for each `sm` in `sass` (no JIT on those GPUs,
    /// so a single-use GPU container does not spend its seconds compiling)
    /// plus PTX for `ptx_arch` (`compute_75`), which the driver JIT-compiles on
    /// any other GPU.
    pub fn fatbin(
        nvcc: &Path,
        cu: &Path,
        dest: &Path,
        ptx_arch: &str,
        sass: &[&str],
    ) -> Result<()> {
        let mut cmd = std::process::Command::new(nvcc);
        cmd.args(["-fatbin", "-std=c++17", "-lineinfo"]);
        // 20044: Slang's `extern "C" __constant__ SLANG_globalParams` becomes a
        // module definition in whole-program mode, as the host expects.
        cmd.args(["-diag-suppress", "20044"]);
        // Slang lowers `Atomic<uint>.load()/.store()` to plain global accesses on
        // CUDA. Plain loads are cached per SM in L1, which atomics (done in L2)
        // do not update, so a kernel could read a counter or hash slot another
        // block changed in the same launch. Default: cache global loads in L2
        // only (`ld.cg`, device-coherent). ENGINE_CUDA_LOAD_CACHE=ca restores
        // L1 caching (the CUDA default) for measurements.
        let cache = std::env::var("ENGINE_CUDA_LOAD_CACHE").unwrap_or_else(|_| "cg".into());
        cmd.args(["-Xptxas", &format!("-dlcm={cache}")]);
        for sm in sass {
            cmd.arg(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
        }
        cmd.arg(format!("-gencode=arch={ptx_arch},code={ptx_arch}"));
        let output = cmd
            .arg("-o")
            .arg(dest)
            .arg(cu)
            .output()
            .with_context(|| format!("spawn {}", nvcc.display()))?;
        if !output.status.success() {
            bail!(
                "nvcc {}: {}",
                cu.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::deferred_load_hazards;

        /// Slang 2026.19's output for prefix_color_tiles_parallel before the fix.
        #[test]
        fn flags_read_after_store_through_deferred_pointer() {
            let miscompiled = "extern \"C\" __global__ void k()\n{\n\
                \x20       uint index_0 = color_0 * groups_0 + group_0;\n\
                \x20       uint * _S5 = (&(globalParams_0->gColorOffsets_0)[index_0]);\n\
                \x20       *(&(globalParams_0->gColorOffsets_0)[index_0]) = add_0;\n\
                \x20       uint prefix_0 = add_0 + *_S5;\n}\n";
            assert_eq!(deferred_load_hazards(miscompiled).len(), 1);
        }

        #[test]
        fn accepts_load_before_store() {
            let fixed = "extern \"C\" __global__ void k()\n{\n\
                \x20       uint next_0 = add_0 + *(&(globalParams_0->gColorOffsets_0)[index_0]);\n\
                \x20       *(&(globalParams_0->gColorOffsets_0)[index_0]) = add_0;\n\
                \x20       BodyState_0 * _S5 = (&(globalParams_0->gBodies_0)[body_1]);\n\
                \x20       *_S5 = b_3;\n\
                \x20       float y = _S5->position_0.y;\n}\n";
            assert!(deferred_load_hazards(fixed).is_empty());
        }
    }
}
