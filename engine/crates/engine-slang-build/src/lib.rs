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

/// slangc's version report (`slangc -version`, both streams), for keys of caches.
pub fn slangc_version() -> Result<String> {
    let out = Command::new(find_slangc()?).arg("-version").output()?;
    Ok(format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
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

/// Worker threads for a build script's parallel compiles: every core, or at most
/// `RUBBLE_BUILD_JOBS` when set (a small CI runner can run out of memory with one
/// `nvcc` per core on the large kernels). Only the build's speed depends on it.
pub fn build_jobs() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    match std::env::var("RUBBLE_BUILD_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(n) if n > 0 => n.min(cores),
        _ => cores,
    }
}

/// `f` over `items` on [`build_jobs`] threads, results in input order. Build scripts
/// compile hundreds of entries at about a second of `slangc` each; one at a time that
/// is minutes.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let threads = build_jobs().min(items.len().max(1));
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

mod batch;
pub use batch::{compile_many, Compile};

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
    compile_entries_to(source, &[(entry, stage)], target, out, include_dirs, defines)
}

/// Several entries of one source in one output module (one slangc run, one cache
/// entry): the standalone WGSL modules with more than one entry point.
pub fn compile_entries_to(
    source: &Path,
    entries: &[(&str, &str)],
    target: SlangTarget,
    out: &Path,
    include_dirs: &[&Path],
    defines: &[(&str, &str)],
) -> Result<()> {
    Slangc {
        source,
        entries,
        target: target.flag(),
        include_dirs,
        defines,
        out,
        out_ext: "",
        reflection: None,
        base: None,
    }
    .run()
}

/// One slangc run and its place in the per-compile cache. Every compile in the build
/// goes through here, so every one is cached by what it depends on ([`compile_key`]).
///
/// slangc runs in the Slang tree's root (the nearest `slang` directory above the
/// source) and is given paths relative to it, so its output (the `#line` directives of
/// the MSL and CUDA targets name the files) is the same in every checkout and from
/// every crate's build script: one cache serves them all.
pub(crate) struct Slangc<'a> {
    pub(crate) source: &'a Path,
    /// `(entry, stage)`.
    pub(crate) entries: &'a [(&'a str, &'a str)],
    pub(crate) target: &'a str,
    pub(crate) include_dirs: &'a [&'a Path],
    pub(crate) defines: &'a [(&'a str, &'a str)],
    pub(crate) out: &'a Path,
    /// The output's extension in the cache (`""`: the key alone).
    pub(crate) out_ext: &'a str,
    /// Slang's reflection JSON, written beside the output and cached with it.
    pub(crate) reflection: Option<&'a Path>,
    /// The directory slangc runs in and is given paths relative to. None: the Slang
    /// tree's root above the source ([`slang_base`]). Sources outside a `slang`
    /// directory (the stress solver's shaders) name theirs, so their output is the
    /// same in every checkout too.
    pub(crate) base: Option<&'a Path>,
}

impl Slangc<'_> {
    fn base_dir(&self) -> Option<PathBuf> {
        match self.base {
            Some(b) => b.canonicalize().ok(),
            None => slang_base(self.source),
        }
    }

    pub(crate) fn run(&self) -> Result<()> {
        let slangc = find_slangc()?;
        if let Some(parent) = self.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let cached = self.cache_entry(&slangc);
        if self.restore(cached.as_deref())? {
            return Ok(());
        }
        self.spawn(&slangc)?;
        self.store(cached.as_deref());
        Ok(())
    }

    /// slangc itself, for this compile (no cache).
    pub(crate) fn spawn(&self, slangc: &Path) -> Result<()> {
        let base = self.base_dir();
        let mut cmd = Command::new(slangc);
        if let Some(b) = &base {
            cmd.current_dir(b);
        }
        cmd.arg("-target").arg(self.target);
        for (entry, stage) in self.entries {
            cmd.arg("-entry").arg(entry).arg("-stage").arg(stage);
        }
        cmd.arg("-o").arg(self.out);
        if let Some(r) = self.reflection {
            cmd.arg("-reflection-json").arg(r);
        }
        for dir in self.include_dirs {
            cmd.arg("-I").arg(rel_to(base.as_deref(), dir));
        }
        for (name, value) in self.defines {
            cmd.arg(format!("-D{name}={value}"));
        }
        cmd.arg(rel_to(base.as_deref(), self.source));
        let output = cmd
            .output()
            .with_context(|| format!("failed to spawn {}", slangc.display()))?;
        if !output.status.success() {
            let names: Vec<&str> = self.entries.iter().map(|(e, _)| *e).collect();
            return Err(anyhow!(
                "slangc failed for {} entry {} target {}:\n{}",
                self.source.display(),
                names.join(","),
                self.target,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    }

    /// The compile's place in the per-compile cache (None: not cached).
    pub(crate) fn cache_entry(&self, slangc: &Path) -> Option<PathBuf> {
        let base = self.base_dir();
        let mut kind = self.target.to_string();
        if self.reflection.is_some() {
            kind.push_str("+reflection-json");
        }
        let key = compile_key(slangc, self.source, base.as_deref(), self.entries, &kind, self.include_dirs, self.defines)?;
        compile_cache_dir().map(|d| d.join(key))
    }

    fn outputs(&self) -> Vec<(&Path, &str)> {
        let mut outputs: Vec<(&Path, &str)> = vec![(self.out, self.out_ext)];
        if let Some(r) = self.reflection {
            outputs.push((r, "json"));
        }
        outputs
    }

    /// Copies the outputs from the cache when every one of them is there.
    pub(crate) fn restore(&self, cached: Option<&Path>) -> Result<bool> {
        let Some(cached) = cached else { return Ok(false) };
        let outputs = self.outputs();
        if !outputs.iter().all(|(_, ext)| kept(cached, ext).is_file()) {
            return Ok(false);
        }
        // Restore the reflection first: a reader takes the module as the sign of a
        // finished compile.
        for (dest, ext) in outputs.iter().rev() {
            let hit = kept(cached, ext);
            std::fs::copy(&hit, dest)?;
            // (A hit is marked used: the cache is trimmed least recently used first.)
            let _ = std::fs::File::options()
                .append(true)
                .open(&hit)
                .and_then(|f| f.set_modified(std::time::SystemTime::now()));
        }
        Ok(true)
    }

    /// Keeps the outputs in the cache: written beside their names and renamed, the
    /// module last (a hit needs every output), as another build may store the same key.
    pub(crate) fn store(&self, cached: Option<&Path>) {
        let Some(cached) = cached else { return };
        if let Some(dir) = cached.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        for (from, ext) in self.outputs().iter().rev() {
            let to = kept(cached, ext);
            let tmp = to.with_extension(format!("tmp{}-{}", std::process::id(), unique()));
            if std::fs::copy(from, &tmp).is_ok() {
                let _ = std::fs::rename(&tmp, &to);
            }
        }
    }
}

/// A cache entry's file for one output (`""`: the key alone).
fn kept(cached: &Path, ext: &str) -> PathBuf {
    if ext.is_empty() {
        cached.to_path_buf()
    } else {
        cached.with_extension(ext)
    }
}

/// `p` relative to the Slang root `base` (as slangc is given it), else as it is.
pub(crate) fn rel_to(base: Option<&Path>, p: &Path) -> PathBuf {
    let full = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    match base {
        Some(b) => full.strip_prefix(b).map(Path::to_path_buf).unwrap_or(full),
        None => full,
    }
}

/// A number no other compile of this process uses at the same time (temporary names).
pub(crate) fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// The Slang tree's root above `source`: the nearest ancestor directory named `slang`.
pub(crate) fn slang_base(source: &Path) -> Option<PathBuf> {
    let full = source.canonicalize().ok()?;
    full.ancestors()
        .skip(1)
        .find(|a| a.file_name().is_some_and(|n| n == "slang"))
        .map(Path::to_path_buf)
}

/// Where single compiles are kept: `RUBBLE_SLANG_OBJ_CACHE` (a directory; `0` or empty:
/// no cache), else `~/.cache/rubble/slang-obj`, one cache for every checkout and
/// worktree (its entries do not depend on where a checkout is), else `slang-obj-cache`
/// in the cargo target directory this build script runs for.
fn compile_cache_dir() -> Option<PathBuf> {
    static DIR: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = match std::env::var_os("RUBBLE_SLANG_OBJ_CACHE") {
            Some(v) if v.is_empty() || v == "0" => return None,
            Some(v) => PathBuf::from(v),
            None => match std::env::var_os("HOME").filter(|h| !h.is_empty()) {
                Some(home) => PathBuf::from(home).join(".cache/rubble/slang-obj"),
                // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out (with a
                // target triple, one level deeper: the cache is then that triple's).
                None => {
                    let out = PathBuf::from(std::env::var_os("OUT_DIR")?);
                    out.ancestors().nth(4)?.join("slang-obj-cache")
                }
            },
        };
        trim_cache(&dir);
        Some(dir)
    })
    .clone()
}

/// Keeps the per-compile cache under `RUBBLE_SLANG_OBJ_CACHE_GIB` (default 8): past
/// it, the least recently used entries go until it is at three quarters. Looks at
/// most once an hour (a stamp file in the cache), so a build pays for it rarely.
fn trim_cache(dir: &Path) {
    let stamp = dir.join(".trimmed");
    let fresh = std::fs::metadata(&stamp)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() < 3600);
    if fresh {
        return;
    }
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(&stamp, b"");
    let cap_gib: f64 = std::env::var("RUBBLE_SLANG_OBJ_CACHE_GIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8.0);
    let cap = (cap_gib * (1u64 << 30) as f64) as u64;
    let Ok(read) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(std::time::SystemTime, u64, PathBuf)> = read
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            if !m.is_file() || e.file_name() == ".trimmed" {
                return None;
            }
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    if total <= cap {
        return;
    }
    files.sort();
    for (_, len, path) in files {
        if total <= cap / 4 * 3 {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
}

/// A key of everything one compile's output depends on: slangc itself (path, size,
/// modification time), its arguments, and the names (relative to the Slang root,
/// as slangc is given them) and bytes of the source and of every file it reaches by
/// `#include "..."` or `import name;`. None (no cache for this compile) when a file
/// in that closure uses `__include`, or imports a module that is not beside it or in
/// an include directory (slangc finds those by its own search): such a compile runs
/// every time.
fn compile_key(
    slangc: &Path,
    source: &Path,
    base: Option<&Path>,
    entries: &[(&str, &str)],
    target: &str,
    include_dirs: &[&Path],
    defines: &[(&str, &str)],
) -> Option<String> {
    use std::hash::{Hash, Hasher};
    let rel = |p: &Path| -> PathBuf {
        let full = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        match base {
            Some(b) => full.strip_prefix(b).map(Path::to_path_buf).unwrap_or(full),
            None => full,
        }
    };
    // The files reached, twice: first every `#include` (and every macro name any of
    // them defines), then only those not inside an `#ifdef NAME` whose NAME neither a
    // `-D` nor any of those files defines (such a block is never compiled: the split
    // group path's file, say, in a module built without it). Anything less plain than
    // `#ifdef NAME` keeps its includes.
    let all = include_closure(source, include_dirs, None)?;
    let mut known: std::collections::HashSet<String> = defines.iter().map(|(n, _)| n.to_string()).collect();
    for (_, text) in &all {
        for line in text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("#define") {
                if let Some(name) = rest.split(|c: char| !(c.is_alphanumeric() || c == '_')).find(|t| !t.is_empty()) {
                    known.insert(name.to_string());
                }
            }
        }
    }
    let files = include_closure(source, include_dirs, Some(&known))?;
    // Two independent 64-bit hashes (std's SipHash with two seeds' worth of prefix).
    let hash = |seed: u8| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut h);
        slangc.hash(&mut h);
        if let Ok(m) = std::fs::metadata(slangc) {
            m.len().hash(&mut h);
            m.modified().ok().hash(&mut h);
        }
        (entries, target).hash(&mut h);
        for d in include_dirs {
            rel(d).hash(&mut h);
        }
        defines.hash(&mut h);
        rel(source).hash(&mut h);
        for (path, text) in &files {
            rel(path).hash(&mut h);
            text.hash(&mut h);
        }
        h.finish()
    };
    // Measurement (tools/build-time/fanout.py): with RUBBLE_SLANG_CLOSURE_LOG=FILE, a line
    // a compile: its source, target, entries and defines, then the files of its closure
    // (tab-separated): which compiles an edit of each file invalidates.
    if let Some(log) = std::env::var_os("RUBBLE_SLANG_CLOSURE_LOG") {
        use std::io::Write;
        let mut line = format!("{}\t{}\t{:?}\t{:?}", rel(source).display(), target, entries, defines);
        for (path, _) in &files {
            line.push('\t');
            line.push_str(&rel(path).display().to_string());
        }
        line.push('\n');
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(log) {
            let _ = f.write_all(line.as_bytes());
        }
    }
    Some(format!("{:016x}{:016x}", hash(1), hash(2)))
}

/// The paths and texts of `source` and of every file it reaches by `#include "..."`
/// or `import name;` (None: a file missing, or one that uses `__include`). With `known`, an
/// include inside an `#ifdef NAME` block whose NAME is not in `known` is not followed.
pub(crate) fn include_closure(
    source: &Path,
    include_dirs: &[&Path],
    known: Option<&std::collections::HashSet<String>>,
) -> Option<Vec<(PathBuf, String)>> {
    let mut files = vec![source.canonicalize().ok()?];
    let mut texts = Vec::new();
    let mut at = 0;
    while at < files.len() {
        let file = files[at].clone();
        at += 1;
        let text = std::fs::read_to_string(&file).ok()?;
        // Per open conditional: whether its block may be compiled.
        let mut open: Vec<bool> = Vec::new();
        for line in text.lines() {
            let line = line.trim_start();
            if line.starts_with("__include") || line.starts_with("__import") {
                return None;
            }
            // `import name;`: the module `name.slang` (dots are directories), found as
            // slangc finds it: beside the importing file, then in the include
            // directories. One that is not there (slangc's own search would have to
            // find it) leaves the compile uncached.
            if let Some(rest) = line.strip_prefix("import ") {
                let name = rest.split(';').next()?.trim().replace('.', "/");
                let name = format!("{name}.slang");
                let found = std::iter::once(file.parent()?.join(&name))
                    .chain(include_dirs.iter().map(|d| d.join(&name)))
                    .find(|p| p.is_file())?;
                let found = found.canonicalize().ok()?;
                if !files.contains(&found) {
                    files.push(found);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("#ifdef") {
                let name = rest.split_whitespace().next().unwrap_or("");
                open.push(known.is_none_or(|k| k.contains(name)));
            } else if line.starts_with("#if") {
                open.push(true);
            } else if line.starts_with("#else") || line.starts_with("#elif") {
                if let Some(last) = open.last_mut() {
                    *last = true;
                }
            } else if line.starts_with("#endif") {
                open.pop();
            }
            let Some(rest) = line.strip_prefix("#include") else { continue };
            if open.iter().any(|live| !live) {
                continue;
            }
            let name = rest.trim().strip_prefix('"')?.split('"').next()?;
            // Relative to the including file, then to the include directories.
            let found = std::iter::once(file.parent()?.join(name))
                .chain(include_dirs.iter().map(|d| d.join(name)))
                .find(|p| p.is_file())?;
            let found = found.canonicalize().ok()?;
            if !files.contains(&found) {
                files.push(found);
            }
        }
        texts.push((file, text));
    }
    Some(texts)
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
pub mod msl_layout;

/// Slang → CUDA C++ → PTX for the shared rigid plan's kernels, in the
/// browser's `ENGINE_WGSL` buffer layout. Slang gathers every global a CUDA
/// entry touches into one `__constant__ SLANG_globalParams` struct; the
/// reflection gives each field's byte offset, which the host fills with device
/// pointers from the shared `BIND(slot)` numbering.
pub mod cuda_layout {
    use anyhow::{anyhow, bail, Context, Result};
    use std::path::{Path, PathBuf};

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
        compile_kernel_with_defines(slangc, root, source, entry, dest, &[("ENGINE_WGSL", "1")])
    }

    /// [`compile_kernel`] with `-D NAME=VALUE` for each define (`ENGINE_WGSL=1`
    /// among them for the browser's layout).
    pub fn compile_kernel_with_defines(
        slangc: &Path,
        root: &Path,
        source: &Path,
        entry: &str,
        dest: &Path,
        defines: &[(&str, &str)],
    ) -> Result<CudaKernel> {
        let _ = slangc;
        let include_dirs = [root.join("physics"), root.join("kernels")];
        compile_kernel_at(None, &include_dirs.iter().map(PathBuf::as_path).collect::<Vec<_>>(), source, entry, dest, defines)
    }

    /// [`compile_kernel_with_defines`] for a source outside the engine's Slang tree:
    /// its own include directories (where its `import`s and `#include`s are found),
    /// and `base`, the directory slangc runs in and is given paths relative to, so the
    /// CUDA text (its `#line` directives name the files) and with it the fatbin
    /// cache's key are the same in every checkout. `None`: the nearest `slang`
    /// directory above the source.
    pub fn compile_kernel_at(
        base: Option<&Path>,
        include_dirs: &[&Path],
        source: &Path,
        entry: &str,
        dest: &Path,
        defines: &[(&str, &str)],
    ) -> Result<CudaKernel> {
        let reflection = dest.with_extension("json");
        super::Slangc {
            source,
            entries: &[(entry, "compute")],
            target: "cuda",
            include_dirs,
            defines,
            out: dest,
            out_ext: "cu",
            reflection: Some(&reflection),
            base,
        }
        .run()
        .with_context(|| format!("slangc {entry}"))?;
        read_kernel(&reflection, entry)
    }

    fn read_kernel(reflection: &Path, entry: &str) -> Result<CudaKernel> {
        let json = std::fs::read_to_string(reflection)?;
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

    /// The suffix of [`grid_stride_entry`]'s second entry.
    pub const GRID_STRIDE_SUFFIX: &str = "_gs";

    /// Slang's CUDA for compute entry `entry` with a second entry, `{entry}_gs(const uint*
    /// counts)`, for an indirect dispatch: CUDA has no indirect launch, so instead of the
    /// host reading the workgroup counts back (a wait for the stream), the host launches a
    /// fixed grid and each block walks the workgroups `(x, y, z)` the GPU wrote at
    /// `counts[0..3]` (x fastest, as an indirect dispatch numbers them), running the
    /// entry's body for each with that workgroup id, a barrier between two. The same body
    /// for the same workgroups: lossless. The entry itself keeps its name and code (its
    /// body moves to an inlined `{entry}_body(uint3)`, `blockIdx` its argument).
    ///
    /// Needs, and checks: one `extern "C" __global__ void entry()`, the only place the
    /// source names `blockIdx`, and no `gridDim` (a workgroup's id is all a body may know
    /// of its grid). Barriers in the body are in uniform control flow (Chrome's rule, which
    /// every module passes), so a block's threads all reach the barrier between two
    /// workgroups.
    pub fn grid_stride_entry(cuda: &str, entry: &str) -> Result<String> {
        let head = format!("extern \"C\" __global__ void {entry}()");
        anyhow::ensure!(cuda.matches(&head).count() == 1, "{entry}: no single `{head}` in the CUDA source");
        anyhow::ensure!(cuda.matches("__global__").count() == 1, "{entry}: more than one __global__ in the CUDA source");
        anyhow::ensure!(!cuda.contains("gridDim"), "{entry}: the CUDA source reads gridDim");
        let at = cuda.find(&head).expect("checked");
        anyhow::ensure!(!cuda[..at].contains("blockIdx"), "{entry}: blockIdx outside the entry");
        let open = at + cuda[at..].find('{').ok_or_else(|| anyhow!("{entry}: entry has no body"))?;
        let mut depth = 0usize;
        let mut close = None;
        for (i, c) in cuda[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| anyhow!("{entry}: unbalanced entry body"))?;
        anyhow::ensure!(!cuda[close + 1..].contains("blockIdx"), "{entry}: blockIdx past the entry");
        let body = cuda[open..=close].replace("blockIdx", "rubble_block");
        let suffix = GRID_STRIDE_SUFFIX;
        let pre = &cuda[..at];
        let post = &cuda[close + 1..];
        Ok(format!(
            "{pre}static __device__ __forceinline__ void {entry}_body(uint3 rubble_block)\n{body}\n\n\
             {head}\n{{\n    {entry}_body(blockIdx);\n}}\n\n\
             // Indirect dispatch on the GPU (engine-slang-build grid_stride_entry): the\n\
             // workgroups at counts[0..3], walked by this launch's blocks.\n\
             extern \"C\" __global__ void {entry}{suffix}(const uint* rubble_counts)\n{{\n\
             \x20   const uint nx = rubble_counts[0];\n\
             \x20   const uint ny = rubble_counts[1];\n\
             \x20   const uint nz = rubble_counts[2];\n\
             \x20   const unsigned long long plane = (unsigned long long)nx * ny;\n\
             \x20   const unsigned long long total = plane * nz;\n\
             \x20   for (unsigned long long v = blockIdx.x; v < total; v += gridDim.x)\n    {{\n\
             \x20       {entry}_body(make_uint3((uint)(v % nx), (uint)((v / nx) % ny), (uint)(v / plane)));\n\
             \x20       __syncthreads();\n    }}\n}}\n{post}"
        ))
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
        fatbin_threads(nvcc, cu, dest, ptx_arch, sass, 0)
    }

    /// [`fatbin`] for a kernel launched with `threads` threads a block: registers per
    /// thread capped so a block fits one SM's 64K registers (`-maxrregcount`; Slang's
    /// CUDA has no `__launch_bounds__`, so ptxas may otherwise use more and the launch
    /// fails with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). The same code: spills, not
    /// different arithmetic. `threads` 0: no cap.
    pub fn fatbin_threads(
        nvcc: &Path,
        cu: &Path,
        dest: &Path,
        ptx_arch: &str,
        sass: &[&str],
        threads: u32,
    ) -> Result<()> {
        let mut cmd = std::process::Command::new(nvcc);
        cmd.args(["-fatbin", "-std=c++17", "-lineinfo"]);
        if threads > 256 {
            let regs = (65536 / threads).min(255) / 8 * 8;
            cmd.arg(format!("-maxrregcount={regs}"));
        }
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
        // Test setting (strict floats): ENGINE_CUDA_FMAD=0 keeps every a * b + c a
        // multiply and an add (-fmad=false), so two modules that do the same arithmetic
        // round alike whatever each one's code around it. Off by default (nvcc fuses).
        if std::env::var("ENGINE_CUDA_FMAD").as_deref() == Ok("0") {
            cmd.arg("-fmad=false");
        }
        for sm in sass {
            cmd.arg(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
        }
        cmd.arg(format!("-gencode=arch={ptx_arch},code={ptx_arch}"));
        let cache = FatbinCache::open(nvcc, &cmd, cu)?;
        if let Some(c) = &cache {
            if c.fetch(dest)? {
                return Ok(());
            }
        }
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
        if let Some(c) = &cache {
            c.store(dest)?;
        }
        Ok(())
    }

    /// `ENGINE_CUDA_FATBIN_CACHE=<dir>` (CI): fatbins kept by the content of what made
    /// them: the CUDA source's bytes, nvcc's `--version` and every argument. An entry
    /// holds the fatbin with that source and those words beside it, and a hit compares
    /// both byte for byte before using it, so a hash collision can only miss. A hit
    /// touches the entry (CI drops the ones a build did not touch). Unset: no cache. Not a
    /// build input (the outputs are the same with or without it), so no rerun on a change.
    struct FatbinCache {
        stem: std::path::PathBuf,
        source: Vec<u8>,
        words: String,
    }

    impl FatbinCache {
        fn open(nvcc: &Path, cmd: &std::process::Command, cu: &Path) -> Result<Option<Self>> {
            let Some(dir) = std::env::var_os("ENGINE_CUDA_FATBIN_CACHE").filter(|d| !d.is_empty()) else {
                return Ok(None);
            };
            static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
            let version = VERSION.get_or_init(|| {
                std::process::Command::new(nvcc)
                    .arg("--version")
                    .output()
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                    .unwrap_or_default()
            });
            if version.is_empty() {
                return Ok(None);
            }
            // The compilers' own bytes too: NVIDIA's aarch64 and x86_64 nvcc print the
            // same --version but write different SASS for some kernels, so a cache shared
            // between build images must not take one's fatbin for the other's.
            static TOOLS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
            let tools = TOOLS.get_or_init(|| {
                use std::hash::{Hash, Hasher};
                let bin = std::fs::canonicalize(nvcc).unwrap_or_else(|_| nvcc.to_path_buf());
                let dir = bin.parent().map(Path::to_path_buf).unwrap_or_default();
                let mut out = String::new();
                for tool in [bin.clone(), dir.join("ptxas"), dir.join("../nvvm/bin/cicc")] {
                    if let Ok(bytes) = std::fs::read(&tool) {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        bytes.hash(&mut h);
                        out.push_str(&format!("{} {:016x}\n", tool.display(), h.finish()));
                    }
                }
                out
            });
            let mut words = format!("{}\n{version}\n{tools}", nvcc.display());
            for a in cmd.get_args() {
                words.push_str(&a.to_string_lossy());
                words.push('\n');
            }
            let source = std::fs::read(cu)?;
            // Two 64-bit SipHashes with different prefixes: the name only; a hit is
            // checked against the stored source and words.
            let hash = |salt: u8| {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                salt.hash(&mut h);
                words.hash(&mut h);
                source.hash(&mut h);
                h.finish()
            };
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            let stem = dir.join(format!("{:016x}{:016x}", hash(1), hash(2)));
            Ok(Some(Self { stem, source, words }))
        }

        fn path(&self, ext: &str) -> std::path::PathBuf {
            self.stem.with_extension(ext)
        }

        /// Copy a matching entry to `dest`; false when there is none.
        fn fetch(&self, dest: &Path) -> Result<bool> {
            let same = |ext: &str, want: &[u8]| std::fs::read(self.path(ext)).is_ok_and(|b| b == want);
            if !same("cu", &self.source) || !same("args", self.words.as_bytes()) || !self.path("fatbin").exists() {
                return Ok(false);
            }
            std::fs::copy(self.path("fatbin"), dest)?;
            let now = std::time::SystemTime::now();
            for ext in ["cu", "args", "fatbin"] {
                std::fs::File::options().append(true).open(self.path(ext))?.set_modified(now)?;
            }
            Ok(true)
        }

        /// Keep `dest` (written by nvcc) as this source's entry; the fatbin last, through
        /// a rename, so a reader never sees a partial one.
        fn store(&self, dest: &Path) -> Result<()> {
            std::fs::write(self.path("cu"), &self.source)?;
            std::fs::write(self.path("args"), &self.words)?;
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let tmp = self.path(&format!("part{}-{n}", std::process::id()));
            std::fs::copy(dest, &tmp)?;
            std::fs::rename(&tmp, self.path("fatbin"))?;
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{deferred_load_hazards, grid_stride_entry};

        #[test]
        fn grid_stride_entry_keeps_the_entry_and_adds_the_walk() {
            let cu = "typedef unsigned int uint;\nstatic __device__ uint f(uint x) { return x; }\n\
                extern \"C\" __global__ void k()\n{\n    uint g = blockIdx.x;\n    if (g > 2U) { return; }\n    __syncthreads();\n}\n";
            let out = grid_stride_entry(cu, "k").unwrap();
            assert!(out.contains("static __device__ __forceinline__ void k_body(uint3 rubble_block)\n{\n    uint g = rubble_block.x;"));
            assert!(out.contains("extern \"C\" __global__ void k()\n{\n    k_body(blockIdx);\n}"));
            assert!(out.contains("extern \"C\" __global__ void k_gs(const uint* rubble_counts)"));
            assert_eq!(out.matches("__global__").count(), 2);
            assert!(out.starts_with("typedef unsigned int uint;\nstatic __device__ uint f"));
            // A body that knows more of its grid than its workgroup id is refused.
            assert!(grid_stride_entry(&cu.replace("return x;", "return blockIdx.x;"), "k").is_err());
            assert!(grid_stride_entry(&cu.replace("uint g = blockIdx.x;", "uint g = gridDim.x;"), "k").is_err());
        }

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
