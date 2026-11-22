//! The kernels compiled at run time (the fast profile on native: `kernels_at_runtime`,
//! set by build.rs). On first use in a process the store for the current Slang sources
//! is found, or compiled (src/kernel_build.rs, every compile cached by what it depends
//! on, engine-slang-build) into a directory keyed by those sources and by the compile
//! code: `target/<profile>/kernel-store/<key>/` (`wgsl/`, `spirv/`, `spirv_entries.txt`,
//! `wgsl_ok.txt`, as build.rs writes OUT_DIR). A kernel edit then compiles no Rust: the
//! next run compiles the modules it changed.

use std::path::{Path, PathBuf};


/// The store directory of the current sources (compiled on the first call).
pub fn dir() -> &'static Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| open().unwrap_or_else(|e| panic!("the kernels (compiled at run time, the fast profile): {e}")))
}

fn slang_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../slang")
}

/// A key of the Slang tree (every file's path and bytes), the compile code
/// (`KERNEL_BUILD_KEY`, build.rs: kernel_build.rs, explicit_post.rs, wgsl_loop_cap.rs,
/// the kernel lists) and the measurement switches the compile reads.
fn key() -> Result<String, String> {
    use std::hash::{Hash, Hasher};
    let mut files = Vec::new();
    walk(&slang_root(), &mut files).map_err(|e| e.to_string())?;
    files.sort();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    env!("KERNEL_BUILD_KEY").hash(&mut h);
    for v in ["RUBBLE_EB_REF", "RUBBLE_EB_SRC", "RUBBLE_EB_REF_UNCAPPED"] {
        std::env::var(v).ok().hash(&mut h);
    }
    if let Ok(f) = std::env::var("RUBBLE_EB_SRC") {
        std::fs::read(f).ok().hash(&mut h);
    }
    for f in &files {
        f.strip_prefix(slang_root()).unwrap_or(f).hash(&mut h);
        std::fs::read(f).map_err(|e| e.to_string())?.hash(&mut h);
    }
    Ok(format!("{:016x}", h.finish()))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            walk(&p, out)?;
        } else {
            out.push(p);
        }
    }
    Ok(())
}

fn open() -> Result<PathBuf, String> {
    // (Under the profile's target directory, not this build's OUT_DIR: every build of
    // the crate in the profile, whatever its features, shares the store.)
    let out = Path::new(env!("OUT_DIR"));
    let store = out.ancestors().nth(3).unwrap_or(out).join("kernel-store");
    let dir = store.join(key()?);
    if dir.join("wgsl_ok.txt").is_file() {
        return Ok(dir);
    }
    // Compiled beside it and renamed into place: another process may compile the same
    // store at the same time (the first rename wins; the other's is dropped).
    let staging = store.join(format!(".staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
    let compiled = crate::kernel_compile::compile_all(&slang_root(), &staging).map_err(|e| format!("{e:#}"))?;
    for w in &compiled.warnings {
        eprintln!("kernels: {w}");
    }
    std::fs::write(staging.join("wgsl_ok.txt"), compiled.wgsl_ok.join("\n") + "\n").map_err(|e| e.to_string())?;
    if std::fs::rename(&staging, &dir).is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    if !dir.join("wgsl_ok.txt").is_file() {
        return Err(format!("{}: not written", dir.display()));
    }
    // (Older stores of this build: kept a while for a switch back; the oldest go.)
    prune(&store, &dir);
    Ok(dir)
}

/// Keeps the four most recent stores.
fn prune(store: &Path, keep: &Path) {
    let Ok(read) = std::fs::read_dir(store) else { return };
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = read
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p != keep && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')))
        .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
        .collect();
    dirs.sort();
    while dirs.len() > 3 {
        let (_, p) = dirs.remove(0);
        let _ = std::fs::remove_dir_all(p);
    }
}

/// A module's WGSL from the store, read once a process.
pub fn wgsl(name: &str) -> &'static str {
    static KEPT: std::sync::Mutex<Option<std::collections::HashMap<String, &'static str>>> = std::sync::Mutex::new(None);
    let mut kept = KEPT.lock().unwrap();
    let kept = kept.get_or_insert_with(Default::default);
    if let Some(w) = kept.get(name) {
        return w;
    }
    let path = dir().join("wgsl").join(format!("{name}.wgsl"));
    let w: &'static str = Box::leak(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .into_boxed_str(),
    );
    kept.insert(name.to_string(), w);
    w
}

/// The world entries whose WGSL compiled, with their text (build.rs's direct_wgsl.rs).
pub fn direct() -> &'static [(&'static str, &'static str)] {
    static DIRECT: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    DIRECT.get_or_init(|| {
        let ok = std::fs::read_to_string(dir().join("wgsl_ok.txt")).unwrap_or_default();
        ok.lines()
            .filter(|l| !l.is_empty())
            .map(|n| {
                let name: &'static str = Box::leak(n.to_string().into_boxed_str());
                (name, wgsl(name))
            })
            .collect()
    })
}

/// The entries whose SPIR-V compiled (build.rs's spirv_entries.txt).
pub fn spirv_entries() -> &'static str {
    static E: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    E.get_or_init(|| std::fs::read_to_string(dir().join("spirv_entries.txt")).unwrap_or_default())
}
