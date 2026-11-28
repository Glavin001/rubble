//! Many entries of one Slang source compiled in one process ([`compile_many`] and the
//! `slang_batch` tool, `batch/slang_batch.cpp`): one parse of the source instead of one
//! per entry. engine/docs/build-architecture.md.

use crate::{
    build_jobs, find_slangc, include_closure, par_map, rel_to, slang_base, unique, SlangTarget, Slangc,
};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;

/// One single-entry compile for [`compile_many`].
#[derive(Clone, Copy, Debug)]
pub struct Compile<'a> {
    pub source: &'a Path,
    pub entry: &'a str,
    pub stage: &'a str,
    pub target: SlangTarget,
    pub out: &'a Path,
    pub include_dirs: &'a [&'a Path],
    pub defines: &'a [(&'a str, &'a str)],
}

impl<'a> Compile<'a> {
    fn slangc(&self, entries: &'a [(&'a str, &'a str)]) -> Slangc<'a> {
        Slangc {
            source: self.source,
            entries,
            target: self.target.flag(),
            include_dirs: self.include_dirs,
            defines: self.defines,
            out: self.out,
            out_ext: "",
            reflection: None,
            base: None,
        }
    }
}

/// Every compile of `jobs`, results in order; each output is what
/// [`compile_stage`] writes for it, byte for byte.
///
/// Faster than one [`compile_stage`] a job when many jobs share a source, target and
/// defines (the rigid kernels: a hundred entries of `world_step.slang`): those not in
/// the cache compile in a few `slang_batch` processes that parse the source once each,
/// instead of a slangc each (about half a second of parsing apiece). What a batch does
/// not write (an error, an entry it skips, a batch past its time limit) compiles with
/// slangc as before, so errors read as they always did.
pub fn compile_many(jobs: &[Compile]) -> Vec<Result<()>> {
    // RUBBLE_SLANG_TIMING=1: each phase's seconds on the build script's stderr.
    let timing = std::env::var("RUBBLE_SLANG_TIMING").is_ok_and(|v| v == "1");
    let t0 = std::time::Instant::now();
    let lap = |what: &str, n: usize| {
        if timing {
            eprintln!("compile_many: {what}: {n} in {:.2} s", t0.elapsed().as_secs_f64());
        }
    };
    let Ok(slangc) = find_slangc() else {
        return jobs.iter().map(|j| j.slangc_run()).collect();
    };
    // Cache hits first.
    let done: Vec<bool> = par_map(jobs, |j| {
        if let Some(parent) = j.out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let entries = [(j.entry, j.stage)];
        let s = j.slangc(&entries);
        s.restore(s.cache_entry(&slangc).as_deref()).unwrap_or(false)
    });
    let mut done = done;
    lap("cache hits", done.iter().filter(|d| **d).count());
    if let Some(tool) = batch_tool(&slangc) {
        // The rest by source, target, include directories and defines: one batch a
        // group (split so every core has one), each entry checked for what the batch
        // cannot do.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, j) in jobs.iter().enumerate() {
            // WGSL and SPIR-V only: their outputs are slangc's byte for byte. The MSL a
            // batch writes is the same code with some `#line` numbers that differ from
            // slangc's (the session places inlined helpers' lines otherwise), so MSL
            // and CUDA (which names lines too) stay one slangc a compile.
            if done[i] || !matches!(j.target, SlangTarget::Wgsl | SlangTarget::Spirv) {
                continue;
            }
            let key = format!("{:?}|{}|{:?}|{:?}|{}", j.source, j.target.flag(), j.include_dirs, j.defines, j.stage);
            let g = *index.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[g].push(i);
        }
        let mut chunks: Vec<Vec<usize>> = Vec::new();
        let pending: usize = groups.iter().map(Vec::len).sum();
        // (Two batches a core: the large entries do not all land in one.)
        let per = pending.div_ceil(2 * build_jobs()).max(4);
        let calls = par_map(&groups, |g| entry_calls(jobs[g[0]].source, jobs[g[0]].include_dirs));
        for (g, calls) in groups.into_iter().zip(calls) {
            let eligible: Vec<usize> = g
                .into_iter()
                .filter(|&i| calls.as_ref().is_some_and(|c| !c.contains(jobs[i].entry)))
                .collect();
            if eligible.len() < 2 {
                continue;
            }
            for c in eligible.chunks(per) {
                chunks.push(c.to_vec());
            }
        }
        lap("batches planned", chunks.len());
        let written: Vec<usize> = par_map(&chunks, |c| run_batch(&tool, jobs, c)).into_iter().flatten().collect();
        // The batch's outputs stand only if they are slangc's: one of each target is
        // compiled again with slangc and compared byte for byte (on this platform, with
        // this Slang). Any difference: every batch output goes back to slangc, and no
        // more batches in this process.
        let mut trusted = true;
        for target in [SlangTarget::Wgsl, SlangTarget::Spirv] {
            let Some(&i) = written.iter().find(|&&i| jobs[i].target == target) else { continue };
            let j = &jobs[i];
            let check = j.out.with_extension(format!("slangc-check{}", unique()));
            let entries = [(j.entry, j.stage)];
            let mut s = j.slangc(&entries);
            s.out = &check;
            // (Not through the cache: this is slangc's own output.)
            let same = s.spawn(&slangc).is_ok() && std::fs::read(&check).ok() == std::fs::read(j.out).ok();
            let _ = std::fs::remove_file(&check);
            if !same {
                trusted = false;
                BATCH_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
                println!("cargo:warning=slang_batch output differs from slangc's ({} {}): compiling with slangc", target.flag(), j.entry);
                break;
            }
        }
        if trusted {
            for &i in &written {
                let j = &jobs[i];
                let entries = [(j.entry, j.stage)];
                let s = j.slangc(&entries);
                s.store(s.cache_entry(&slangc).as_deref());
                done[i] = true;
            }
        }
    }
    lap("batches done", done.iter().filter(|d| **d).count());
    let rest: Vec<usize> = (0..jobs.len()).filter(|&i| !done[i]).collect();
    let mut results: Vec<Result<()>> = (0..jobs.len()).map(|_| Ok(())).collect();
    for (i, r) in rest.iter().zip(par_map(&rest, |&i| jobs[i].slangc_run())) {
        results[*i] = r;
    }
    lap("slangc done", rest.len());
    results
}

impl Compile<'_> {
    fn slangc_run(&self) -> Result<()> {
        let entries = [(self.entry, self.stage)];
        self.slangc(&entries).run()
    }
}

/// Set when a batch's output was not slangc's (`compile_many`'s check): no more
/// batches in this process.
static BATCH_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The batch compiler built with this crate (`build.rs`), when its Slang is slangc's.
fn batch_tool(slangc: &Path) -> Option<PathBuf> {
    if BATCH_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    static TOOL: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    TOOL.get_or_init(|| {
        if std::env::var("RUBBLE_SLANG_BATCH").is_ok_and(|v| v == "0") {
            return None;
        }
        let tool = PathBuf::from(option_env!("ENGINE_SLANG_BATCH").filter(|p| !p.is_empty())?);
        let version = |cmd: &mut Command| {
            let o = cmd.output().ok()?;
            o.status.success().then(|| {
                let mut s = String::from_utf8_lossy(&o.stdout).to_string();
                s.push_str(&String::from_utf8_lossy(&o.stderr));
                s.trim().to_string()
            })
        };
        let mine = version(Command::new(&tool).arg("--version"))?;
        let theirs = version(Command::new(slangc).arg("-v"))?;
        (!mine.is_empty() && mine == theirs).then_some(tool)
    })
    .clone()
}

/// Every compute entry of `source`'s include closure whose body calls another entry
/// point (None: the closure could not be read). The Slang library's link step does not
/// finish on such an entry (slangc's compile does): they compile with slangc.
fn entry_calls(source: &Path, include_dirs: &[&Path]) -> Option<std::collections::HashSet<String>> {
    // Once a source and include list per process: every group of that source asks.
    type Memo = std::collections::HashMap<String, Option<std::collections::HashSet<String>>>;
    static MEMO: std::sync::Mutex<Option<Memo>> = std::sync::Mutex::new(None);
    let key = format!("{source:?}|{include_dirs:?}");
    if let Some(hit) = MEMO.lock().unwrap().get_or_insert_with(Default::default).get(&key) {
        return hit.clone();
    }
    let found = scan_entry_calls(source, include_dirs);
    MEMO.lock().unwrap().get_or_insert_with(Default::default).insert(key, found.clone());
    found
}

fn scan_entry_calls(source: &Path, include_dirs: &[&Path]) -> Option<std::collections::HashSet<String>> {
    let texts = include_closure(source, include_dirs, None)?;
    let mut names = Vec::new();
    for (_, text) in &texts {
        let mut shader = false;
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("[shader(") {
                shader = true;
            } else if shader && !t.starts_with('[') && !t.is_empty() {
                if let Some(rest) = t.strip_prefix("void ") {
                    if let Some(name) = rest.split('(').next() {
                        names.push(name.trim().to_string());
                    }
                }
                shader = false;
            }
        }
    }
    let mut calls = std::collections::HashSet::new();
    for (_, text) in &texts {
        for name in &names {
            let Some(body) = function_body(text, name) else { continue };
            if names.iter().any(|other| other != name && calls_name(body, other)) {
                calls.insert(name.clone());
            }
        }
    }
    Some(calls)
}

/// The text between the braces of `void name(...) { ... }` in `text`.
fn function_body<'t>(text: &'t str, name: &str) -> Option<&'t str> {
    let at = text.find(&format!("void {name}("))?;
    let open = at + text[at..].find('{')?;
    let mut depth = 0usize;
    for (k, c) in text[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[open + 1..open + k]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Whether `body` has `name(` as a word of its own.
fn calls_name(body: &str, name: &str) -> bool {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    body.match_indices(name).any(|(at, _)| {
        let before = body[..at].chars().next_back();
        let after = body[at + name.len()..].trim_start().chars().next();
        !before.is_some_and(word) && after == Some('(')
    })
}

/// Runs one batch of `jobs[chunk]` (one source, target and define set); returns the
/// indices whose output it wrote. A batch past its time limit is stopped: what it
/// finished stands, the rest compile with slangc.
fn run_batch(tool: &Path, jobs: &[Compile], chunk: &[usize]) -> Vec<usize> {
    use std::io::BufRead;
    let first = &jobs[chunk[0]];
    let base = slang_base(first.source);
    let mut job = format!(
        "target {}\nsource {}\n",
        first.target.flag(),
        rel_to(base.as_deref(), first.source).display()
    );
    for d in first.include_dirs {
        job.push_str(&format!("include {}\n", rel_to(base.as_deref(), d).display()));
    }
    for (n, v) in first.defines {
        job.push_str(&format!("define {n}={v}\n"));
    }
    for &i in chunk {
        let j = &jobs[i];
        job.push_str(&format!("entry {} {} {}\n", j.entry, j.stage, j.out.display()));
    }
    let file = std::env::temp_dir().join(format!("rubble-slang-batch-{}-{}.job", std::process::id(), unique()));
    if std::fs::write(&file, job).is_err() {
        return Vec::new();
    }
    let mut cmd = Command::new(tool);
    cmd.arg(&file).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    if let Some(b) = &base {
        cmd.current_dir(b);
    }
    let Ok(mut child) = cmd.spawn() else {
        let _ = std::fs::remove_file(&file);
        return Vec::new();
    };
    let stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut ok = Vec::new();
        for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(k) = line.strip_prefix("ok ").and_then(|k| k.trim().parse::<usize>().ok()) {
                ok.push(k);
            }
        }
        ok
    });
    // About a second an entry at most (a few tenths is usual), and 20 s to start.
    let limit = std::time::Duration::from_secs(20 + chunk.len() as u64);
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < limit => std::thread::sleep(std::time::Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
        }
    }
    let ok = reader.join().unwrap_or_default();
    let _ = std::fs::remove_file(&file);
    if std::env::var("RUBBLE_SLANG_TIMING").is_ok_and(|v| v == "1") {
        eprintln!(
            "slang_batch: {} {} {:?}: {} of {} in {:.2} s",
            first.target.flag(),
            first.source.file_name().unwrap_or_default().to_string_lossy(),
            first.defines,
            ok.len(),
            chunk.len(),
            start.elapsed().as_secs_f64()
        );
    }
    ok.into_iter().filter_map(|k| chunk.get(k).copied()).collect()
}

