//! The generated WGSL of every kernel that exists on `main` (30ab0e1d) is
//! byte-identical to main's, except the kernels listed in
//! `engine/scripts/pr105-shader-exceptions.tsv` with the reason (PR #105 regression
//! audit). The golden list `engine/scripts/pr105-main-shaders.tsv` was recorded from a
//! build of main with the pinned slangc (TOOLCHAIN.md) by
//! `engine/scripts/shader-golden.py record`; that script checks the SPIR-V, MSL and
//! CUDA outputs of a build the same way. No GPU: this reads the build's files.
//!
//! When a kernel is changed on purpose after this, add its line to the exceptions
//! with the reason, or re-record the golden from the new base.

use std::collections::BTreeMap;

fn fnv(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in data {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// `*` matches any run of characters (one star per pattern is all the list uses).
fn glob(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        Some((a, b)) => name.len() >= a.len() + b.len() && name.starts_with(a) && name.ends_with(b),
        None => pattern == name,
    }
}

#[test]
fn every_main_kernel_wgsl_is_mains_or_listed() {
    let scripts = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts");
    let golden = std::fs::read_to_string(format!("{scripts}/pr105-main-shaders.tsv")).expect("golden list");
    let exceptions: Vec<(String, String)> = std::fs::read_to_string(format!("{scripts}/pr105-shader-exceptions.tsv"))
        .expect("exceptions")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (p, why) = l.split_once('\t').expect("pattern<TAB>reason");
            assert!(why.len() > 20, "{p}: an exception needs its reason");
            (p.to_string(), why.to_string())
        })
        .collect();
    let dir = engine_webgpu::EXPLICIT_BEAMS_WGSL_DIR;
    let mut used: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut same, mut listed, mut bad) = (0, 0, Vec::new());
    for line in golden.lines() {
        let (key, want) = line.split_once('\t').expect("file<TAB>hash");
        let Some(name) = key.strip_prefix("wgsl/") else {
            continue;
        };
        let got = std::fs::read(format!("{dir}/{name}")).ok().map(|b| {
            let kept: Vec<&[u8]> = b.split(|c| *c == b'\n').filter(|l| !l.starts_with(b"#line ")).collect();
            fnv(&kept.join(&b'\n'))
        });
        if got.as_deref() == Some(want) {
            same += 1;
            continue;
        }
        match exceptions.iter().find(|(p, _)| glob(p, key)) {
            Some((p, _)) => {
                listed += 1;
                *used.entry(p.as_str()).or_default() += 1;
            }
            None => bad.push(format!("{name}: {}", if got.is_some() { "differs from main" } else { "missing" })),
        }
    }
    eprintln!("{same} kernels identical to main, {listed} differ as listed");
    assert!(same >= 300, "only {same} kernels compared equal: is this the pinned slangc (TOOLCHAIN.md)?");
    assert!(
        bad.is_empty(),
        "{} kernel(s) that exist on main have different generated WGSL and are not listed in \
         scripts/pr105-shader-exceptions.tsv:\n{}",
        bad.len(),
        bad.join("\n")
    );
    for (p, _) in &exceptions {
        if p.starts_with("wgsl/") {
            assert!(used.contains_key(p.as_str()), "exception `{p}` matches no changed kernel: remove it");
        }
    }
    // The rigid step, terrain, fracture ingest and soft kernels are never excepted.
    for (p, _) in &exceptions {
        assert!(p.starts_with("wgsl/explicit_"), "`{p}`: only explicit-beams kernels may differ from main");
    }
}
