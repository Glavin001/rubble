//! The crash reference suite on CUDA (`engine_softbody::crash_reference`,
//! `engine/docs/crash-reference.md`): every scenario, [`RUNS`] jittered runs, its tier 0 and
//! tier 1 criteria. GPU tests run when `ENGINE_CUDA_DEVICE=1`; without it they skip.
//!
//! `CRASH_CONTACT=discrete|edge|continuous` (default edge), `CRASH_RUNS=N`,
//! `CRASH_REPORTS=DIR` (each run's report as JSON).

use engine_cuda::explicit::crash_runner::CudaRunner;
use engine_cuda::CudaExplicitBeams;
use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter, RUNS};

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host (built with nvcc)");
        return false;
    }
    assert!(CudaExplicitBeams::available(), "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded");
    true
}

fn contact() -> Contact {
    std::env::var("CRASH_CONTACT").ok().and_then(|x| Contact::from_name(&x)).unwrap_or(Contact::Edge)
}

fn scenario(id: &str) {
    if !gpu() {
        return;
    }
    let s = cr::get(id).expect("a registered scenario");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    let runs = std::env::var("CRASH_RUNS").ok().and_then(|x| x.parse().ok()).unwrap_or(RUNS);
    let mut failed = Vec::new();
    for run in 0..runs {
        let built = s.build(&templates, Jitter::run(run), contact()).expect("build");
        let t = std::time::Instant::now();
        let mut g = CudaRunner::new(&built).expect("CUDA");
        let report = cr::run(&s, &templates, &built, run, &mut g, &mut |_| {}).expect("run");
        eprint!("{}", report.table());
        eprintln!("  ({:.1} s on {})", t.elapsed().as_secs_f32(), g.0.device_name());
        if let Ok(dir) = std::env::var("CRASH_REPORTS") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/{}-{}-{run}.json", s.id, report.contact.name()), serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
        failed.extend(report.failures().into_iter().map(|r| format!("{} run {run}: {} = {} outside {}", s.id, r.name, r.shown(), r.band())));
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The same run on CUDA and on the CPU reference: the same code measures both, so their
/// metrics are printed side by side (CUDA's parity is by tolerance, as `explicit_parity.rs`).
#[test]
fn cuda_beside_cpu_s02() {
    if !gpu() {
        return;
    }
    let s = cr::get("S02").unwrap();
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).unwrap();
    let built = s.build(&templates, Jitter::run(0), contact()).unwrap();
    let a = cr::run(&s, &templates, &built, 0, &mut CudaRunner::new(&built).unwrap(), &mut |_| {}).unwrap();
    let b = cr::run(&s, &templates, &built, 0, &mut CpuRunner::new(&built), &mut |_| {}).unwrap();
    for (k, v) in &a.metrics {
        eprintln!("{k}: CUDA {v:.4}, CPU {:.4}", b.metrics[k]);
    }
    assert_eq!(a.metrics["all.finite"], 1.0);
}

#[test]
fn s01() {
    scenario("S01");
}

#[test]
fn s02() {
    scenario("S02");
}

#[test]
fn s03() {
    scenario("S03");
}

#[test]
fn s07() {
    scenario("S07");
}

#[test]
fn s08() {
    scenario("S08");
}

#[test]
fn s10() {
    scenario("S10");
}

#[test]
fn s09() {
    scenario("S09");
}

#[test]
fn s04() {
    scenario("S04");
}

#[test]
fn s05() {
    scenario("S05");
}

#[test]
fn s06() {
    scenario("S06");
}
