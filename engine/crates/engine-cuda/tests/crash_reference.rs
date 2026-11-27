//! The crash reference suite on CUDA (`engine_softbody::crash_reference`,
//! `engine/docs/crash-reference.md`): every scenario, [`RUNS`] jittered runs, its tier 0 and
//! tier 1 criteria. GPU tests run when `ENGINE_CUDA_DEVICE=1`; without it they skip.
//!
//! `CRASH_CONTACT=discrete|edge|continuous` (default edge), `CRASH_RUNS=N`, `CRASH_RUN_FROM=K`
//! (runs K .. K + N, for a scenario whose runs take longer than one container),
//! `CRASH_SKIP=S06,..` (those scenarios' tests pass at once: a sweep runs them apart),
//! `CRASH_REPORTS=DIR` (each run's report as JSON), `CRASH_REPORT_LINES=1` (each run's
//! report without its time histories on stdout, in numbered parts after `CRASH_REPORT_PART
//! ID-contact-run k/n `: how a sweep on Modal reads them back from the log, `tools/crash-reference/sweep.py`).
//!
//! A scenario's test fails on its verdict (`crash_reference::verdict`): an asserted
//! criterion that fails and is not a known failure for the setting, or a known failure
//! recorded for every run that passes in all of them.

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
    if std::env::var("CRASH_SKIP").is_ok_and(|x| x.split(',').any(|k| k.trim().eq_ignore_ascii_case(id))) {
        eprintln!("{id}: skipped (CRASH_SKIP)");
        return;
    }
    let s = cr::get(id).expect("a registered scenario");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
    let runs = std::env::var("CRASH_RUNS").ok().and_then(|x| x.parse().ok()).unwrap_or(RUNS);
    let mut reports = Vec::new();
    let from: usize = std::env::var("CRASH_RUN_FROM").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
    for run in from..from + runs {
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
        if std::env::var("CRASH_REPORT_LINES").as_deref() == Ok("1") {
            // In parts of at most 1,000 characters (a fan-out's log cut longer lines).
            let text = serde_json::to_string(&cr::CrashReport { series: None, ..report.clone() }).unwrap();
            let chars: Vec<char> = text.chars().collect();
            let parts: Vec<String> = chars.chunks(1000).map(|c| c.iter().collect()).collect();
            for (i, p) in parts.iter().enumerate() {
                println!("CRASH_REPORT_PART {}-{}-{run} {}/{} {p}", s.id, report.contact.name(), i + 1, parts.len());
            }
        }
        reports.push(report);
    }
    let v = cr::verdict(&reports);
    for x in v.known.iter().chain(&v.notes) {
        eprintln!("{x}");
    }
    assert!(v.ok(), "{}", v.failed.join("\n"));
}

/// Whether continuous contact engages on CUDA (`all.continuous_engaged`,
/// `CudaExplicitBeams::continuous_ran`): for one vehicle against static boxes (S11) it does
/// not (the split group path it runs on needs two vehicles), and the continuous run is the
/// edge run, metric for metric; for two vehicles (S08) it does. When one vehicle engages,
/// this fails: the reports' note and `engine/docs/crash-reference.md` then change.
#[test]
fn continuous_engagement() {
    if !gpu() {
        return;
    }
    // (`tune`: a change to the runner after it is set up, for the diagnosis below.)
    let go_with = |id: &str, contact: Contact, tune: &dyn Fn(&mut CudaRunner)| {
        let s = cr::get(id).unwrap();
        let templates = engine_softbody_authoring::crash_templates::templates_for(&s).unwrap();
        let built = s.build(&templates, Jitter::run(0), contact).unwrap();
        let mut g = CudaRunner::new(&built).unwrap();
        tune(&mut g);
        cr::run(&s, &templates, &built, 0, &mut g, &mut |_| {}).unwrap()
    };
    let go = |id: &str, contact: Contact| go_with(id, contact, &|_| {});
    let (edge, edge2, cont) = (go("S11", Contact::Edge), go("S11", Contact::Edge), go("S11", Contact::Continuous));
    let differ = |a: &cr::CrashReport, b: &cr::CrashReport| a.metrics.iter().filter(|(k, v)| b.metrics.get(*k).is_none_or(|w| w.to_bits() != v.to_bits()) && !k.starts_with("all.continuous")).count();
    let (repeat, against) = (differ(&edge, &edge2), differ(&edge, &cont));
    // What makes them differ: the wide passes off (the continuous setting's), or the setting.
    let edge_narrow = go_with("S11", Contact::Edge, &|g| {
        let _ = g.0.set_wide(false);
    });
    let cont_wide = go_with("S11", Contact::Continuous, &|g| {
        let _ = g.0.set_wide(true);
    });
    let (narrow, cw) = (differ(&edge_narrow, &cont), differ(&edge, &cont_wide));
    let head = |r: &cr::CrashReport| ["all.energy_gain_j", "car.crush_peak", "car.rebound_ratio", "car.speed_end"].map(|k| format!("{k} {:.4}", r.metrics.get(k).copied().unwrap_or(f64::NAN))).join(", ");
    eprintln!("CONTINUOUS_ENGAGEMENT S11 engaged={:?} edge_repeat_differ={repeat} continuous_vs_edge_differ={against} edge_wide_off_vs_continuous_differ={narrow} continuous_wide_on_vs_edge_differ={cw} metrics={}", cont.metrics.get("all.continuous_engaged"), edge.metrics.len());
    eprintln!("  edge: {}\n  edge, wide off: {}\n  continuous: {}\n  continuous, wide on: {}", head(&edge), head(&edge_narrow), head(&cont), head(&cont_wide));
    assert_eq!(cont.metrics.get("all.continuous_engaged"), Some(&0.0), "continuous contact now engages for one vehicle on CUDA: update the report's note and the doc");
    assert!(!cont.notes().is_empty());
    if repeat == 0 {
        assert_eq!(narrow, 0, "one vehicle's continuous run differs from its edge run with the same passes on CUDA");
    }
    let two = go("S08", Contact::Continuous);
    eprintln!("CONTINUOUS_ENGAGEMENT S08 engaged={:?}", two.metrics.get("all.continuous_engaged"));
    assert_eq!(two.metrics.get("all.continuous_engaged"), Some(&1.0), "two vehicles: continuous contact runs on the split group path");
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
fn s11() {
    scenario("S11");
}

#[test]
fn s12() {
    scenario("S12");
}

#[test]
fn s13() {
    scenario("S13");
}

#[test]
fn s14() {
    scenario("S14");
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
