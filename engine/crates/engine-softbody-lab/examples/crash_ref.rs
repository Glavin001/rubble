//! The crash reference suite on the CPU reference (`engine_softbody::crash_reference`).
//!
//! `crash_ref -- [ID ...] [--runs N | --run K] [--contact discrete|edge|continuous] [--json]
//! [--out DIR]` (no id: every scenario; `--list`: their ids; `--run K`: run K alone, the
//! jitter of `Jitter::run(K)`; `--out DIR`: each run's report, without its time histories, as
//! `DIR/ID-contact-run.json`).
//! One report a run, then each scenario's verdict (`crash_reference::verdict`): exit code 1
//! when a tier 0 or tier 1 criterion fails that is not a known failure, or a known failure
//! recorded for every run passes in all of them (remove the entry).

use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    if a.iter().any(|x| x == "--list") {
        for s in cr::all() {
            println!("{} {}", s.id, s.title);
        }
        return Ok(());
    }
    let runs: Vec<usize> = match flag("--run") {
        Some(k) => vec![k.parse().map_err(|_| format!("--run {k}"))?],
        None => (0..flag("--runs").and_then(|x| x.parse().ok()).unwrap_or(1)).collect(),
    };
    let contact = flag("--contact").map_or(Ok(Contact::Edge), |x| Contact::from_name(&x).ok_or(format!("no contact setting {x}")))?;
    let json = a.iter().any(|x| x == "--json");
    let out = flag("--out");
    let ids: Vec<&String> = a.iter().take_while(|x| !x.starts_with("--")).collect();
    let mut failed = Vec::new();
    for s in cr::all().into_iter().filter(|s| ids.is_empty() || ids.iter().any(|i| i.eq_ignore_ascii_case(s.id))) {
        let templates = engine_softbody_authoring::crash_templates::templates_for(&s)?;
        let mut reports = Vec::new();
        for &run in &runs {
            let built = s.build(&templates, Jitter::run(run), contact)?;
            let t = std::time::Instant::now();
            let report = cr::run(&s, &templates, &built, run, &mut CpuRunner::new(&built), &mut |_| {})?;
            if let Some(dir) = &out {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                let text = serde_json::to_string(&cr::CrashReport { series: None, ..report.clone() }).map_err(|e| e.to_string())?;
                std::fs::write(format!("{dir}/{}-{}-{run}.json", s.id, contact.name()), text).map_err(|e| e.to_string())?;
            }
            if json {
                println!("{}", serde_json::to_string(&report).map_err(|e| e.to_string())?);
            } else {
                print!("{}", report.table());
                eprintln!("  ({:.1} s)", t.elapsed().as_secs_f32());
            }
            reports.push(report);
        }
        let v = cr::verdict(&reports);
        for x in v.known.iter().chain(&v.notes) {
            eprintln!("{x}");
        }
        for x in &v.failed {
            eprintln!("FAILED {x}");
        }
        failed.extend(v.failed);
    }
    if !failed.is_empty() {
        return Err(format!("{} failed:\n{}", failed.len(), failed.join("\n")));
    }
    Ok(())
}
