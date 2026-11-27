//! The crash reference suite on the CPU reference (`engine_softbody::crash_reference`).
//!
//! `crash_ref -- [ID ...] [--runs N] [--contact discrete|edge|continuous] [--json] [--params design=FILE[,design=FILE]] [--run K]`
//! (`--params`: a design's parameters, e.g. a physics block with other front rules; the
//! scenarios as written, their designs from those parameters)
//! (no id: every scenario; `--list`: their ids). One report a run; exit code 1 when a
//! tier 0 or tier 1 criterion fails.

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
    let runs: usize = flag("--runs").and_then(|x| x.parse().ok()).unwrap_or(1);
    let contact = flag("--contact").map_or(Ok(Contact::Edge), |x| Contact::from_name(&x).ok_or(format!("no contact setting {x}")))?;
    let json = a.iter().any(|x| x == "--json");
    let params = flag("--params").map_or(Ok(Vec::new()), |v| engine_softbody_authoring::crash_templates::params_from_list(&v))?;
    let ids: Vec<&String> = a.iter().take_while(|x| !x.starts_with("--")).collect();
    let mut failed = 0;
    for s in cr::all().into_iter().filter(|s| ids.is_empty() || ids.iter().any(|i| i.eq_ignore_ascii_case(s.id))) {
        let templates = engine_softbody_authoring::crash_templates::templates_for_with(&s, &params)?;
        // (`--run K`: run K alone, to split a slow scenario's runs over several processes)
        let only: Option<usize> = flag("--run").and_then(|x| x.parse().ok());
        for run in only.map_or(0..runs, |k| k..k + 1) {
            let built = s.build(&templates, Jitter::run(run), contact)?;
            let t = std::time::Instant::now();
            let report = cr::run(&s, &templates, &built, run, &mut CpuRunner::new(&built), &mut |_| {})?;
            failed += report.failures().len();
            if json {
                println!("{}", serde_json::to_string(&report).map_err(|e| e.to_string())?);
            } else {
                print!("{}", report.table());
                eprintln!("  ({:.1} s)", t.elapsed().as_secs_f32());
            }
        }
    }
    if failed > 0 {
        return Err(format!("{failed} criteria failed"));
    }
    Ok(())
}
