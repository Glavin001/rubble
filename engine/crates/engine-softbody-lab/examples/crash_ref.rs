//! The crash reference suite on the CPU reference (`engine_softbody::crash_reference`).
//!
//! `crash_ref -- [ID ...] [--scenario FILE.json ...] [--runs N | --run K] [--contact discrete|edge|continuous] [--json]
//! [--out DIR] [--params design=FILE[,design=FILE]]` (no id and no file: every Rust scenario; `--list`: their ids; `--run K`: run
//! K alone, the jitter of `Jitter::run(K)`; `--out DIR`: each run's report, without its time
//! histories, as `DIR/ID-contact-run.json`; `--export ID`: a Rust scenario as data, the JSON a
//! `--scenario` file holds). `--scenario FILE`: a data scenario (`crash_reference::ScenarioDesc`,
//! as the Reference Studio writes it), run by the same code; its contact setting is its own
//! unless `--contact` is given.
//! One report a run, then each scenario's verdict (`crash_reference::verdict`): exit code 1
//! when a tier 0 or tier 1 criterion fails that is not a known failure, or a known failure
//! recorded for every run passes in all of them (remove the entry).
//! `--params`: a design's parameters (e.g. a physics block with other front rules): the
//! scenarios as written, their designs generated from those parameters.

use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    // `--export ID`: the Rust scenario as data (`CrashScenario::to_desc`, run 0), printed as JSON.
    if let Some(id) = flag("--export") {
        let s = cr::get(&id).ok_or(format!("no scenario {id}"))?;
        let templates = engine_softbody_authoring::crash_templates::templates_for(&s)?;
        println!("{}", serde_json::to_string_pretty(&s.to_desc(&templates)?).map_err(|e| e.to_string())?);
        return Ok(());
    }
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
    let contact = flag("--contact").map(|x| Contact::from_name(&x).ok_or(format!("no contact setting {x}"))).transpose()?;
    let json = a.iter().any(|x| x == "--json");
    let out = flag("--out");
    let params = flag("--params").map_or(Ok(Vec::new()), |v| engine_softbody_authoring::crash_templates::params_from_list(&v))?;
    let ids: Vec<&String> = a.iter().take_while(|x| !x.starts_with("--")).collect();
    let mut list: Vec<(cr::CrashScenario, Contact)> = Vec::new();
    for (i, x) in a.iter().enumerate() {
        if x == "--scenario" {
            let file = a.get(i + 1).ok_or("--scenario FILE.json")?;
            let text = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
            let d: cr::ScenarioDesc = serde_json::from_str(&text).map_err(|e| format!("{file}: {e}"))?;
            d.check()?;
            list.push((d.scenario(), contact.unwrap_or(d.contact)));
        }
    }
    if list.is_empty() || !ids.is_empty() {
        list.extend(cr::all().into_iter().filter(|s| ids.is_empty() || ids.iter().any(|i| i.eq_ignore_ascii_case(&s.id))).map(|s| (s, contact.unwrap_or(Contact::Edge))));
    }
    let mut failed = Vec::new();
    for (s, contact) in list {
        let templates = engine_softbody_authoring::crash_templates::templates_for_with(&s, &params)?;
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
