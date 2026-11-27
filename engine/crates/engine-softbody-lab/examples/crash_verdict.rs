//! The crash reference suite's verdict (`crash_reference::verdict`) on saved reports: each file
//! is one scenario and contact setting's runs, one `crash_ref --json` report a line (runs made in
//! separate processes, `crash_ref --run K`), judged again by the scenario's current criteria. Prints the unknown failures (which fail it), the known
//! failures with their runs, and the notes.
//!
//! `crash_verdict -- FILE.jsonl [FILE.jsonl ...]` (exit code 1 when any file's verdict fails)

use engine_softbody::crash_reference::{self as cr, CrashReport};

fn main() -> Result<(), String> {
    let mut failed = 0;
    for f in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?;
        let mut reports: Vec<CrashReport> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| format!("{f}: {e}"))).collect::<Result<_, _>>()?;
        // (judged again by the scenario's current criteria and known failures, so reports saved
        // before an entry was recorded read the same as new ones)
        for r in &mut reports {
            let s = cr::get(&r.id).ok_or(format!("{f}: no scenario {}", r.id))?;
            r.results = cr::judge(&s, &r.metrics, r.contact);
        }
        let v = cr::verdict(&reports);
        println!("{f}: {} runs, {}", reports.len(), if v.ok() { "ok" } else { "FAILED" });
        for x in &v.failed {
            println!("  FAILED {x}");
        }
        for x in &v.known {
            println!("  known {x}");
        }
        for x in &v.notes {
            println!("  note {x}");
        }
        failed += usize::from(!v.ok());
    }
    if failed > 0 {
        return Err(format!("{failed} verdicts failed"));
    }
    Ok(())
}
