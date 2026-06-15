//! Native lane for the dual-target detection harness (`rubble-testkit`).
//!
//! Runs the full scenario ladder against whatever GPU adapter is available
//! (software lavapipe in CI), recording every tick and running the
//! derived-tolerance invariants + analytic oracles. Failures are partitioned
//! against the gap registry (`rubble_testkit::gaps`):
//!   * catalogued gaps are tolerated so CI stays green while they're tracked;
//!   * an *unregistered* failure breaks the build (a real regression);
//!   * a registered gap that now *passes* is reported so the registry can be
//!     cleaned up.
//!
//! Set `RUBBLE_RUN_KNOWN_FAILURES=1` to require *everything* to pass (the CI lane
//! that surfaces the catalogued gaps). Run with `--nocapture` to see the full
//! per-scenario report — the gap-detection deliverable.

use rubble_testkit::{gaps, run_all_native};

fn run_known_failures_enabled() -> bool {
    std::env::var_os("RUBBLE_RUN_KNOWN_FAILURES").is_some()
}

#[test]
fn ladder_detects_no_unregistered_gaps() {
    let reports = run_all_native();
    if reports.iter().all(|r| r.skipped_no_gpu) {
        eprintln!("SKIP: no GPU adapter available for any scenario");
        return;
    }

    let mut unexpected = Vec::new(); // failing, not registered
    let mut known_failing = Vec::new(); // failing, registered
    let mut fixed = Vec::new(); // passing, but registered as a gap

    println!("\n=== Rubble testkit — scenario ladder ===");
    for r in &reports {
        if r.skipped_no_gpu {
            println!("{}", r.summary());
            continue;
        }
        let registered = gaps::lookup(&r.scenario).is_some();
        println!("{}", r.summary());
        match (r.violations.is_empty(), registered) {
            (true, true) => fixed.push(r.scenario.clone()),
            (true, false) => {}
            (false, true) => known_failing.push(r.scenario.clone()),
            (false, false) => unexpected.push(r.scenario.clone()),
        }
    }

    println!("\n--- gap registry ---");
    for g in gaps::known_gaps() {
        println!("[{:?}] {}\n    {}", g.category, g.scenario, g.reason);
    }
    println!("\ncatalogued gaps still failing : {known_failing:?}");
    if !fixed.is_empty() {
        println!("registered gaps that now PASS (update the registry!): {fixed:?}");
    }
    println!("UNEXPECTED failures (regressions): {unexpected:?}\n");

    if run_known_failures_enabled() {
        let all_failing: Vec<String> = reports
            .iter()
            .filter(|r| !r.skipped_no_gpu && !r.violations.is_empty())
            .map(|r| r.scenario.clone())
            .collect();
        assert!(
            all_failing.is_empty(),
            "RUBBLE_RUN_KNOWN_FAILURES set: scenarios still failing the harness: {all_failing:?}"
        );
    } else {
        assert!(
            unexpected.is_empty(),
            "NEW gaps not present in the registry (regression — investigate before adding): {unexpected:?}"
        );
    }
}
