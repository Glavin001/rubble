//! "Test the tests": the fault-detection matrix.
//!
//! Injects each known fault (negated/zeroed/scaled gravity, single solver
//! iteration, zeroed friction) into every scenario and checks which
//! baseline-*passing* scenarios then fail. A scenario that fails under a fault
//! "catches" it; a fault no scenario catches is a **blind spot** in the suite.
//!
//! This is what proves the harness has teeth — that its derived tolerances are
//! tight enough to fail on a real bug, not merely loose enough to pass anything.
//! It is `#[ignore]`d (heavy lane: it runs every scenario once per fault); run
//! with `--ignored`.

use rubble_testkit::faults::{all_faults, FaultKind};
use rubble_testkit::{run_native, run_native_with_fault, scenarios};

/// Faults we do not yet expect the *passing* suite to catch, with the reason.
/// These are suite blind spots to close, tracked here so the matrix stays green
/// while they're documented (analogous to the engine gap registry).
///
/// Currently EMPTY: every fault in the catalog is caught by at least one
/// baseline-passing scenario. (ZeroFriction is caught by
/// `gentle_incline_friction_holds`, which holds tangentially under friction and
/// slides without it.)
fn known_blind_spots() -> &'static [(FaultKind, &'static str)] {
    &[]
}

#[test]
#[ignore = "fault-injection lane; run with --ignored"]
fn fault_detection_matrix() {
    let scns = scenarios();

    // Baseline: which scenarios pass cleanly? Only these can unambiguously
    // "catch" a fault (a failure under the fault is then caused by the fault).
    let baseline: Vec<_> = scns.iter().map(run_native).collect();
    if baseline.iter().all(|r| r.skipped_no_gpu) {
        eprintln!("SKIP: no GPU adapter available");
        return;
    }
    let passing: Vec<bool> = baseline.iter().map(|r| r.passed()).collect();

    println!("\n=== Fault detection matrix (✗ = fault caught by scenario) ===");
    println!(
        "baseline-passing scenarios: {:?}\n",
        scns.iter()
            .zip(&passing)
            .filter(|(_, &p)| p)
            .map(|(s, _)| s.name)
            .collect::<Vec<_>>()
    );

    let mut undetected: Vec<FaultKind> = Vec::new();

    for fault in all_faults() {
        let mut caught_by: Vec<&str> = Vec::new();
        for (i, scn) in scns.iter().enumerate() {
            if !passing[i] {
                continue; // only clean baselines give an unambiguous signal
            }
            let faulted = run_native_with_fault(scn, fault);
            if !faulted.skipped_no_gpu && !faulted.violations.is_empty() {
                caught_by.push(scn.name);
            }
        }
        println!(
            "{:>22} : {}",
            fault.label(),
            if caught_by.is_empty() {
                "NOT DETECTED".to_string()
            } else {
                format!("caught by {caught_by:?}")
            }
        );
        if caught_by.is_empty() {
            undetected.push(fault);
        }
    }

    let blind: Vec<FaultKind> = known_blind_spots().iter().map(|(f, _)| *f).collect();
    let unexpected: Vec<FaultKind> = undetected
        .iter()
        .copied()
        .filter(|f| !blind.contains(f))
        .collect();

    println!("\n--- documented blind spots ---");
    for (f, why) in known_blind_spots() {
        println!("{:?}: {why}", f);
    }
    println!("\nundetected (unexpected): {unexpected:?}\n");

    assert!(
        unexpected.is_empty(),
        "faults caught by NO passing scenario and not documented as blind spots \
         (the suite cannot detect these bugs): {unexpected:?}"
    );
}
