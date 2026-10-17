//! The micro tier in-process (`stress_gpu::micro`): every catalogue case through the
//! production path with its semantic checks, path coverage and knob runs, plus the fuzz
//! regressions. Cases without a cached reference are skipped with a message (compute
//! them with `stress-gpu-gate --build --micro`, CPU only, seconds).

use stress_gpu::gpu::Gpu;
use stress_gpu::micro::{run_micro, SnapshotState};

#[test]
fn micro_tier() {
    let gpu = Gpu::new().expect("GPU");
    let report = run_micro(&gpu, None, false, false);
    let mut failures = Vec::new();
    for r in &report.rows {
        if r.missing {
            eprintln!("{} ({}): no cached reference, skipped (run stress-gpu-gate --build --micro)", r.case, r.knobs);
        } else if !r.fails.is_empty() {
            failures.push(format!("{} ({}): {}", r.case, r.knobs, r.fails.join("; ")));
        }
        if let Some(SnapshotState::Changed(what)) = &r.snapshot {
            eprintln!("{} ({}): GPU snapshot changed ({})", r.case, r.knobs, what.join(","));
        }
    }
    eprintln!("{} micro runs, {} skipped", report.rows.len(), report.missing());
    assert!(failures.is_empty(), "micro tier:\n{}", failures.join("\n"));
}
