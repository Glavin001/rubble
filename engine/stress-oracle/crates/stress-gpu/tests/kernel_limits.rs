//! Hard GPU limits for the world kernels, checked against the limits measured on this
//! machine (`limits/<target>.toml`, from `scripts/limits.sh`). Static: the kernels are
//! compiled and their driver statistics read (`scripts/kernel_budget.sh --json`); no
//! GPU work runs.
//!
//! These are targets, not regressions (`scripts/kernel_budget.sh` guards regressions):
//! several fail today by design and are ignored by a plain `cargo test`. Run them to see
//! where each kernel stands; each prints a table of kernel, value, limit and verdict:
//!
//!   cargo test --release -p stress-gpu --test kernel_limits -- --ignored --nocapture
//!
//! A limit counts per-substep ("hot") kernels unless stated: those run hundreds to
//! thousands of times per frame, so a cliff there multiplies. macOS with Xcode only; on
//! other platforms, or when the private driver statistics are unavailable, each test
//! reports that and passes.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Kernels that run every substep (the island, wide, contact and impactor pipelines).
const HOT: [&str; 12] = [
    "island_frame",
    "wide_bonds",
    "wide_chunks",
    "wide_drift",
    "wide_rigid",
    "wide_end",
    "wide_wake",
    "contact_forces",
    "contact_sums",
    "impactor_shares",
    "impactor_crush",
    "impactor_integrate",
];

struct Budget {
    limits: serde_json::Value,
    kernels: Vec<(String, serde_json::Value)>,
}

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The kernels' statistics, collected once per test process (None: unavailable here).
fn budget() -> Option<&'static Budget> {
    static BUDGET: OnceLock<Option<Budget>> = OnceLock::new();
    BUDGET
        .get_or_init(|| {
            if !cfg!(target_os = "macos") {
                eprintln!("kernel limits: macOS (Metal) only; skipped");
                return None;
            }
            let out = std::env::temp_dir().join(format!("stress-gpu-kernel-limits-{}.json", std::process::id()));
            // The budget script's exit status is about regressions against its baseline,
            // not these limits: only its JSON matters here.
            let status = std::process::Command::new("bash").arg("scripts/kernel_budget.sh").arg("--json").arg(&out).current_dir(crate_dir()).output();
            let text = match (status, std::fs::read_to_string(&out)) {
                (Ok(_), Ok(t)) => t,
                (status, _) => {
                    eprintln!("kernel limits: scripts/kernel_budget.sh produced no statistics ({status:?}); skipped");
                    return None;
                }
            };
            std::fs::remove_file(&out).ok();
            let v: serde_json::Value = serde_json::from_str(&text).expect("budget json");
            let mut kernels: Vec<(String, serde_json::Value)> = v["kernels"].as_object().expect("kernels").iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            kernels.sort_by(|a, b| a.0.cmp(&b.0));
            eprintln!("kernel limits: from {}", v["limits_file"].as_str().unwrap_or("?"));
            Some(Budget { limits: v["limits"].clone(), kernels })
        })
        .as_ref()
}

/// Check `value(kernel) <= limit` over `kernels` (hot ones unless `all`); print the table
/// and fail listing the kernels over their limit. Unavailable values (-1) are skipped.
fn check(name: &str, unit: &str, all: bool, limit: impl Fn(&Budget) -> Option<f64>, value: impl Fn(&serde_json::Value) -> f64) {
    let Some(b) = budget() else { return };
    let Some(limit) = limit(b) else {
        eprintln!("{name}: limit not measured for this target; skipped");
        return;
    };
    eprintln!("\n{name} (limit {limit} {unit}, {} kernels)", if all { "all" } else { "hot" });
    eprintln!("{:20} {:>10} {:>8}  verdict", "kernel", unit, "x limit");
    let mut over = Vec::new();
    let mut checked = 0;
    for (kernel, k) in &b.kernels {
        if !all && !HOT.contains(&kernel.as_str()) {
            continue;
        }
        let v = value(k);
        if v < 0.0 {
            eprintln!("{kernel:20} {:>10} {:>8}  unavailable", "-", "-");
            continue;
        }
        checked += 1;
        let ok = v <= limit;
        let x = if limit > 0.0 { format!("{:.2}x", v / limit) } else { "-".into() };
        eprintln!("{kernel:20} {v:>10} {x:>8}  {}", if ok { "ok" } else { "OVER" });
        if !ok {
            over.push(format!("{kernel} {v} {unit} ({x})"));
        }
    }
    eprintln!("{} of {checked} within the limit", checked - over.len());
    assert!(over.is_empty(), "{name}: {} kernel(s) over {limit} {unit}: {}", over.len(), over.join(", "));
}

fn field(k: &serde_json::Value, key: &str) -> f64 {
    k[key].as_f64().unwrap_or(-1.0)
}

fn limit(b: &Budget, key: &str) -> Option<f64> {
    b.limits[key].as_f64()
}

/// Machine code within the measured instruction-cache knee (straight-line code; mixed code
/// with loads and branches knees earlier): beyond it every executed instruction streams
/// from memory at about twice the cost.
#[test]
#[ignore = "hard-limit target; currently failing (run with --ignored)"]
fn hot_kernels_fit_the_instruction_cache() {
    check("instruction cache", "code bytes", false, |b| limit(b, "icache_code_bytes"), |k| field(k, "code_bytes"));
}

/// No spills in per-substep kernels: each reload is a dependent memory load (~45-150 ns).
#[test]
#[ignore = "hard-limit target; currently failing (run with --ignored)"]
fn hot_kernels_do_not_spill() {
    check("spilled bytes per thread", "bytes", false, |_| Some(0.0), |k| field(k, "spilled_bytes"));
}

/// Spill footprint (spilled bytes x threads per group) within the measured stack knee:
/// beyond it the spill stack leaves fast on-chip storage (stack loads ~55 -> 96-150 ns).
#[test]
#[ignore = "hard-limit target; currently failing (run with --ignored)"]
fn spill_stack_per_threadgroup_within_the_knee() {
    check("spill stack per threadgroup", "bytes", true, |b| limit(b, "stack_bytes_per_threadgroup"), |k| field(k, "spilled_bytes_x_threads"));
}

/// Register headroom: a hot kernel at the driver's cap (96 temporaries) is spilling or one
/// value away from it.
#[test]
#[ignore = "hard-limit target; currently failing (run with --ignored)"]
fn hot_kernels_below_the_register_cap() {
    check("temporary registers", "registers", false, |b| limit(b, "max_temp_registers").map(|r| r - 1.0), |k| field(k, "temp_registers"));
}

/// Static threadgroup memory within the device limit (32 KB on Apple GPUs).
#[test]
#[ignore = "hard-limit target (run with --ignored)"]
fn threadgroup_memory_within_the_device_limit() {
    check("static threadgroup memory", "bytes", true, |b| limit(b, "max_threadgroup_memory_bytes"), |k| field(k, "static_threadgroup_memory"));
}

/// Every pipeline accepts the threadgroup size it is dispatched with.
#[test]
#[ignore = "hard-limit target (run with --ignored)"]
fn pipelines_accept_their_threadgroup_size() {
    let Some(b) = budget() else { return };
    let bad: Vec<String> = b
        .kernels
        .iter()
        .filter(|(_, k)| field(k, "threads_per_threadgroup") > 0.0 && field(k, "max_threads_per_threadgroup") >= 0.0 && field(k, "max_threads_per_threadgroup") < field(k, "threads_per_threadgroup"))
        .map(|(n, k)| format!("{n} (max {} < {})", field(k, "max_threads_per_threadgroup"), field(k, "threads_per_threadgroup")))
        .collect();
    eprintln!("pipelines: {} of {} accept their threadgroup size", b.kernels.len() - bad.len(), b.kernels.len());
    assert!(bad.is_empty(), "pipelines below their threadgroup size: {}", bad.join(", "));
}
