//! The structured result of running one scenario: a list of invariant/oracle
//! violations (each pinpointing the exact tick + body + measured-vs-bound) plus,
//! on failure, the full per-tick trajectory for observability.

use serde::{Deserialize, Serialize};

use crate::metrics::{SystemMetrics, TickRecord};

/// A single failed check. `measured` vs `bound` makes every failure
/// self-explanatory: there is never ambiguity about what was expected or by how
/// much it was missed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Tick index at which the violation was first observed (0 = initial state).
    pub tick: usize,
    /// Body index involved, if the check is per-body.
    pub body: Option<usize>,
    /// Short machine-readable category, e.g. `"non_finite"`, `"penetration"`,
    /// `"energy_increase"`, `"ballistic_position"`.
    pub kind: String,
    /// The measured quantity (e.g. penetration depth, energy ratio, distance).
    pub measured: f64,
    /// The derived bound it exceeded.
    pub bound: f64,
    /// Human-readable context, including how the bound was derived.
    pub detail: String,
}

impl Violation {
    pub fn new(
        tick: usize,
        body: Option<usize>,
        kind: &str,
        measured: f64,
        bound: f64,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            tick,
            body,
            kind: kind.to_string(),
            measured,
            bound,
            detail: detail.into(),
        }
    }
}

/// Result of running a scenario for N ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioReport {
    pub scenario: String,
    pub steps: usize,
    pub body_count: usize,
    /// True if no GPU adapter was available (the run was vacuously skipped).
    pub skipped_no_gpu: bool,
    pub violations: Vec<Violation>,
    pub final_metrics: Option<SystemMetrics>,
    /// Populated only when there is at least one violation, so the success path
    /// stays tiny while failures carry full observability.
    pub trajectory_on_failure: Option<Vec<TickRecord>>,
}

impl ScenarioReport {
    pub fn skipped(scenario: &str) -> Self {
        Self {
            scenario: scenario.to_string(),
            steps: 0,
            body_count: 0,
            skipped_no_gpu: true,
            violations: Vec::new(),
            final_metrics: None,
            trajectory_on_failure: None,
        }
    }

    /// A scenario "passes" only if it actually ran (not skipped) and recorded no
    /// violations. A skipped run is *not* a pass — callers decide how to treat it.
    pub fn passed(&self) -> bool {
        !self.skipped_no_gpu && self.violations.is_empty()
    }

    /// One-line human summary suitable for assert messages / CI logs.
    pub fn summary(&self) -> String {
        if self.skipped_no_gpu {
            return format!("{}: SKIPPED (no GPU adapter)", self.scenario);
        }
        if self.violations.is_empty() {
            return format!(
                "{}: PASS ({} bodies, {} ticks)",
                self.scenario, self.body_count, self.steps
            );
        }
        let mut s = format!(
            "{}: FAIL ({} violation(s) over {} ticks)\n",
            self.scenario,
            self.violations.len(),
            self.steps
        );
        for v in self.violations.iter().take(12) {
            s.push_str(&format!(
                "  [tick {}{}] {}: measured={:.6} bound={:.6} — {}\n",
                v.tick,
                v.body.map(|b| format!(" body {b}")).unwrap_or_default(),
                v.kind,
                v.measured,
                v.bound,
                v.detail
            ));
        }
        if self.violations.len() > 12 {
            s.push_str(&format!("  … {} more\n", self.violations.len() - 12));
        }
        s
    }
}
