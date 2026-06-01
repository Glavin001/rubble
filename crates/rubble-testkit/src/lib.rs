//! # rubble-testkit
//!
//! A dual-target (native + wasm) **bug-detection** harness for the Rubble physics
//! engine. It is deliberately not just a set of tests but a small framework whose
//! pieces compile to both targets so the native `cargo`/lavapipe lane and the
//! browser `Chrome+SwiftShader` lane can never silently disagree about what
//! "correct" means.
//!
//! Design principles (see the module docs for detail):
//! * **Layered oracles** — analytic discrete-recurrence ([`oracle`]), conservation
//!   laws ([`metrics`]), and (native) independent geometry — never hand-calculated
//!   magic numbers.
//! * **Derived tolerances** — every bound in [`invariants`]/[`oracle`] is computed
//!   from f32 round-off or a config-defined physical limit, so a failure is a real
//!   finding, not a cue to loosen the margin.
//! * **Per-tick observability** — invariants run on *every* tick and a failing run
//!   carries its full trajectory ([`report::ScenarioReport::trajectory_on_failure`]).
//!
//! The goal of this crate is to *detect* bugs and map gaps, not to hide them: a
//! newly strict check that fails is a gap to catalogue, not a tolerance to relax.

pub mod gaps;
pub mod invariants;
pub mod metrics;
pub mod oracle;
pub mod report;
pub mod scenario;

#[cfg(not(target_arch = "wasm32"))]
pub mod runner;

pub use gaps::{known_gaps, GapCategory, KnownGap};
pub use metrics::{BodyMeta, BodySample, SystemMetrics, TickRecord};
pub use report::{ScenarioReport, Violation};
pub use scenario::{scenario_by_name, scenario_names, scenarios, Scenario};

#[cfg(not(target_arch = "wasm32"))]
pub use runner::{run_all_native, run_native};
