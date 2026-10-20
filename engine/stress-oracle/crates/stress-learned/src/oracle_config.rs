//! The oracle configuration of the impact-outcome dataset: one explicit, recorded
//! parameter (plan P1: "make the oracle configuration a single explicit, recorded
//! parameter"), so every event can be regenerated when P0's pin changes.
//!
//! Every field the configuration sets is written in full to each event's `context.json`
//! (including every `Methods` switch, which `scene.json` omits when it equals the
//! built-in default and which `STRESS_METHODS` can otherwise change), and re-applied
//! from there on replay.

use serde::{Deserialize, Serialize};
use stress_ref::capability::Profile;
use stress_ref::scene::{Methods, Scene, SolveMode};
use stress_ref::world::World;

/// Names `OracleConfig::named` accepts.
pub const ORACLE_CONFIGS: [&str; 1] = ["a12_realtime"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleConfig {
    pub name: String,
    /// Not yet confirmed as P0's pin (data recorded under it may need regenerating).
    pub provisional: bool,
    /// Where the configuration comes from.
    pub source: String,
    pub methods: Methods,
    pub solve_mode: SolveMode,
    pub courant_safety: f64,
    pub stress_courant_safety: Option<f64>,
    pub frame_dt: f64,
}

impl OracleConfig {
    pub fn named(name: &str) -> Result<OracleConfig, String> {
        match name {
            // `stress_rubble::scenes::realtime` (A12), the configuration P0 reproduced the
            // M5 truths with (35.034 / 6, 23.622 / 133); P0 has not closed its open
            // decisions (the layer contact for the energy ledger among them).
            "a12_realtime" => {
                let mut methods = Methods::established();
                methods.true_step = true;
                Ok(OracleConfig {
                    name: name.into(),
                    provisional: true,
                    source: "stress_rubble::scenes::realtime (A12); P0 step 1 (p0-feasibility-2026-10-20.md)".into(),
                    methods,
                    solve_mode: SolveMode::Adaptive,
                    courant_safety: 0.35,
                    stress_courant_safety: Some(0.5),
                    frame_dt: 1.0 / 60.0,
                })
            }
            _ => Err(format!("unknown oracle config {name:?} (known: {ORACLE_CONFIGS:?})")),
        }
    }

    /// Set the scene's solver switches to this configuration.
    pub fn apply(&self, scene: &mut Scene) {
        scene.sim.methods = self.methods;
        scene.sim.solve_mode = self.solve_mode;
        scene.sim.courant_safety = self.courant_safety;
        scene.sim.stress_courant_safety = self.stress_courant_safety;
        scene.sim.frame_dt = self.frame_dt;
    }

    /// Whether `scene` is set to this configuration.
    pub fn check(&self, scene: &Scene) -> Result<(), String> {
        let s = &scene.sim;
        if s.methods != self.methods || s.solve_mode != self.solve_mode || s.courant_safety != self.courant_safety
            || s.stress_courant_safety != self.stress_courant_safety || s.frame_dt != self.frame_dt
        {
            return Err(format!("scene {} is not under oracle config {}", scene.name, self.name));
        }
        Ok(())
    }
}

/// The baseline model of the same frame-start state: `blast_parity`'s quasi-static solve
/// (its scene switches and solver budget), as a template world to `World::adopt_model`.
pub const QS_BASELINE: &str = "blast_parity_cpu";

/// A `blast_parity` world of `scene` (the oracle-configured scene), for its bond model and
/// switches only.
pub fn qs_template(scene: &Scene) -> World {
    let mut s = scene.clone();
    Profile::BlastParity.apply_scene(&mut s);
    let mut w = World::new(&s);
    Profile::BlastParity.apply_config(&mut w.solver.config);
    w
}
