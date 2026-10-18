//! Capability profiles: named sets of the solver's capability switches
//! (engine/docs/stress-baseline-goal.md, rules 2-3).
//!
//! Every capability beyond Blast's quasi-static solve stays in the code as a switch:
//! `SimFeatures` (softening, rate effects, fatigue, ...), `Methods` (the oracle's model
//! switches), `SolverConfig` (static budgets, cascade passes, tiered impact islands), and
//! the GPU world's own options. A profile sets all of them at once, so a run says which
//! profile it is instead of listing switches. A capability is turned on in a profile only
//! once it keeps that profile's targets; nothing is removed to make a profile.

use crate::scene::{Scene, SolveMode};
use crate::scene_pack::ImportOptions;
use crate::solver::{SolverConfig, UnconvergedSolve};

/// A named capability profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// The baseline: Blast's quasi-static GPU stress solver as PhysX's native destruction
    /// stage runs it in vibe-land (engine/docs/vibe-land-physx-review.md):
    /// - quasi-static mode, no impact islands, every body the engine's;
    /// - none of the model features Blast lacks (softening, Weibull strengths, crack
    ///   contact, rate effects, static fatigue, buckling, rebar, damping);
    /// - Blast's model: least-work isotropic springs with its floors, rotational `k l^2`,
    ///   the bending gain capped at 3 (`Methods::blast_model`);
    /// - the pack's fatal limits as strengths, its node masses and supports;
    /// - per tick: 1 Newton step x 16 conjugate-gradient iterations, tolerance 1e-3,
    ///   unconverged answers committed, at most 2 cascade passes;
    /// - the GPU-resident frame.
    BlastParity,
}

/// The model features `BlastParity` turns off (`SimFeatures` names).
pub const BLAST_PARITY_FEATURES_OFF: [&str; 8] = ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"];

impl Profile {
    pub fn parse(name: &str) -> Option<Profile> {
        match name {
            "blast_parity" | "blast-parity" => Some(Profile::BlastParity),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::BlastParity => "blast_parity",
        }
    }

    /// How packs import under the profile.
    pub fn import_options(self) -> ImportOptions {
        match self {
            Profile::BlastParity => ImportOptions { authored_strengths: true, authored_fatal: true, pack_masses: true, pack_modulus: true },
        }
    }

    /// The scene's switches: solve mode, model features and methods.
    pub fn apply_scene(self, scene: &mut Scene) {
        match self {
            Profile::BlastParity => {
                scene.sim.solve_mode = SolveMode::QuasiStatic;
                for f in BLAST_PARITY_FEATURES_OFF {
                    scene.sim.features.set(f, false).expect("a SimFeatures name");
                }
                scene.sim.methods.blast_model = true;
                // In the scene, so the world's setup solves already take it.
                scene.sim.static_tolerance = Some(1e-3);
            }
        }
    }

    /// Whether the profile runs impact islands (the coupling's routing).
    pub fn impact_islands(self) -> bool {
        match self {
            Profile::BlastParity => false,
        }
    }

    /// Whether the profile runs the GPU-resident frame.
    pub fn resident(self) -> bool {
        match self {
            Profile::BlastParity => true,
        }
    }

    /// The solver's per-tick behaviour.
    pub fn apply_config(self, config: &mut SolverConfig) {
        match self {
            Profile::BlastParity => {
                config.static_budget = Some((1, 16));
                config.static_unconverged = UnconvergedSolve::CommitIterate;
                config.static_passes = Some(2);
            }
        }
    }
}
