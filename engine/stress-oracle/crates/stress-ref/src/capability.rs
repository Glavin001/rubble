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
    /// `BlastParity` with routing and impact islands (the tiered coupling): the struck
    /// structure's impact region simulated before and through the engine's step, so a
    /// fracture separates the piece that breaks free within the frame, forward, with no
    /// rewind (findings/rewind-scaling.md, design C). Everything else is `BlastParity`'s.
    BlastParityRouted,
    /// The v0 stress mode, forward only (no re-simulation): `BlastParity`'s resident frame
    /// with no static solve.
    /// - A strike is predicted at the end of the frame before it (the routing check, and
    ///   stress-gpu punch.slang `threshold_predict`): every bond of the struck island whose
    ///   tensile capacity is below the force reaching it breaks, and the island splits
    ///   before the engine steps, so the engine meets the pieces (`predicts_strikes`).
    /// - After each step, a bond breaks when the contact force on either of its chunks
    ///   (sustained contacts, and each impact's Hertz peak force) exceeds its tensile
    ///   capacity (`SolverConfig::contact_threshold`): debris, settling.
    ///
    /// Deliberately simple and fast: it exercises the breaking, splitting and re-parenting
    /// path, not the stress model.
    Threshold,
}

/// The model features `BlastParity` turns off (`SimFeatures` names).
pub const BLAST_PARITY_FEATURES_OFF: [&str; 8] = ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"];

impl Profile {
    pub fn parse(name: &str) -> Option<Profile> {
        match name {
            "blast_parity" | "blast-parity" => Some(Profile::BlastParity),
            "blast_parity_routed" | "blast-parity-routed" => Some(Profile::BlastParityRouted),
            "threshold" => Some(Profile::Threshold),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::BlastParity => "blast_parity",
            Profile::BlastParityRouted => "blast_parity_routed",
            Profile::Threshold => "threshold",
        }
    }

    /// How packs import under the profile.
    pub fn import_options(self) -> ImportOptions {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold => ImportOptions { authored_strengths: true, authored_fatal: true, pack_masses: true, pack_modulus: true },
        }
    }

    /// The scene's switches: solve mode, model features and methods.
    pub fn apply_scene(self, scene: &mut Scene) {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold => {
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
            Profile::BlastParity | Profile::Threshold => false,
            Profile::BlastParityRouted => true,
        }
    }

    /// Whether a fracture under the engine's loads makes its whole structure island seeds
    /// for the next frames (the GPU world's `promote_fractures`; no effect without impact
    /// islands). `BlastParityRouted`: off, only an impactor's contact routes. Measured
    /// 2026-10-18 on the quick tier: with it, the settle's fractures put every cluster in
    /// islands for six frames of 65–121 s each.
    pub fn promote_fractures(self) -> bool {
        match self {
            Profile::BlastParity => true,
            // Threshold: a promoted structure's frames would leave the resident path.
            Profile::BlastParityRouted | Profile::Threshold => false,
        }
    }

    /// Whether a strike's breaks are predicted at the end of the frame before it, in its
    /// submission, and split before the engine steps (the GPU world's `predict_strikes`:
    /// the routing check and punch.slang `threshold_predict`, every frame resident).
    pub fn predicts_strikes(self) -> bool {
        matches!(self, Profile::Threshold)
    }

    /// Whether the profile runs the GPU-resident frame.
    pub fn resident(self) -> bool {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold => true,
        }
    }

    /// The rigid engine's policy: solver iterations per step and substeps a frame at most
    /// (`None`: the engine's own). `BlastParity`: vibe-land's PhysX, 4 iterations and one
    /// step a frame, no CCD substeps (your decision of 2026-10-18; fast thin pieces can
    /// tunnel).
    pub fn engine_policy(self) -> (Option<u32>, Option<u32>) {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold => (Some(4), Some(1)),
        }
    }

    /// The solver's per-tick behaviour.
    pub fn apply_config(self, config: &mut SolverConfig) {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted => {
                config.static_budget = Some((1, 16));
                config.static_unconverged = UnconvergedSolve::CommitIterate;
                config.static_passes = Some(2);
            }
            Profile::Threshold => {
                config.static_budget = Some((1, 16));
                config.static_unconverged = UnconvergedSolve::CommitIterate;
                config.static_passes = Some(1);
                config.contact_threshold = true;
            }
        }
    }
}
