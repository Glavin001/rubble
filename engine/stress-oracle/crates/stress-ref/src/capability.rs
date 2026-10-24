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
    /// `Threshold`'s fracture world (stress-gpu `GpuFracture`), its strikes predicted as
    /// `Threshold`'s, with the contact-force test replaced by a load path
    /// ([`LoadModel::LoadPath`], engine/docs/stress-threshold-gpu.md "Load path"): gravity
    /// and the engine's contact forces and torques carried through the bonds by a
    /// warm-started relaxation of the bond graph's quasi-static equilibrium (a fixed
    /// number of sweeps a frame, as Blast's stress solver iterates), each bond checked
    /// against its tension, shear, compression (and buckling) capacities and its bending.
    /// The bonds are the physical model's (their geometric sections and stiffnesses), not
    /// Blast's (`Methods::blast_model` off). Runs on the fracture world only (stress-rubble
    /// `ThresholdWorld`).
    LoadPath,
    /// `Threshold`'s fracture world and its contact-force test, with the strike's flat
    /// Hertz-force test replaced by an analytic local impact model
    /// ([`StrikeModel::Analytic`], engine/docs/stress-threshold-gpu.md "Local impact"):
    /// Hertz contact, an energy ballistic limit with a plug cone, the wave's back-face
    /// spall, and strain-rate strength (DIF), decided before the engine steps. The strike
    /// also resolves the shot's contact (the engine steps once a frame with no swept
    /// collision): through, the shot gives up the ballistic limit's energy and leaves with
    /// its plug, their momentum kept; not through, it comes to rest on the face, its
    /// momentum into the wall. Runs on the fracture world only (stress-rubble
    /// `ThresholdWorld`).
    ImpactLocal,
}

/// How the fracture world (stress-gpu `GpuFracture`) turns a frame's loads into breaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadModel {
    /// A bond breaks when the contact force on either of its chunks beats its tensile
    /// capacity (fracture.slang `fr_breaks`).
    #[default]
    Threshold,
    /// The load path (fracture.slang `fr_lp_*`): `sweeps` Chebyshev-accelerated block
    /// Jacobi sweeps a frame on the bond graph's equilibrium, warm-started (an even count).
    LoadPath { sweeps: u32 },
}

/// How the fracture world decides a predicted strike's breaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StrikeModel {
    /// Every bond whose share of the Hertz force beats its tensile capacity
    /// (fracture.slang `fr_strike_breaks`).
    #[default]
    Hertz,
    /// The analytic local impact model (fracture.slang `fr_il_*`, the host's
    /// `impact_local`). `fracture_work`: a perforating shot gives up the ballistic limit's
    /// energy before the step.
    Analytic { fracture_work: bool },
}

/// The load path's sweeps a frame in the `LoadPath` profile.
pub const LOAD_PATH_SWEEPS: u32 = 16;

/// The model features `BlastParity` turns off (`SimFeatures` names).
pub const BLAST_PARITY_FEATURES_OFF: [&str; 8] = ["softening", "weibull", "crack_contact", "rate_effects", "static_fatigue", "buckling", "rebar", "damping"];

impl Profile {
    pub fn parse(name: &str) -> Option<Profile> {
        match name {
            "blast_parity" | "blast-parity" => Some(Profile::BlastParity),
            "blast_parity_routed" | "blast-parity-routed" => Some(Profile::BlastParityRouted),
            "threshold" => Some(Profile::Threshold),
            "load_path" | "load-path" => Some(Profile::LoadPath),
            "impact_local" | "impact-local" => Some(Profile::ImpactLocal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::BlastParity => "blast_parity",
            Profile::BlastParityRouted => "blast_parity_routed",
            Profile::Threshold => "threshold",
            Profile::LoadPath => "load_path",
            Profile::ImpactLocal => "impact_local",
        }
    }

    /// How packs import under the profile.
    pub fn import_options(self) -> ImportOptions {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => ImportOptions { authored_strengths: true, authored_fatal: true, pack_masses: true, pack_modulus: true },
        }
    }

    /// The scene's switches: solve mode, model features and methods.
    pub fn apply_scene(self, scene: &mut Scene) {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => {
                scene.sim.solve_mode = SolveMode::QuasiStatic;
                for f in BLAST_PARITY_FEATURES_OFF {
                    scene.sim.features.set(f, false).expect("a SimFeatures name");
                }
                // Blast's bond model (its capped section gains and rotational `k l^2`), but
                // for the load path: a load path is the physical quasi-static solve, the
                // bonds' geometric sections and stiffnesses (statics.rs's, which beam theory
                // checks). Under Blast's model a 0.1 m bond's section modulus is 20 times
                // its geometric one, so a cantilever never breaks at its root.
                scene.sim.methods.blast_model = self != Profile::LoadPath;
                // In the scene, so the world's setup solves already take it.
                scene.sim.static_tolerance = Some(1e-3);
            }
        }
    }

    /// Whether the profile runs impact islands (the coupling's routing).
    pub fn impact_islands(self) -> bool {
        match self {
            Profile::BlastParity | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => false,
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
            Profile::BlastParityRouted | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => false,
        }
    }

    /// Whether a strike's breaks are predicted at the end of the frame before it, in its
    /// submission, and split before the engine steps (the GPU world's `predict_strikes`:
    /// the routing check and punch.slang `threshold_predict`, every frame resident).
    pub fn predicts_strikes(self) -> bool {
        matches!(self, Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal)
    }

    /// Whether the profile runs on the fracture world alone (stress-rubble
    /// `ThresholdWorld`): its models exist only there.
    pub fn fracture_world_only(self) -> bool {
        matches!(self, Profile::LoadPath | Profile::ImpactLocal)
    }

    /// The fracture world's models under the profile, then the environment's overrides
    /// (native runs; a browser takes the profile's): `STRESS_FRACTURE_LOAD=threshold|load_path`,
    /// `STRESS_LOAD_PATH_SWEEPS=N` (even), `STRESS_FRACTURE_STRIKE=hertz|analytic`,
    /// `STRESS_IMPACT_FRACTURE_WORK=0|1`.
    pub fn fracture_models(self) -> Result<(LoadModel, StrikeModel), String> {
        let (mut load, mut strike) = match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold => (LoadModel::Threshold, StrikeModel::Hertz),
            Profile::LoadPath => (LoadModel::LoadPath { sweeps: LOAD_PATH_SWEEPS }, StrikeModel::Hertz),
            Profile::ImpactLocal => (LoadModel::Threshold, StrikeModel::Analytic { fracture_work: true }),
        };
        let var = |name: &str| std::env::var(name).ok();
        let sweeps = match var("STRESS_LOAD_PATH_SWEEPS") {
            Some(v) => v.parse::<u32>().map_err(|_| format!("STRESS_LOAD_PATH_SWEEPS={v}: not a count"))?,
            None => match load {
                LoadModel::LoadPath { sweeps } => sweeps,
                LoadModel::Threshold => LOAD_PATH_SWEEPS,
            },
        };
        if sweeps == 0 || sweeps % 2 != 0 {
            return Err(format!("load path: {sweeps} sweeps; it takes an even count of at least 2"));
        }
        match var("STRESS_FRACTURE_LOAD").as_deref() {
            None => {}
            Some("threshold") => load = LoadModel::Threshold,
            Some("load_path") => load = LoadModel::LoadPath { sweeps },
            Some(v) => return Err(format!("STRESS_FRACTURE_LOAD={v}: threshold or load_path")),
        }
        if let LoadModel::LoadPath { .. } = load {
            load = LoadModel::LoadPath { sweeps };
        }
        let work = match var("STRESS_IMPACT_FRACTURE_WORK").as_deref() {
            None => None,
            Some("0") => Some(false),
            Some("1") => Some(true),
            Some(v) => return Err(format!("STRESS_IMPACT_FRACTURE_WORK={v}: 0 or 1")),
        };
        match var("STRESS_FRACTURE_STRIKE").as_deref() {
            None => {}
            Some("hertz") => strike = StrikeModel::Hertz,
            Some("analytic") => strike = StrikeModel::Analytic { fracture_work: true },
            Some(v) => return Err(format!("STRESS_FRACTURE_STRIKE={v}: hertz or analytic")),
        }
        if let (StrikeModel::Analytic { .. }, Some(w)) = (strike, work) {
            strike = StrikeModel::Analytic { fracture_work: w };
        }
        Ok((load, strike))
    }

    /// Whether the profile runs the GPU-resident frame.
    pub fn resident(self) -> bool {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => true,
        }
    }

    /// The rigid engine's policy: solver iterations per step and substeps a frame at most
    /// (`None`: the engine's own). `BlastParity`: vibe-land's PhysX, 4 iterations and one
    /// step a frame, no CCD substeps (your decision of 2026-10-18; fast thin pieces can
    /// tunnel).
    pub fn engine_policy(self) -> (Option<u32>, Option<u32>) {
        match self {
            Profile::BlastParity | Profile::BlastParityRouted | Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => (Some(4), Some(1)),
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
            Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal => {
                config.static_budget = Some((1, 16));
                config.static_unconverged = UnconvergedSolve::CommitIterate;
                config.static_passes = Some(1);
                config.contact_threshold = true;
            }
        }
    }
}
