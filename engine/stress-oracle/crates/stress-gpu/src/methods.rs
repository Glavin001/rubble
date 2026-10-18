//! How the GPU honours each of stress-ref's method switches (`Methods::NAMES`).
//!
//! Every switch is classified (a test fails on an unclassified name), and
//! `world::unsupported` rejects every scene that enables a switch the GPU would not
//! honour (`unhonoured`), so a method can never be silently dropped. Each entry cites the
//! oracle's code and the GPU's (`ref/` is `crates/stress-ref/src/`, `gpu/` this crate's
//! `src/`, `sl/` its `shaders/`); each was checked against the code, not the design notes.

use stress_ref::scene::{Methods, SolveMode};

/// What a switch needs before the oracle acts on it at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prerequisite {
    /// Another method switch.
    Method(&'static str),
    /// A solve mode.
    Mode(SolveMode),
}

/// How the GPU honours a switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Honoured {
    /// The GPU runs the method itself (a kernel, or host code of this crate).
    Implemented,
    /// The host runs the oracle's own code on the mirror (`GpuSolver::mirror`).
    Delegated,
    /// The oracle acts on the switch only with the prerequisite, which the GPU does not
    /// run yet: inert without it. With it, the switch is not honoured.
    InertWithout(Prerequisite),
    /// Not on the GPU: a scene enabling it is rejected.
    Rejected,
}

/// One switch: how the GPU honours it, where the oracle applies it, where the GPU does.
#[derive(Clone, Copy, Debug)]
pub struct Classification {
    pub name: &'static str,
    pub honoured: Honoured,
    pub oracle: &'static str,
    pub gpu: &'static str,
}

use Honoured::*;
use Prerequisite::*;

/// Every switch of `Methods::NAMES`.
pub const TABLE: [Classification; 18] = [
    Classification {
        name: "layer_contact",
        honoured: Rejected,
        oracle: "ref/world.rs apply_layer_contacts, hand_over_split_joints, layer_force, pair_layer_contact; ref/polytope.rs; ref/solver.rs split_cluster (joints handed over)",
        gpu: "the penalty contact only (gpu/contacts.rs, sl/world.slang contact_forces): LAYER_CONTACT_PORT.md",
    },
    Classification {
        name: "scaled_step_bound",
        honoured: Delegated,
        oracle: "ref/solver.rs chunk_frequencies, chunk_stable_dts_over (mass-normalised Gershgorin rows); ref/world.rs substep_dt (1/dt^2 = 1/dt_s^2 + 1/dt_c^2)",
        gpu: "the mirror's stable_dt / chunk_stable_dts and the shell's substep_dt (gpu/world.rs step_frame, stress_dt; gpu/coupled.rs); the coupled frame's combination in gpu/world.rs combined_dt is the oracle's formula",
    },
    Classification {
        name: "exact_joint_patch",
        honoured: Implemented,
        oracle: "ref/structure.rs (strength.exact_patch), ref/joint.rs",
        gpu: "gpu/joint.rs FLAG_EXACT_PATCH; sl/joint.slang",
    },
    Classification {
        name: "patch_after_damage",
        honoured: Implemented,
        oracle: "ref/structure.rs (strength.patch_after_damage), ref/joint.rs",
        gpu: "gpu/joint.rs FLAG_PATCH_AFTER_DAMAGE; sl/joint.slang",
    },
    Classification {
        name: "exact_rate_filter",
        honoured: Implemented,
        oracle: "ref/structure.rs (strength.exact_rate_filter), ref/joint.rs",
        gpu: "gpu/joint.rs FLAG_EXACT_RATE_FILTER; sl/joint.slang (expm1 gain)",
    },
    Classification {
        name: "frame_contact_step",
        honoured: InertWithout(Method("layer_contact")),
        oracle: "ref/world.rs step_frame: frame_contact_dt only when layer_contact is on too",
        gpu: "nothing: GpuWorld never calls World::step_frame, so with layer contact ported it must be ported too",
    },
    Classification {
        name: "exact_frame_transfer",
        honoured: Delegated,
        oracle: "ref/solver.rs split_cluster (exact velocities, fold_rigid_drift(exact))",
        gpu: "every split runs ReferenceSolver::split_cluster on the mirror (gpu/solver.rs, gpu/world.rs)",
    },
    Classification {
        name: "contact_crushing",
        honoured: InertWithout(Method("layer_contact")),
        oracle: "ref/world.rs pair_layer_contact only",
        gpu: "nothing (no layer contact)",
    },
    Classification {
        name: "tangent_statics",
        honoured: Delegated,
        oracle: "ref/statics.rs equilibrate: the secant for SECANT_STEPS, then the globalised tangent re-solve (residual_floor), then the secant's remaining steps",
        gpu: "gpu/solver.rs equilibrate: the GPU secant (sl/world.slang island_statics) for at most SECANT_STEPS; a cluster it leaves unconverged is solved by the mirror's equilibrate_or_keep and uploaded",
    },
    Classification {
        name: "true_step",
        honoured: Delegated,
        oracle: "ref/solver.rs stable_dt, refresh_true_steps, structure_true_step",
        gpu: "the mirror's stable_dt (its true_steps come with it from World::new); coupled frames: own_dashpot_step, a sharper certificate of the same scheme (gpu/stability.rs certified_damped_step)",
    },
    Classification {
        name: "lattice_contact_step",
        honoured: InertWithout(Method("layer_contact")),
        oracle: "ref/world.rs contact_stable_dt: the lattice bound only with layer_contact",
        gpu: "the shell's contact_dt (standalone); coupled island frames: gpu/world.rs plan_contact_dt calls engaged_contact_dt, the penalty network's bound",
    },
    Classification {
        name: "implicit_contact_steps",
        honoured: InertWithout(Mode(SolveMode::Implicit)),
        oracle: "ref/world.rs accumulate_implicit (implicit mode only)",
        gpu: "nothing: the implicit mode is rejected",
    },
    Classification {
        name: "implicit_contact_windows",
        honoured: InertWithout(Mode(SolveMode::Implicit)),
        oracle: "ref/world.rs implicit_stress_dt, the implicit frame's window choice (implicit mode only)",
        gpu: "nothing: the implicit mode is rejected",
    },
    Classification {
        name: "implicit_failure_windows",
        honoured: InertWithout(Mode(SolveMode::Implicit)),
        oracle: "ref/world.rs implicit_stress_dt, record_implicit_failures, the frame redone with failing clusters windowed (implicit mode only)",
        gpu: "nothing: the implicit mode is rejected",
    },
    Classification {
        name: "corotational_bonds",
        honoured: Rejected,
        oracle: "ref/bond.rs CoRotation, corotation_with, bond_loads; ref/solver.rs (finite turns per substep, exact composition, mass damping of free clusters)",
        gpu: "small-strain bonds only (sl/joint.slang; the hidden rotation is the additive th, gpu/contacts.rs)",
    },
    Classification {
        name: "rayleigh_damping",
        honoured: Rejected,
        oracle: "ref/world.rs fit_mass_damping; ref/solver.rs mass_damping (explicit), ref/implicit.rs (Newmark residual and tangent)",
        gpu: "the bond dashpots only, unscaled",
    },
    Classification {
        name: "multirate",
        honoured: Rejected,
        oracle: "ref/solver.rs open_multirate_window, bond_rates, firing_bonds (bond rates and impulses); ref/world.rs open_multirate_window (per frame, chunk contact bounds)",
        gpu: "every bond at every substep; the design is findings/multirate.md",
    },
    Classification {
        name: "blast_model",
        honoured: Delegated,
        oracle: "ref/structure.rs (Blast's least-work springs sqrt(EA/L) with its area and length floors, rotational k l^2, section moduli A / gain)",
        gpu: "the bond tables the host builds from the oracle's structure (stress-layout), which every kernel reads",
    },
];

/// A switch's classification.
pub fn classify(name: &str) -> Option<&'static Classification> {
    TABLE.iter().find(|c| c.name == name)
}

/// A switch's setting, by name.
pub fn switch(methods: &Methods, name: &str) -> bool {
    serde_json::to_value(methods).ok().and_then(|v| v.get(name).and_then(|b| b.as_bool())).unwrap_or_else(|| panic!("unknown method switch {name}"))
}

/// The switches a scene with these methods, in this mode, enables but the GPU would not
/// honour: each rejected one that is on, and each inert one that is on with its
/// prerequisite (the oracle would act on it).
pub fn unhonoured(methods: &Methods, mode: SolveMode) -> Vec<&'static Classification> {
    TABLE
        .iter()
        .filter(|c| {
            switch(methods, c.name)
                && match c.honoured {
                    Implemented | Delegated => false,
                    Rejected => true,
                    InertWithout(Method(other)) => switch(methods, other),
                    InertWithout(Mode(m)) => mode == m,
                }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every switch the oracle has is classified, once, and nothing else is.
    #[test]
    fn every_method_is_classified() {
        let unclassified: Vec<&str> = Methods::NAMES.iter().copied().filter(|n| classify(n).is_none()).collect();
        assert!(unclassified.is_empty(), "unclassified method switches (add them to methods::TABLE): {unclassified:?}");
        for c in &TABLE {
            assert!(Methods::NAMES.contains(&c.name), "{} is not a method switch", c.name);
            assert_eq!(TABLE.iter().filter(|o| o.name == c.name).count(), 1, "{} classified twice", c.name);
            if let InertWithout(Method(other)) = c.honoured {
                assert!(Methods::NAMES.contains(&other), "{}: prerequisite {other} is not a method switch", c.name);
            }
        }
    }

    /// The switches the GPU does not honour: none of the established methods; of the proven
    /// ones the layer contact (and the switches that act with it), and the co-rotational
    /// bonds and Rayleigh damping (small-strain bonds and bond dashpots on the GPU).
    #[test]
    fn unhonoured_switches() {
        let names = |m: &Methods, mode| unhonoured(m, mode).iter().map(|c| c.name).collect::<Vec<_>>();
        assert!(names(&Methods::established(), SolveMode::Explicit).is_empty());
        assert_eq!(names(&Methods::proven(), SolveMode::Adaptive), ["layer_contact", "contact_crushing", "lattice_contact_step", "corotational_bonds", "rayleigh_damping"]);
        let rigid = Methods { layer_contact: false, corotational_bonds: false, rayleigh_damping: false, ..Methods::proven() };
        for mode in [SolveMode::Explicit, SolveMode::Adaptive, SolveMode::QuasiStatic] {
            assert!(names(&rigid, mode).is_empty(), "{mode:?}: {:?}", names(&rigid, mode));
        }
        assert_eq!(names(&rigid, SolveMode::Implicit), ["implicit_contact_steps", "implicit_contact_windows", "implicit_failure_windows"]);
        // A switch inert without its prerequisite is not honoured with it.
        let frame = Methods { frame_contact_step: true, ..Methods::established() };
        assert!(names(&frame, SolveMode::Explicit).is_empty());
        assert_eq!(names(&Methods { layer_contact: true, ..frame }, SolveMode::Explicit), ["layer_contact", "frame_contact_step"]);
    }

    /// `world::unsupported` rejects a scene exactly when it enables a switch the GPU would
    /// not honour (or for a reason outside the methods: the implicit mode), and names
    /// every such switch: from both method sets, each switch flipped, in every mode.
    #[test]
    fn unsupported_rejects_every_unhonoured_switch() {
        let mut base = stress_ref::builders::catalog().into_iter().next().expect("a catalogue scene");
        base.sim.refine_utilization = None;
        for start in [Methods::established(), Methods::proven()] {
            for mode in [SolveMode::Explicit, SolveMode::Adaptive, SolveMode::QuasiStatic, SolveMode::Implicit] {
                for name in Methods::NAMES {
                    for on in [false, true] {
                        let mut s = base.clone();
                        s.sim.methods = start;
                        s.sim.methods.set(name, on).unwrap();
                        s.sim.solve_mode = mode;
                        let why = crate::world::unsupported(&s);
                        let bad = unhonoured(&s.sim.methods, mode);
                        let case = format!("{} with {name}={on}, {mode:?}", if start.is_proven() { "proven" } else { "established" });
                        if mode == SolveMode::Implicit {
                            assert!(why.is_some(), "{case}: the implicit mode runs");
                            continue;
                        }
                        assert_eq!(why.is_some(), !bad.is_empty(), "{case}: unsupported {why:?}, unhonoured {:?}", bad.iter().map(|c| c.name).collect::<Vec<_>>());
                        for c in bad {
                            let why = why.as_deref().unwrap_or("");
                            assert!(why.contains(c.name), "{case}: the rejection does not name {}: {why}", c.name);
                        }
                    }
                }
            }
        }
    }
}
