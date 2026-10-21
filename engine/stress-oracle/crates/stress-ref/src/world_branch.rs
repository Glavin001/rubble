//! Branches for the learned impact-outcome recorder (P1 of
//! `docs/learned-stress-solver/learned-stress-solver-plan.md`): the same frame-start state
//! stepped under another model, so a dataset holds the oracle's outcome and the baselines'
//! outcomes (`blast_parity`'s quasi-static solve, the punch) on identical states.
//!
//! Research plumbing only: nothing on the established path calls these.

use super::*;

impl World {
    /// Take `template`'s model while keeping this world's state: its scene switches
    /// (`sim`), its solver configuration, its structures' static data (features, methods, bond
    /// materials) and mass damping, and every bond's construction-time model
    /// (section geometry, stiffness, strength, rebar, Weibull factor, damping). `template` is a world built
    /// from the same scene under another profile (for example `Profile::BlastParity`
    /// applied before `World::new`); bodies, poses, velocities, hidden state and damage
    /// stay this world's. Errors if the two worlds do not hold the same structures and bonds.
    pub fn adopt_model(&mut self, template: &World) -> Result<(), String> {
        let (a, b) = (&self.solver, &template.solver);
        if a.structures.len() != b.structures.len() || a.bonds.len() != b.bonds.len() {
            return Err("adopt_model: different structures".into());
        }
        for s in 0..a.bonds.len() {
            if a.structures[s].chunks.len() != b.structures[s].chunks.len() || a.bonds[s].len() != b.bonds[s].len() {
                return Err(format!("adopt_model: structure {s} differs in chunks or bonds"));
            }
            for (x, y) in a.bonds[s].iter().zip(&b.bonds[s]) {
                if x.source != y.source || x.level != y.level || x.geometry.a != y.geometry.a || x.geometry.b != y.geometry.b {
                    return Err(format!("adopt_model: structure {s} bond {} is not the template's", x.source));
                }
            }
        }
        if self.scene.sim.refine_utilization.is_some() || template.scene.sim.refine_utilization.is_some() {
            return Err("adopt_model: refinement changes the bond set".into());
        }
        let duration = self.scene.sim.duration;
        self.scene.sim = template.scene.sim.clone();
        self.scene.sim.duration = duration;
        self.solver.config = template.solver.config.clone();
        // The structures' static data carries each one's features, methods and bond
        // material; the geometry and masses are the same (checked above by the bond set).
        for (x, y) in self.solver.structures.iter().zip(&template.solver.structures) {
            if x.chunks.iter().zip(&y.chunks).any(|(a, b)| a.mass != b.mass || a.center != b.center) {
                return Err("adopt_model: the template's chunks differ".into());
            }
        }
        self.solver.structures = template.solver.structures.clone();
        self.solver.mass_damping = template.solver.mass_damping.clone();
        for s in 0..self.solver.bonds.len() {
            for (x, y) in self.solver.bonds[s].iter_mut().zip(&template.solver.bonds[s]) {
                // `blast_model` also changes the section moduli of the geometry.
                x.geometry = y.geometry.clone();
                x.stiffness = y.stiffness;
                x.strength = y.strength.clone();
                x.rebar = y.rebar;
                x.weibull = y.weibull;
                x.damping = y.damping;
            }
        }
        self.solver.refresh_true_steps();
        self.contact_dt = self.contact_stable_dt();
        self.chunk_contact_dts = if self.scene.sim.methods.multirate { self.chunk_contact_dts() } else { Vec::new() };
        Ok(())
    }
}
