//! Hooks for the perfect-model ceiling of the learned impact-outcome plan (P0 step 3,
//! `docs/learned-stress-solver/learned-stress-solver-plan.md`): the boundary a learned
//! call would have, tested with the oracle's own outcome in place of a model's.
//!
//! A learned call replaces an island frame's explicit solve with its outcome: which bonds
//! broke, each chunk's velocity and the impactor's at the frame's end. The engine then
//! holds only what such a call can hand it: rigid bodies, their velocities and the bonds'
//! damage, none of the oracle's hidden deformation. These hooks let a harness run an
//! engine-coupled world, take the oracle's frame as the "model", and apply its outcome
//! the ways the plan's variants ask for (reduced or full state, window-end or pre-impact
//! poses, the frame's contact left to the engine or not).
//!
//! Research plumbing only: nothing on the established path calls these.

use super::*;
use crate::engine::BodyKey;

impl World {
    /// This world without its engine and with every body its own (the standalone oracle
    /// from this state): what a perfect model of the next frame would compute.
    pub fn standalone_copy(&self) -> World {
        let mut w = self.clone();
        w.coupling = None;
        for cl in &mut w.solver.clusters {
            cl.driven = false;
        }
        for imp in &mut w.impactors {
            imp.driven = false;
        }
        w
    }

    /// Return to `saved` (a clone taken at a frame start, with `engine_save` called then),
    /// keeping this world's engine, rewound to its saved motion.
    pub fn restore_keeping_engine(&mut self, saved: World) {
        let engine = std::mem::replace(&mut self.coupling.as_mut().expect("coupled").engine, EngineCell(None));
        *self = saved;
        let c = self.coupling.as_mut().expect("coupled");
        c.engine = engine;
        c.engine.restore();
    }

    /// Take `model`'s whole state (a `standalone_copy` stepped on), keeping this world's
    /// engine and coupling, and give the engine the new bodies and motion: the outcome
    /// at the model's frame end ("window-end poses").
    pub fn adopt_keeping_engine(&mut self, model: World) {
        let coupling = self.coupling.take();
        let (islands, island_frames, redone) = (self.impact_islands, self.island_frames, self.redone_frames);
        *self = model;
        self.coupling = coupling;
        (self.impact_islands, self.island_frames, self.redone_frames) = (islands, island_frames + 1, redone);
        self.push_motion_to_engine();
    }

    /// Fold every cluster's hidden motion into its rigid motion and clear the hidden
    /// field: the state an engine of rigid bodies holds. Linear and angular momentum of
    /// each cluster are kept exactly (computed from the chunks' world velocities before
    /// the field is cleared); the hidden deformation and its elastic energy are dropped.
    pub fn reduce_to_rigid(&mut self) {
        let sv = &mut self.solver;
        for ci in 0..sv.clusters.len() {
            let s = sv.clusters[ci].structure;
            let chunks = sv.clusters[ci].chunks.clone();
            let (mut p, mut l) = (Vec3::ZERO, Vec3::ZERO);
            let com = sv.clusters[ci].com_world();
            let rot = sv.clusters[ci].rotation();
            for &c in &chunks {
                let (v, w) = sv.chunk_velocity(s, c);
                let x = sv.chunk_position(s, c);
                let ch = &sv.structures[s].chunks[c];
                p += v * ch.mass;
                l += (x - com).cross(v) * ch.mass + (rot * ch.inertia * rot.transpose()) * w;
            }
            for &c in &chunks {
                let cs = &mut sv.chunks[s][c];
                (cs.u, cs.th, cs.v, cs.w) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            }
            let cl = &mut sv.clusters[ci];
            if cl.anchored {
                continue;
            }
            let iw = rot * cl.inertia * rot.transpose();
            cl.velocity = p * (1.0 / cl.mass);
            cl.angular_velocity = iw.inverse().expect("invertible inertia") * (l - (cl.com_world() - com).cross(p));
        }
    }

    /// Every free cluster's rigid motion fitted to given chunk world velocities (`None`:
    /// the chunk's current one): the momentum-conserving mean, as the split gives a child.
    /// The hidden field is cleared.
    pub fn set_rigid_from_chunks(&mut self, velocity: &dyn Fn(usize, usize) -> Option<(Vec3, Vec3)>) {
        let sv = &mut self.solver;
        for ci in 0..sv.clusters.len() {
            if sv.clusters[ci].anchored {
                continue;
            }
            let s = sv.clusters[ci].structure;
            let chunks = sv.clusters[ci].chunks.clone();
            let com = sv.clusters[ci].com_world();
            let rot = sv.clusters[ci].rotation();
            let (mut p, mut l) = (Vec3::ZERO, Vec3::ZERO);
            for &c in &chunks {
                let (v, w) = velocity(s, c).unwrap_or_else(|| sv.chunk_velocity(s, c));
                let x = sv.chunk_position(s, c);
                let ch = &sv.structures[s].chunks[c];
                p += v * ch.mass;
                l += (x - com).cross(v) * ch.mass + (rot * ch.inertia * rot.transpose()) * w;
            }
            for &c in &chunks {
                let cs = &mut sv.chunks[s][c];
                (cs.u, cs.th, cs.v, cs.w) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            }
            let cl = &mut sv.clusters[ci];
            let iw = rot * cl.inertia * rot.transpose();
            cl.velocity = p * (1.0 / cl.mass);
            cl.angular_velocity = iw.inverse().expect("invertible inertia") * (l - (cl.com_world() - com).cross(p));
        }
    }

    /// Advance every free cluster and impactor ballistically over `dt` (gravity, no
    /// contact) and the clocks with them: a frame whose contacts the call already
    /// resolved ("contact filtered"). The engine is then given the new motion.
    pub fn advance_ballistic(&mut self, dt: f64) {
        let g = self.solver.config.gravity;
        for cl in self.solver.clusters.iter_mut().filter(|c| !c.anchored) {
            let com0 = cl.com_world();
            let rot1 = cl.pose.rotation.integrate(cl.angular_velocity, dt);
            let com1 = com0 + cl.velocity * dt + g * (0.5 * dt * dt);
            cl.pose.rotation = rot1;
            cl.pose.position = com1 - rot1.rotate(cl.com);
            cl.velocity += g * dt;
        }
        for imp in &mut self.impactors {
            imp.pose.position += imp.velocity * dt + g * (0.5 * dt * dt);
            imp.pose.rotation = imp.pose.rotation.integrate(imp.angular_velocity, dt);
            imp.velocity += g * dt;
        }
        self.solver.time += dt;
        self.frame += 1;
        self.push_motion_to_engine();
    }

    /// Give the engine every body's current motion (new bodies are created with it).
    pub fn push_motion_to_engine(&mut self) {
        let keys: Vec<BodyKey> =
            self.solver.clusters.iter().map(|c| BodyKey::Cluster(c.id)).chain((0..self.impactors.len()).map(BodyKey::Impactor)).collect();
        self.coupled_write_back(&keys);
    }

    /// One coupled frame with no impact islands: the engine steps every body, contacts
    /// included ("contact not filtered").
    pub fn step_frame_engine_only(&mut self) {
        let islands = std::mem::replace(&mut self.impact_islands, false);
        self.step_frame();
        self.impact_islands = islands;
    }
}
