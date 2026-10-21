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

// ------------------------------------------------------------------ decoded state

/// One call's decoded outcome (`learned-impact-decoded-state/1`,
/// `learned-impact/DECODED_STATE.md` on the P2 branch): frame-end motion of every chunk and
/// impactor of the routed island, the bonds that break in the frame and the damage of the
/// rest, keyed by stable keys. Chunk and bond keys are `(structure << 32) | index` (the
/// structure's index in the scene; a bond's index in `ReferenceSolver::bonds[structure]`);
/// an impactor's key is its index in `World::impactors`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecodedState {
    pub schema_version: String,
    #[serde(default)]
    pub calls_schema: String,
    #[serde(default)]
    pub event_id: String,
    #[serde(default)]
    pub call_index: i64,
    pub frame: i64,
    pub island: i64,
    #[serde(default)]
    pub meta: serde_json::Value,
    #[serde(default)]
    pub flags: serde_json::Value,
    pub chunks: Vec<DecodedBody>,
    pub impactors: Vec<DecodedBody>,
    pub bonds_broken: Vec<DecodedBreak>,
    pub bond_damage: Vec<DecodedDamage>,
}

/// A body's frame-end motion: velocity, spin, and its centre's displacement and rotation
/// (world-axis rotation vector) over the frame.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecodedBody {
    pub key: i64,
    #[serde(default)]
    pub anchored: bool,
    pub v: [f64; 3],
    pub w: [f64; 3],
    pub dx: [f64; 3],
    pub drot: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_perforate: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecodedBreak {
    pub key: i64,
    pub bin: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecodedDamage {
    pub key: i64,
    pub damage: f64,
}

pub const DECODED_STATE_VERSION: &str = "learned-impact-decoded-state/1";

/// The stable key of chunk or bond `index` of structure `s`.
pub fn stable_key(s: usize, index: usize) -> i64 {
    ((s as i64) << 32) | index as i64
}

fn unkey(k: i64) -> (usize, usize) {
    ((k >> 32) as usize, (k & 0xffff_ffff) as usize)
}

/// What one call covers: the routed island's chunks, live bonds and impactors (keys).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct IslandKeys {
    pub chunks: Vec<i64>,
    pub bonds: Vec<i64>,
    pub impactors: Vec<i64>,
}

impl World {
    /// The bodies the current routing holds in islands (not driven): the call's keys. One
    /// call covers all of them (the harness's islands are not split into components yet).
    pub fn routed_island_keys(&self) -> IslandKeys {
        let mut k = IslandKeys::default();
        for cl in self.solver.clusters.iter().filter(|c| !c.driven) {
            let s = cl.structure;
            k.chunks.extend(cl.chunks.iter().map(|&c| stable_key(s, c)));
            k.bonds.extend(cl.bonds.iter().filter(|&&b| self.solver.bonds[s][b].connected()).map(|&b| stable_key(s, b)));
        }
        k.impactors.extend(self.impactors.iter().enumerate().filter(|(_, i)| !i.driven).map(|(i, _)| i as i64));
        k.chunks.sort_unstable();
        k.bonds.sort_unstable();
        k
    }

    /// A chunk's world orientation (its cluster's rotation and its hidden rotation).
    fn chunk_orientation(&self, s: usize, c: usize) -> Quat {
        let cl = &self.solver.clusters[self.solver.chunks[s][c].cluster];
        (cl.pose.rotation * Quat::from_rotation_vector(self.solver.chunks[s][c].th)).normalized()
    }

    /// The oracle's own outcome of the frame for the island `keys` (the perfect model):
    /// `model` is this world's standalone copy stepped one frame from this state.
    pub fn decoded_from_oracle(&self, model: &World, keys: &IslandKeys, frame: i64, island: i64) -> DecodedState {
        let fdt = self.scene.sim.frame_dt;
        let t0 = self.solver.time;
        let body = |a: (Vec3, Quat), b: (Vec3, Quat), v: Vec3, w: Vec3, key: i64, anchored: bool| DecodedBody {
            key,
            anchored,
            v: v.to_array(),
            w: w.to_array(),
            dx: (b.0 - a.0).to_array(),
            drot: (b.1 * a.1.conjugate()).normalized().to_rotation_vector().to_array(),
            p_perforate: None,
        };
        let chunks = keys
            .chunks
            .iter()
            .map(|&k| {
                let (s, c) = unkey(k);
                let (v, w) = model.solver.chunk_velocity(s, c);
                let anchored = self.solver.clusters[self.solver.chunks[s][c].cluster].anchored;
                body((self.solver.chunk_position(s, c), self.chunk_orientation(s, c)), (model.solver.chunk_position(s, c), model.chunk_orientation(s, c)), v, w, k, anchored)
            })
            .collect();
        let impactors = keys
            .impactors
            .iter()
            .map(|&k| {
                let (a, b) = (&self.impactors[k as usize], &model.impactors[k as usize]);
                body((a.pose.position, a.pose.rotation), (b.pose.position, b.pose.rotation), b.velocity, b.angular_velocity, k, false)
            })
            .collect();
        // Each broken bond's time bin (1..8 over the frame) from its event.
        let mut when = std::collections::HashMap::new();
        for e in &model.solver.events[self.solver.events.len().min(model.solver.events.len())..] {
            if let crate::solver::SolverEvent::Broken { time, structure, bond, .. } = e {
                when.entry(stable_key(*structure, *bond)).or_insert(*time);
            }
        }
        let (mut bonds_broken, mut bond_damage) = (Vec::new(), Vec::new());
        for &k in &keys.bonds {
            let (s, b) = unkey(k);
            let mb = &model.solver.bonds[s][b];
            if mb.connected() {
                bond_damage.push(DecodedDamage { key: k, damage: mb.joint.damage });
            } else {
                let t = when.get(&k).copied().unwrap_or(t0 + fdt);
                let bin = (((t - t0) / fdt * 8.0).ceil() as i64).clamp(1, 8) as u32;
                bonds_broken.push(DecodedBreak { key: k, bin });
            }
        }
        DecodedState {
            schema_version: DECODED_STATE_VERSION.into(),
            calls_schema: "learned-impact-outcome/1".into(),
            event_id: self.scene.name.clone(),
            call_index: frame,
            frame,
            island,
            meta: serde_json::json!({"model": "oracle"}),
            flags: serde_json::json!({"free": !keys.chunks.iter().any(|&k| { let (s, c) = unkey(k); self.solver.clusters[self.solver.chunks[s][c].cluster].anchored }), "energy_bound_active": false, "energy_bound_infeasible": false}),
            chunks,
            impactors,
            bonds_broken,
            bond_damage,
        }
    }

    /// Apply a decoded state to this world's routed island at window end: the breaks (in
    /// time-bin order), the survivors' damage, then every island cluster's rigid motion
    /// (hidden field cleared): its centre moved to the mass-weighted mean of its chunks'
    /// frame-end positions, turned by the mass-weighted mean chunk rotation, and its
    /// velocity and spin the momentum-exact fit to the chunks' decoded velocities; then the
    /// impactors'. The caller has already stepped the rest of the world (the engine, with
    /// the island frozen) and advances the clocks.
    pub fn apply_decoded(&mut self, d: &DecodedState) -> Result<(), String> {
        if d.schema_version != DECODED_STATE_VERSION {
            return Err(format!("decoded state version {:?}", d.schema_version));
        }
        // Frame-start chunk placements, before any split moves chunks between clusters.
        let mut target: std::collections::HashMap<(usize, usize), (Vec3, Vec3, Vec3, Vec3)> = std::collections::HashMap::new();
        for b in &d.chunks {
            let (s, c) = unkey(b.key);
            if s >= self.solver.chunks.len() || c >= self.solver.chunks[s].len() || !self.solver.chunks[s][c].active {
                return Err(format!("chunk key {} is not an active chunk", b.key));
            }
            let x1 = self.solver.chunk_position(s, c) + Vec3::from_array(b.dx);
            target.insert((s, c), (x1, Vec3::from_array(b.drot), Vec3::from_array(b.v), Vec3::from_array(b.w)));
        }
        let mut breaks: Vec<&DecodedBreak> = d.bonds_broken.iter().collect();
        breaks.sort_by_key(|b| (b.bin, b.key));
        for b in breaks {
            let (s, bi) = unkey(b.key);
            if s >= self.solver.bonds.len() || bi >= self.solver.bonds[s].len() {
                return Err(format!("bond key {} out of range", b.key));
            }
            self.solver.break_bonds(&[(s, bi)]);
        }
        for x in &d.bond_damage {
            let (s, bi) = unkey(x.key);
            let b = self.solver.bonds.get_mut(s).and_then(|v| v.get_mut(bi)).ok_or_else(|| format!("bond key {} out of range", x.key))?;
            b.joint.damage = b.joint.damage.max(x.damage);
        }
        // Every cluster holding a decoded chunk.
        let mut clusters: Vec<usize> = target.keys().map(|&(s, c)| self.solver.chunks[s][c].cluster).collect();
        clusters.sort_unstable();
        clusters.dedup();
        for &ci in &clusters {
            if self.solver.clusters[ci].anchored {
                for c in self.solver.clusters[ci].chunks.clone() {
                    let s = self.solver.clusters[ci].structure;
                    let cs = &mut self.solver.chunks[s][c];
                    (cs.u, cs.th, cs.v, cs.w) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                }
                continue;
            }
            let s = self.solver.clusters[ci].structure;
            let chunks = self.solver.clusters[ci].chunks.clone();
            let (mut m, mut x, mut rot) = (0.0, Vec3::ZERO, Vec3::ZERO);
            for &c in &chunks {
                let mc = self.solver.structures[s].chunks[c].mass;
                let (x1, dr) = target.get(&(s, c)).map(|t| (t.0, t.1)).unwrap_or_else(|| (self.solver.chunk_position(s, c), Vec3::ZERO));
                m += mc;
                x += x1 * mc;
                rot += dr * mc;
            }
            let (com1, turn) = (x * (1.0 / m), Quat::from_rotation_vector(rot * (1.0 / m)));
            let cl = &mut self.solver.clusters[ci];
            cl.pose.rotation = (turn * cl.pose.rotation).normalized();
            cl.pose.position = com1 - cl.pose.rotation.rotate(cl.com);
            for &c in &chunks {
                let cs = &mut self.solver.chunks[s][c];
                (cs.u, cs.th, cs.v, cs.w) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
            }
        }
        let vel = |s: usize, c: usize| target.get(&(s, c)).map(|t| (t.2, t.3));
        let sv = &mut self.solver;
        for &ci in &clusters {
            if sv.clusters[ci].anchored {
                continue;
            }
            let s = sv.clusters[ci].structure;
            let chunks = sv.clusters[ci].chunks.clone();
            let com = sv.clusters[ci].com_world();
            let r = sv.clusters[ci].rotation();
            let (mut p, mut l) = (Vec3::ZERO, Vec3::ZERO);
            for &c in &chunks {
                let (v, w) = vel(s, c).unwrap_or_else(|| sv.chunk_velocity(s, c));
                let xc = sv.chunk_position(s, c);
                let ch = &sv.structures[s].chunks[c];
                p += v * ch.mass;
                l += (xc - com).cross(v) * ch.mass + (r * ch.inertia * r.transpose()) * w;
            }
            let cl = &mut sv.clusters[ci];
            let iw = r * cl.inertia * r.transpose();
            cl.velocity = p * (1.0 / cl.mass);
            cl.angular_velocity = iw.inverse().ok_or("singular inertia")? * l;
        }
        for b in &d.impactors {
            let imp = self.impactors.get_mut(b.key as usize).ok_or_else(|| format!("impactor key {} out of range", b.key))?;
            imp.pose.position += Vec3::from_array(b.dx);
            imp.pose.rotation = (Quat::from_rotation_vector(Vec3::from_array(b.drot)) * imp.pose.rotation).normalized();
            imp.velocity = Vec3::from_array(b.v);
            imp.angular_velocity = Vec3::from_array(b.w);
        }
        Ok(())
    }

    /// One frame whose routed island's outcome comes from `provider` (a model, or the
    /// oracle): the routing selects the island, the engine steps every other body with the
    /// island frozen (its contacts with the island are the call's), the decoded state is
    /// applied at window end, the clocks advance and the engine takes the new bodies and
    /// motion. With no island this is an ordinary coupled frame. Returns the island's keys
    /// (empty: no call).
    pub fn step_frame_with_provider(&mut self, provider: &mut dyn FnMut(&World, &IslandKeys) -> Result<DecodedState, String>) -> Result<IslandKeys, String> {
        let fdt = self.scene.sim.frame_dt;
        let start = self.clone();
        if self.predict_islands(fdt, &std::collections::HashSet::new()) == 0 {
            self.step_frame();
            return Ok(IslandKeys::default());
        }
        let f = self.coupled_prepare(&std::collections::HashSet::new());
        let keys = self.routed_island_keys();
        // The provider sees the frame-start state (the engine's step moved only driven bodies,
        // whose frame-start motion `start` holds).
        let mut view = start;
        for (cl, now) in view.solver.clusters.iter_mut().zip(&self.solver.clusters) {
            cl.driven = now.driven;
        }
        for (imp, now) in view.impactors.iter_mut().zip(&self.impactors) {
            imp.driven = now.driven;
        }
        let d = provider(&view, &keys)?;
        self.set_driven_motion(1.0);
        self.apply_decoded(&d)?;
        self.solver.time += fdt;
        self.frame += 1;
        self.island_frames += 1;
        self.coupled_write_back(&f.frozen);
        Ok(keys)
    }
}
