//! Harness A of the learned-impact plan (2026-10-22): the standalone oracle with each routed
//! island's frame supplied through a provider. The island's full state crosses the boundary:
//! every per-chunk, per-bond and per-cluster field of the island's chunks, bonds and clusters
//! (`SolverIsland`), its impactors, and every contact history entry that involves its bodies.
//! The rest of the world is stepped natively; the routing guarantees the island and the
//! rest do not touch within the frame (their swept bounds are apart), and the frame's substep
//! schedule is the whole world's, so the host's native step of the rest is the oracle's.
//!
//! With the oracle as provider (`oracle_island_step`: the frame stepped natively on a copy,
//! the island's state taken from it) this is the standalone oracle bit for bit; a model
//! provider returns its own `IslandState`. The merge's destructurings are exhaustive, so every
//! field of `World` and `ReferenceSolver` has an explicit policy: a new field cannot cross
//! (or fail to cross) the boundary unnoticed.

use std::collections::HashSet;

use super::*;
use crate::solver::SolverIsland;

/// The island's full state at a frame's end.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IslandState {
    pub solver: SolverIsland,
    pub impactors: Vec<(usize, Impactor)>,
    loads: Vec<((usize, usize), [(Vec3, Vec3); 3])>,
    per_chunk: Vec<((usize, usize), (f64, f64, Vec<f64>))>,
    sticks: Vec<(ContactKey, Stick)>,
    pair_offsets: Vec<(ChunkPair, Vec<(f64, Vec3)>)>,
    ground_offsets: Vec<((u8, usize, usize), Vec<f64>)>,
}

/// An island: its chunks (by `(structure, chunk)`) and impactors (by index).
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct IslandMembers {
    pub chunks: HashSet<(usize, usize)>,
    pub impactors: HashSet<usize>,
}

impl IslandState {
    /// The island this state is of: its chunks and impactors.
    pub fn members(&self) -> IslandMembers {
        IslandMembers { chunks: self.solver.chunks.iter().map(|(k, _)| *k).collect(), impactors: self.impactors.iter().map(|(i, _)| *i).collect() }
    }
}

impl IslandMembers {
    fn key(&self, k: &ContactKey) -> bool {
        match *k {
            ContactKey::Ground(s, c) => self.chunks.contains(&(s, c)),
            ContactKey::ImpactorGround(i) => self.impactors.contains(&i),
            ContactKey::Impactor(s, c, i) => self.chunks.contains(&(s, c)) || self.impactors.contains(&i),
            ContactKey::Pair(sa, ca, sb, cb) => self.chunks.contains(&(sa, ca)) || self.chunks.contains(&(sb, cb)),
        }
    }
    fn pair(&self, k: &ChunkPair) -> bool {
        self.chunks.contains(&(k.0, k.1)) || self.chunks.contains(&(k.2, k.3))
    }
    fn ground(&self, k: &(u8, usize, usize)) -> bool {
        if k.0 == 0 {
            self.chunks.contains(&(k.1, k.2))
        } else {
            self.impactors.contains(&k.1)
        }
    }
}

fn load_at(l: &ChunkLoads, s: usize, c: usize) -> (Vec3, Vec3) {
    (l.force.get(s).and_then(|v| v.get(c)).copied().unwrap_or(Vec3::ZERO), l.torque.get(s).and_then(|v| v.get(c)).copied().unwrap_or(Vec3::ZERO))
}

fn set_load(l: &mut ChunkLoads, s: usize, c: usize, x: (Vec3, Vec3)) {
    if let Some(v) = l.force.get_mut(s).and_then(|v| v.get_mut(c)) {
        *v = x.0;
    }
    if let Some(v) = l.torque.get_mut(s).and_then(|v| v.get_mut(c)) {
        *v = x.1;
    }
}

impl World {
    /// The island the routing gives the coming frame (`predict_island_members`), as members.
    pub fn island_members(&mut self) -> IslandMembers {
        let fdt = self.scene.sim.frame_dt;
        let (clusters, impactors) = self.predict_island_members(fdt, &HashSet::new());
        let ids: HashSet<u64> = clusters.into_iter().collect();
        let mut m = IslandMembers { impactors: impactors.into_iter().collect(), ..Default::default() };
        for cl in self.solver.clusters.iter().filter(|c| ids.contains(&c.id)) {
            m.chunks.extend(cl.chunks.iter().map(|&c| (cl.structure, c)));
        }
        m
    }

    /// The island's full state in this world (a frame's end).
    pub fn extract_island(&self, m: &IslandMembers) -> IslandState {
        let mut keys: Vec<(usize, usize)> = m.chunks.iter().copied().collect();
        keys.sort_unstable();
        let mut imps: Vec<usize> = m.impactors.iter().copied().collect();
        imps.sort_unstable();
        let mut sticks: Vec<(ContactKey, Stick)> = self.sticks.iter().filter(|(k, _)| m.key(k)).map(|(k, v)| (*k, v.clone())).collect();
        sticks.sort_by_key(|(k, _)| format!("{k:?}"));
        let mut pair_offsets: Vec<(ChunkPair, Vec<(f64, Vec3)>)> = self.pair_offsets.iter().filter(|(k, _)| m.pair(k)).map(|(k, v)| (*k, v.clone())).collect();
        pair_offsets.sort_by_key(|(k, _)| *k);
        let mut ground_offsets: Vec<((u8, usize, usize), Vec<f64>)> = self.ground_offsets.iter().filter(|(k, _)| m.ground(k)).map(|(k, v)| (*k, v.clone())).collect();
        ground_offsets.sort_by_key(|(k, _)| *k);
        IslandState {
            solver: self.solver.extract_island(&m.chunks),
            impactors: imps.iter().map(|&i| (i, self.impactors[i].clone())).collect(),
            loads: keys.iter().map(|&(s, c)| ((s, c), [load_at(&self.loads, s, c), load_at(&self.frame_loads, s, c), load_at(&self.implicit_loads, s, c)])).collect(),
            per_chunk: keys
                .iter()
                .map(|&(s, c)| {
                    let hits = self.contact_hits.get(s).and_then(|v| v.get(c)).copied().unwrap_or(0.0);
                    let dt = self.chunk_contact_dts.get(s).and_then(|v| v.get(c)).copied().unwrap_or(0.0);
                    let ex = self.exposure.get(s).and_then(|v| v.get(c)).cloned().unwrap_or_default();
                    ((s, c), (hits, dt, ex))
                })
                .collect(),
            sticks,
            pair_offsets,
            ground_offsets,
        }
    }

    /// `host` (the frame stepped natively) with the island's state replaced by `st`; every
    /// field's policy explicit (the destructuring is exhaustive).
    pub fn merge_island(host: &World, st: &IslandState, m: &IslandMembers) -> World {
        let World {
            scene,
            solver,
            mut impactors,
            contact,
            frame,
            mut loads,
            mut frame_loads,
            mut implicit_loads,
            implicit_elapsed,
            events_done,
            probes,
            next_sample,
            sample_interval,
            initial_positions,
            mut exposure,
            exposure_version,
            topology_version,
            blast_caches,
            contact_dt,
            mut chunk_contact_dts,
            implicit_report,
            implicit_worst_residual,
            added_mass_fraction,
            static_unconverged_frames,
            mut contact_hits,
            implicit_contact_seen,
            mut pair_offsets,
            mut ground_offsets,
            mut sticks,
            neighbours,
            applied,
            coupling,
            island_frames,
            max_island_bodies,
            engine_body_frames,
            island_body_frames,
            redone_frames,
            impact_islands,
            qs_islands,
            tiered_fast_joins,
            debris_reach_routing,
            decoded_keeps_deformation,
            handed_ground_overlaps,
            routing_tiered,
            decoded_resets_joints,
            hand_back_after,
            handed_back,
            quiet_frames,
            fast_contact,
            window_forced,
            implicit_failures,
            learned_bins,
        } = host.clone();
        // per element: the island's from `st`
        let solver = ReferenceSolver::merge_island(&solver, &st.solver, &m.chunks);
        for (i, imp) in &st.impactors {
            impactors[*i] = imp.clone();
        }
        for &((s, c), [a, b, d]) in &st.loads {
            set_load(&mut loads, s, c, a);
            set_load(&mut frame_loads, s, c, b);
            set_load(&mut implicit_loads, s, c, d);
        }
        for ((s, c), (hits, dt, ex)) in &st.per_chunk {
            if let Some(v) = contact_hits.get_mut(*s).and_then(|v| v.get_mut(*c)) {
                *v = *hits;
            }
            if let Some(v) = chunk_contact_dts.get_mut(*s).and_then(|v| v.get_mut(*c)) {
                *v = *dt;
            }
            if let Some(v) = exposure.get_mut(*s).and_then(|v| v.get_mut(*c)) {
                *v = ex.clone();
            }
        }
        sticks.retain(|k, _| !m.key(k));
        sticks.extend(st.sticks.iter().map(|(k, v)| (*k, v.clone())));
        pair_offsets.retain(|k, _| !m.pair(k));
        pair_offsets.extend(st.pair_offsets.iter().map(|(k, v)| (*k, v.clone())));
        ground_offsets.retain(|k, _| !m.ground(k));
        ground_offsets.extend(st.ground_offsets.iter().map(|(k, v)| (*k, v.clone())));
        // the globals (time, frame, ledgers, logs, counters, caches, flags) are the host's
        World {
            scene,
            solver,
            impactors,
            contact,
            frame,
            loads,
            frame_loads,
            implicit_loads,
            implicit_elapsed,
            events_done,
            probes,
            next_sample,
            sample_interval,
            initial_positions,
            exposure,
            exposure_version,
            topology_version,
            blast_caches,
            contact_dt,
            chunk_contact_dts,
            implicit_report,
            implicit_worst_residual,
            added_mass_fraction,
            static_unconverged_frames,
            contact_hits,
            implicit_contact_seen,
            pair_offsets,
            ground_offsets,
            sticks,
            neighbours,
            applied,
            coupling,
            island_frames,
            max_island_bodies,
            engine_body_frames,
            island_body_frames,
            redone_frames,
            impact_islands,
            qs_islands,
            tiered_fast_joins,
            debris_reach_routing,
            decoded_keeps_deformation,
            handed_ground_overlaps,
            routing_tiered,
            decoded_resets_joints,
            hand_back_after,
            handed_back,
            quiet_frames,
            fast_contact,
            window_forced,
            implicit_failures,
            learned_bins,
        }
    }

    /// One frame of a standalone world in which the routed island's frame comes from
    /// `provider` (given the frame-start world and the island; it returns the island's state
    /// at the frame's end), the rest stepped natively. Returns the island (empty: an ordinary
    /// frame).
    pub fn step_frame_island_provider(&mut self, provider: &mut dyn FnMut(&World, &IslandMembers) -> Result<IslandState, String>) -> Result<IslandMembers, String> {
        if self.coupling.is_some() {
            return Err("step_frame_island_provider needs a standalone world".into());
        }
        let m = self.island_members();
        if m.chunks.is_empty() && m.impactors.is_empty() {
            self.solver.reset_peak_utilizations();
            self.step_frame();
            return Ok(m);
        }
        let st = provider(self, &m)?;
        // Each frame's peak utilisation starts afresh after the frame-start state the
        // provider reads (whose `peak_utilization` is the previous frame's: the recorder's
        // order, which the call builder's `log1p_peak_util_prev` reads). Diagnostic only: no
        // physics reads it.
        self.solver.reset_peak_utilizations();
        self.step_frame();
        // The host's own island (stepped natively, then replaced): where the provider's
        // topology differs from it, the host's structure-level caches were derived from the
        // wrong network and are refreshed. (The oracle's island step differs in nothing.)
        let own = self.extract_island(&m);
        let topology = |x: &IslandState| -> (Vec<(usize, usize, bool)>, Vec<(usize, Vec<usize>)>) {
            (x.solver.bonds.iter().map(|((s, b), r)| (*s, *b, r.alive)).collect(), x.solver.clusters.iter().map(|(_, c)| (c.structure, c.chunks.clone())).collect())
        };
        let changed = topology(&own) != topology(&st);
        *self = World::merge_island(self, &st, &m);
        if changed {
            self.topology_version += 1;
            self.solver.unchanged.clear();
            self.solver.refresh_true_steps();
        }
        Ok(m)
    }
}

/// The oracle's island step: the frame stepped natively on a copy of the frame-start world,
/// the island's state taken from it (harness A's reference and ceiling).
pub fn oracle_island_step(start: &World, m: &IslandMembers) -> Result<IslandState, String> {
    let mut w = start.clone();
    w.solver.reset_peak_utilizations();
    w.step_frame();
    Ok(w.extract_island(m))
}
