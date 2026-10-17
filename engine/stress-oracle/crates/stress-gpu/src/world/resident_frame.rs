//! The resident quasi-static frame (docs/stress-fracture-gpu-resident.md, phases 2-5 for
//! the quasi-static coupling): the engine steps without reading back, its contact rows
//! become manifolds, filtered loads and each cluster's static loads on the GPU
//! (`ingest`, `resident`), and the static solves and their damage commit run there too
//! (`equilibrate_resident_with`, `commit_resident`). The host keeps the frame's control:
//! a cascade pass's verdicts (converged, damaged, disconnected) and the engine's
//! overflow status are its only reads. A split syncs the mirror (the engine's motion
//! included), splits on the host and rebuilds the tables.
//!
//! Taken instead of the fast quasi-static frame when the world was built with
//! `with_resident_engine`; the same conditions (`fast_frame`), quiet clusters off (their
//! choice reads the host's engine loads, which this frame does not form).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use stress_ref::engine::BodyKey;
use stress_ref::scene::Scene;
use stress_ref::solver::Activity;
use stress_ref::statics::StaticOptions;
use stress_ref::world::World;

use super::GpuWorld;
use crate::gpu::Gpu;
use crate::ingest::EngineIngest;
use crate::resident::{EngineFrame, ResidentEngine, ResidentQs, SharedEngine};

pub(super) struct ResidentState {
    ingest: EngineIngest,
    /// Substeps the ingest holds.
    substeps: u32,
    qs: ResidentQs,
    engine_version: u64,
    layout_generation: u64,
}

impl GpuWorld {
    /// The world coupled to `engine` with the resident quasi-static frame available
    /// (`with_engine` otherwise): the engine shares the stress solver's device.
    pub fn with_resident_engine<T: ResidentEngine + 'static>(gpu: &Gpu, scene: &Scene, engine: T) -> Result<GpuWorld, String> {
        let shared = Rc::new(RefCell::new(engine));
        let mut w = GpuWorld::with_engine(gpu, scene, Box::new(SharedEngine(shared.clone())))?;
        w.solver.quiet_quasi_static = false;
        w.resident_engine = Some(shared);
        Ok(w)
    }

    pub(super) fn resident_now(&self) -> bool {
        self.resident_engine.is_some() && self.fast_qs_conditions() && !self.shell.qs_islands
    }

    /// The tables of the current layouts (rebuilt when the engine's colliders or the
    /// solver's layout changed, or the frame needs more substeps).
    fn resident_tables(&mut self, gpu: &Gpu, substeps: u32) {
        let engine = self.resident_engine.clone().expect("resident engine");
        let e = engine.borrow();
        let version = e.layout_version();
        let generation = self.solver.layout_generation;
        if let Some(r) = &self.resident {
            if r.engine_version == version && r.layout_generation == generation && r.substeps >= substeps {
                return;
            }
        }
        let m = &self.solver.mirror;
        let chunk_index: HashMap<(usize, usize), u32> = self.solver.gpu_chunks().iter().enumerate().map(|(k, &sc)| (sc, k as u32)).collect();
        let by_id: HashMap<u64, usize> = m.clusters.iter().enumerate().map(|(ci, cl)| (cl.id, ci)).collect();
        let ground_modulus = self.scene.ground.as_ref().map(|g| self.scene.material(&g.material).youngs_modulus).unwrap_or(3e10);
        let impactors = &self.solver.impactors;
        // world_coupling.rs `engine_contacts_to_impulses`.
        let other = |key: Option<BodyKey>| -> [f32; 3] {
            let p = match key {
                Some(BodyKey::Impactor(i)) => {
                    let imp = &impactors[i];
                    (imp.mass, imp.material.youngs_modulus, World::impactor_radius(imp))
                }
                Some(BodyKey::Cluster(id)) => match by_id.get(&id) {
                    Some(&ci) => {
                        let cl = &m.clusters[ci];
                        let ch = &m.structures[cl.structure].chunks[cl.chunks[0]];
                        (if cl.anchored { f64::INFINITY } else { cl.mass }, ch.youngs_modulus, ch.half_extents.min_elem())
                    }
                    None => (f64::INFINITY, ground_modulus, 1.0),
                },
                None => (f64::INFINITY, ground_modulus, 1.0),
            };
            [p.0 as f32, p.1 as f32, p.2 as f32]
        };
        let table = e.collider_table(&|s, c| chunk_index.get(&(s, c)).copied(), &other);
        let frames: Vec<Option<EngineFrame>> = m.clusters.iter().map(|cl| if cl.anchored { None } else { e.engine_frame(BodyKey::Cluster(cl.id)) }).collect();
        let (rows, bodies) = e.engine_capacity();
        let chunks = chunk_index.len().max(1) as u32;
        let dt = self.scene.sim.frame_dt;
        match self.resident.as_mut() {
            Some(r) if r.substeps >= substeps && r.ingest_chunks() == chunks => {
                r.ingest.set_colliders(gpu, &table);
                r.qs.rebuild(gpu, &self.solver, &frames, dt);
                r.engine_version = version;
                r.layout_generation = generation;
            }
            Some(r) => {
                let cap = substeps.max(r.substeps).max(1).next_power_of_two();
                r.ingest = EngineIngest::new(gpu, rows, cap, bodies, chunks, &table);
                r.substeps = cap;
                r.qs.rebuild(gpu, &self.solver, &frames, dt);
                r.engine_version = version;
                r.layout_generation = generation;
            }
            None => {
                let cap = substeps.max(1).next_power_of_two();
                self.resident = Some(ResidentState {
                    ingest: EngineIngest::new(gpu, rows, cap, bodies, chunks, &table),
                    substeps: cap,
                    qs: ResidentQs::new(gpu, &self.solver, &frames, dt),
                    engine_version: version,
                    layout_generation: generation,
                });
            }
        }
    }

    /// The mirror current: the solver's state and, after resident frames, the engine's
    /// motion of the bodies it drives (which those frames leave on the GPU).
    pub fn sync(&mut self, gpu: &Gpu) {
        self.solver.sync(gpu);
        if self.resident_frames > 0 {
            self.resident_sync_motion();
        }
    }

    /// The engine's motion of the driven bodies into the mirror (before a split, and
    /// whenever the mirror is wanted current).
    fn resident_sync_motion(&mut self) {
        let engine = self.resident_engine.clone().expect("resident engine");
        let motion: HashMap<BodyKey, stress_ref::engine::BodyMotion> = engine.borrow().motion().into_iter().map(|m| (m.key, m)).collect();
        for cl in self.solver.mirror.clusters.iter_mut().filter(|c| c.driven && !c.anchored) {
            if let Some(m) = motion.get(&BodyKey::Cluster(cl.id)) {
                (cl.pose, cl.velocity, cl.angular_velocity) = (m.pose, m.velocity, m.angular_velocity);
            }
        }
        for (i, imp) in self.solver.impactors.iter_mut().enumerate() {
            if let Some(m) = motion.get(&BodyKey::Impactor(i)) {
                (imp.pose, imp.velocity, imp.angular_velocity) = (m.pose, m.velocity, m.angular_velocity);
            }
        }
    }

    pub(super) fn resident_frame(&mut self, gpu: &Gpu) -> Result<(), String> {
        let t_prepare = std::time::Instant::now();
        let fdt = self.scene.sim.frame_dt;
        let engine = self.resident_engine.clone().expect("resident engine");
        for cl in &mut self.solver.mirror.clusters {
            cl.driven = true;
        }
        let n = engine.borrow_mut().resident_substeps(fdt);
        self.resident_tables(gpu, n);
        {
            let r = self.resident.as_ref().expect("resident tables");
            engine.borrow_mut().step_resident(gpu, fdt, n, &r.ingest)?;
        }
        self.prepare_seconds += t_prepare.elapsed().as_secs_f64();
        let tail = std::time::Instant::now();
        let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
        let max = StaticOptions::default().max_cascade.max(1);
        let mut unconverged = false;
        let mut split = false;
        for pass in 0..max {
            self.resident_tables(gpu, n);
            let m = &self.solver.mirror;
            let clusters: Vec<usize> = (0..m.clusters.len()).filter(|&ci| !m.clusters[ci].bonds.is_empty()).collect();
            let t_pass = std::time::Instant::now();
            let r = self.resident.take().expect("resident tables");
            let converged = self.solver.equilibrate_resident_with(gpu, &clusters, None, &|s, enc| {
                r.qs.encode_poses(gpu, enc, &r.ingest, s);
                if pass == 0 {
                    r.qs.encode_filter(gpu, enc, &r.ingest, s);
                }
                r.qs.encode_static_loads(gpu, enc, &r.ingest, s);
            });
            self.resident = Some(r);
            let t_solve = t_pass.elapsed().as_secs_f64() * 1e3;
            unconverged |= converged.iter().any(|&c| !c);
            let solved: Vec<usize> = clusters.iter().zip(&converged).filter(|(_, &c)| c).map(|(&ci, _)| ci).collect();
            for &ci in &clusters {
                self.solver.mirror.clusters[ci].activity = Activity::Settled;
            }
            let (damaged, disconnected) = self.solver.commit_resident(gpu, &solved, if pass == 0 { fdt } else { 0.0 });
            if trace {
                eprintln!("resident qs pass {pass}: {} clusters ({} converged), solve {t_solve:.1} ms, commit {:.1} ms, {} damaged, {} disconnected", clusters.len(), solved.len(), t_pass.elapsed().as_secs_f64() * 1e3 - t_solve, damaged.len(), disconnected.len());
            }
            if !disconnected.is_empty() {
                // A split: the mirror (state and the engine's motion), split and tables.
                self.solver.sync(gpu);
                self.resident_sync_motion();
                let mut pending = disconnected.clone();
                pending.sort_unstable();
                for &ci in pending.iter().rev() {
                    self.solver.mirror.split_cluster(ci);
                }
                if let Some(l) = self.solver.loads.as_mut() {
                    l.topology_version += 1;
                }
                self.solver.rebuild(gpu, 0.0)?;
                split = true;
                continue;
            }
            if damaged.is_empty() {
                break;
            }
        }
        if unconverged {
            self.shell.static_unconverged_frames += 1;
        }
        self.tail_seconds += tail.elapsed().as_secs_f64();
        let t_back = std::time::Instant::now();
        engine.borrow_mut().resident_finish()?;
        self.resident.as_ref().expect("resident tables").qs.check(gpu)?;
        if split {
            // New clusters get their engine bodies (at the engine's end of the frame).
            for cl in &mut self.solver.mirror.clusters {
                cl.driven = true;
            }
            self.with_shell_mut(|w| w.coupled_write_back(&[]));
        }
        self.write_back_seconds += t_back.elapsed().as_secs_f64();
        if trace {
            let speeds: Vec<String> = engine.borrow().motion().iter().filter(|m| matches!(m.key, BodyKey::Impactor(_))).map(|m| format!("{:.3}", m.velocity.norm())).collect();
            eprintln!("resident frame {}: {n} engine substeps, impactor speeds {speeds:?}", self.frame);
        }
        self.solver.mirror.time += fdt;
        self.frame += 1;
        self.resident_frames += 1;
        Ok(())
    }
}

impl ResidentState {
    fn ingest_chunks(&self) -> u32 {
        self.ingest.chunk_count()
    }
}
