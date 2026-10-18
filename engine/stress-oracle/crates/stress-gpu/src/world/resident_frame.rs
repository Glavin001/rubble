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

/// Cascade passes encoded per submission (the cascade rarely needs more; a longer one
/// takes further windows, one status read each).
const CASCADE_WINDOW: usize = 4;

pub(super) struct ResidentState {
    ingest: EngineIngest,
    /// Substeps the ingest holds.
    substeps: u32,
    qs: ResidentQs,
    engine_version: u64,
    layout_generation: u64,
    /// The layout's last change was a split the GPU laid out (its entries moved there).
    gpu_laid_out: bool,
    /// `STRESS_GPU_LAYOUT=1`: the collider table kept on the GPU after its splits
    /// (colliders.rs), and whether the last split refreshed it.
    colliders: Option<crate::colliders::ColliderRefresh>,
    colliders_refreshed: bool,
}

impl GpuWorld {
    /// The world coupled to `engine` with the resident quasi-static frame available
    /// (`with_engine` otherwise): the engine shares the stress solver's device.
    pub fn with_resident_engine<T: ResidentEngine + 'static>(gpu: &Gpu, scene: &Scene, engine: T) -> Result<GpuWorld, String> {
        let shared = Rc::new(RefCell::new(engine));
        let mut w = GpuWorld::with_engine(gpu, scene, Box::new(SharedEngine(shared.clone())))?;
        w.solver.quiet_quasi_static = false;
        w.resident_engine = Some(shared);
        // Room for splits laid out in place (`GpuSolver::split_resident`).
        // A shattering impact makes hundreds of fragments at once; an island record is
        // 400 B, so room for thousands costs little and keeps splits in place.
        w.solver.island_headroom = (2 * w.solver.mirror.clusters.len()).max(4096);
        w.solver.state_on_gpu = true;
        w.solver.sync(gpu);
        w.solver.rebuild(gpu, 0.0)?;
        Ok(w)
    }

    pub(super) fn resident_now(&self) -> bool {
        self.resident_engine.is_some() && self.fast_qs_conditions() && !self.shell.qs_islands
    }

    /// The engine's collider table as the coupling sees it (`ColliderInfo`), against the
    /// solver's current layout and mirror; and the static geometry's modulus.
    fn host_collider_table(&self, e: &dyn ResidentEngine) -> (Vec<crate::ingest::ColliderInfo>, f64) {
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
        (e.collider_table(&|s, c| chunk_index.get(&(s, c)).copied(), &other), ground_modulus)
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
        let (table, ground_modulus) = self.host_collider_table(&*e);
        let gpu_colliders = std::env::var("STRESS_GPU_LAYOUT").is_ok_and(|v| v == "1");
        let collider_cap = e.collider_capacity();
        let frames: Vec<Option<EngineFrame>> = m.clusters.iter().map(|cl| if cl.anchored { None } else { e.engine_frame(BodyKey::Cluster(cl.id)) }).collect();
        let (rows, bodies) = e.engine_capacity();
        let chunks = self.solver.gpu_chunks().len().max(1) as u32;
        let dt = self.scene.sim.frame_dt;
        match self.resident.as_mut() {
            // A split the GPU laid out: the tables' entries already moved; only the
            // engine's colliders are new.
            Some(r) if r.gpu_laid_out && r.layout_generation == generation && r.substeps >= substeps && r.ingest_chunks() == chunks => {
                r.ingest.set_colliders(gpu, &table);
                r.engine_version = version;
                r.gpu_laid_out = false;
            }
            // Only the engine's bodies changed (an impactor added): the tables' islands and
            // their bodies are as they were; only the colliders are new.
            Some(r) if r.layout_generation == generation && r.substeps >= substeps && r.ingest_chunks() == chunks => {
                r.ingest.set_colliders(gpu, &table);
                r.engine_version = version;
            }
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
                    gpu_laid_out: false,
                    colliders: None,
                    colliders_refreshed: false,
                });
            }
        }
        // `STRESS_GPU_LAYOUT=1`: the GPU keeps the table after its splits (colliders.rs):
        // its keys from this table, room for the engine's colliders.
        if gpu_colliders {
            let r = self.resident.as_mut().expect("resident tables");
            if r.colliders.is_none() {
                let props: Vec<Vec<[f32; 2]>> = m.structures.iter().map(|st| st.chunks.iter().map(|ch| [ch.youngs_modulus as f32, ch.half_extents.min_elem() as f32]).collect()).collect();
                r.colliders = Some(crate::colliders::ColliderRefresh::new(gpu, &props, collider_cap, ground_modulus as f32));
            }
            r.ingest.set_colliders_with_room(gpu, &table, collider_cap);
            r.colliders.as_mut().expect("collider refresh").set_keys(gpu, &table, self.solver.gpu_chunks(), collider_cap);
        }
    }

    /// After the engine adopted a split whose collider table the GPU refreshed: the table
    /// is current (`STRESS_GPU_COLLIDER_CHECK=1`: compared with the host's).
    fn colliders_adopted(&mut self, gpu: &Gpu) {
        let engine = self.resident_engine.clone().expect("resident engine");
        let e = engine.borrow();
        let Some(r) = self.resident.as_ref() else { return };
        if !r.colliders_refreshed {
            return;
        }
        if std::env::var("STRESS_GPU_COLLIDER_CHECK").is_ok_and(|v| v == "1") {
            let (table, _) = self.host_collider_table(&*e);
            let n = table.len() as u64 * std::mem::size_of::<crate::ingest::ColliderInfo>() as u64;
            let read = gpu.read_ranges(&[(&r.ingest.collider_info, 0, n)]);
            let gpu_table: Vec<crate::ingest::ColliderInfo> = bytemuck::pod_collect_to_vec(&read[0]);
            self.solver.collider_check.compare(&gpu_table, &table);
        }
        let r = self.resident.as_mut().expect("resident tables");
        r.engine_version = e.layout_version();
        r.gpu_laid_out = false;
        r.colliders_refreshed = false;
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
        let t_substeps = std::time::Instant::now();
        let n = engine.borrow_mut().resident_substeps(fdt);
        let phases = std::env::var("STRESS_RESIDENT_PHASES").is_ok();
        if phases {
            eprintln!("resident phases: substep count {n}: {:.2} ms", t_substeps.elapsed().as_secs_f64() * 1e3);
        }
        let t_tables = std::time::Instant::now();
        self.resident_tables(gpu, n);
        if phases {
            eprintln!("resident phases: tables {:.2} ms", t_tables.elapsed().as_secs_f64() * 1e3);
        }
        let t_engine = std::time::Instant::now();
        {
            let r = self.resident.as_ref().expect("resident tables");
            engine.borrow_mut().step_resident(gpu, fdt, n, &r.ingest)?;
        }
        if phases {
            let encode = t_engine.elapsed().as_secs_f64() * 1e3;
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("resident phases: engine {n} substeps + ingest: encode {encode:.1} ms, done {:.1} ms", t_engine.elapsed().as_secs_f64() * 1e3);
        }
        self.prepare_seconds += t_prepare.elapsed().as_secs_f64();
        let tail = std::time::Instant::now();
        let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
        let max = self.solver.mirror.config.static_passes.unwrap_or(StaticOptions::default().max_cascade).max(1);
        let mut split = false;
        // M3's check: GPU-made bodies of this frame's splits, compared after the host makes its own.
        let mut pending_spawn: Vec<(Vec<u64>, Vec<usize>, Option<Vec<crate::gpu_split::GpuBody>>)> = Vec::new();
        let mut first = 0;
        let mut unconverged = 0;
        loop {
            // The cascade from pass `first` on, under GPU control; then the frame's one
            // read: the cascade's status, the engine's counters, the pulse slots' status.
            self.resident_tables(gpu, n);
            let t_pass = std::time::Instant::now();
            let r = self.resident.take().expect("resident tables");
            let window = CASCADE_WINDOW.min(max - first);
            let run = self.solver.resident_cascade(gpu, fdt, first, window, &|s, enc, pass| {
                r.qs.encode_poses(gpu, enc, &r.ingest, s);
                if pass == 0 {
                    r.qs.encode_filter(gpu, enc, &r.ingest, s);
                }
                r.qs.encode_static_loads(gpu, enc, &r.ingest, s);
            });
            if phases {
                let encode = t_pass.elapsed().as_secs_f64() * 1e3;
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                eprintln!("resident phases: cascade encode {encode:.1} ms, done {:.1} ms", t_pass.elapsed().as_secs_f64() * 1e3);
            }
            let e = engine.borrow();
            let status = e.resident_status();
            let engine_parts = status.len();
            let mut parts = vec![run.range()];
            parts.extend(status);
            parts.push(r.qs.status_range());
            let read = gpu.read_ranges(&parts);
            drop(e);
            self.resident = Some(r);
            let out = run.outcome(&read[0]);
            if std::env::var("STRESS_GPU_SPLIT_HASH").is_ok() {
                use std::hash::{Hash, Hasher};
                let (hs, hb) = self.solver.state_hashes(gpu);
                eprintln!("frame hash: pass {first} stress state {hs:x} bonds {hb:x} ingest {:x}", { let mut h = std::collections::hash_map::DefaultHasher::new(); read[1].hash(&mut h); h.finish() });
            }
            engine.borrow_mut().resident_finish(&read[1..1 + engine_parts])?;
            self.resident.as_ref().expect("resident tables").qs.check(&read[1 + engine_parts])?;
            unconverged += out.unconverged;
            for cl in self.solver.mirror.clusters.iter_mut().filter(|c| !c.bonds.is_empty()) {
                cl.activity = Activity::Settled;
            }
            if std::env::var("STRESS_GPU_QS_RESULTS").is_ok() {
                for (i, r) in self.solver.statics_results(gpu).iter().enumerate() {
                    if out.solved.contains(&i) || r.0 != 0.0 {
                        let cl = &self.solver.mirror.clusters[i];
                        let name = self.scene.bodies.get(cl.structure).map_or("?", |b| b.name.as_str());
                        eprintln!("island {i}: residual {:.3e}, newton {}, cg {}, converged {}, {} chunks, {} bonds, anchored {}, structure {} {name}", r.0, r.1, r.2, r.3, r.4, r.5, cl.anchored, cl.structure);
                    }
                }
            }
            if trace {
                eprintln!(
                    "resident qs passes {first}..: {} run, {} of {} solved converged, {} unconverged in all, skipped {} repeats and {} fixed points, certified {}, {:.1} ms, {} disconnected",
                    out.passes,
                    out.converged.len(),
                    out.solved.len(),
                    out.unconverged,
                    out.skipped_unconverged,
                    out.skipped_fixed,
                    out.certified,
                    t_pass.elapsed().as_secs_f64() * 1e3,
                    out.disconnected.len()
                );
            }
            let Some(at) = out.disconnected_at else {
                // The window ran out with damage still growing: the next window.
                if out.passes == window && out.damaged && first + window < max {
                    first += window;
                    continue;
                }
                break;
            };
            // A split: the mirror (state and the engine's motion), split and tables; the
            // cascade goes on from the next pass.
            let t_split = std::time::Instant::now();
            // The engine's motion into the mirror (a host split takes the parents' frames
            // from it; the GPU's takes them from the engine's buffer).
            if !std::env::var("STRESS_GPU_LAYOUT").is_ok_and(|v| v == "1") {
                self.resident_sync_motion();
            }
            // `STRESS_GPU_SPAWN=1` (M3): the children's engine bodies made on the GPU, in the
            // engine's own buffers, and the parents' retired there; the engine adopts them
            // once the host's split names the children.
            // `STRESS_GPU_LAYOUT=1` (M4, with the spawn): the tables laid out on the GPU too.
            let layout_live = std::env::var("STRESS_GPU_LAYOUT").is_ok_and(|v| v == "1");
            let live = layout_live || std::env::var("STRESS_GPU_SPAWN").is_ok_and(|v| v == "1");
            let spawned = if layout_live {
                self.split_pass_live(gpu, &engine, &out.disconnected)
            } else if live {
                self.spawn_live(gpu, &engine, &out.disconnected, false)
            } else {
                None
            };
            // `STRESS_GPU_SPLIT_CHECK=1`: the GPU's split of these islands, compared with
            // the host's below (M2).
            let split_check = std::env::var("STRESS_GPU_SPLIT_CHECK").is_ok_and(|v| v == "1");
            let gpu_children = split_check.then(|| {
                let parents: Vec<usize> = out.disconnected.clone();
                let ids: Vec<u64> = parents.iter().map(|&ci| self.solver.mirror.clusters[ci].id).collect();
                let run = self.solver.gpu_split_children(gpu, &parents);
                // M3: the children's rubble bodies as the GPU makes them, at the slots the
                // engine would give them next.
                let (first_body, first_collider) = engine.borrow().body_counts();
                let spawn = self.part_table(gpu, &engine).map(|table| self.solver.gpu_spawn_bodies(gpu, &run, &table, first_body, first_collider, None, &[]));
                // M4: the GPU's layout of this split into copies of the tables.
                let shadow = spawn.as_ref().map(|(_, bufs)| self.solver.gpu_layout_shadow(gpu, &run, bufs));
                (ids, self.solver.gpu_chunks().to_vec(), run.children, spawn.map(|s| s.0), shadow)
            });
            // In place when the tables can take it, opt-in (`STRESS_QS_SPLIT_IN_PLACE=1`):
            // off by default since it makes the bombard nondeterministic run to run and
            // differ from the rebuild (a GPU race after it, not yet found; the tables it
            // uploads are identical across runs, `STRESS_GPU_SPLIT_HASH=1`).
            let gpu_layout = layout_live && spawned.is_some();
            let in_place = if gpu_layout {
                let s = spawned.as_ref().expect("spawned");
                let resident = self.resident.as_ref().map(|r| r.qs.layout());
                let l = crate::solver::LiveSplit { comps: s.comps.clone(), run: None, resident };
                let done = self.solver.split_resident_with(gpu, &out.disconnected, Some(l))?;
                // The resident tables' entries moved on the GPU: their island count follows.
                if let Some(r) = self.resident.as_mut() {
                    r.qs.adopt_layout(&self.solver);
                    r.layout_generation = self.solver.layout_generation;
                    r.gpu_laid_out = true;
                }
                done
            } else {
                std::env::var("STRESS_QS_SPLIT_IN_PLACE").is_ok_and(|v| v == "1") && self.solver.split_resident(gpu, &out.disconnected)?
            };
            if in_place {
                if trace {
                    eprintln!("resident split: {} clusters split in place, {:.1} ms", out.disconnected.len(), t_split.elapsed().as_secs_f64() * 1e3);
                }
            } else {
                self.solver.sync(gpu);
                // The parents' chunk states as the GPU left them (the children's motion
                // comes from them; `sync` brings the bonds only).
                self.solver.sync_chunk_state(gpu, &out.disconnected);
                let t_synced = t_split.elapsed().as_secs_f64() * 1e3;
                for &ci in out.disconnected.iter().rev() {
                    self.solver.mirror.split_cluster(ci);
                }
                let t_splits = t_split.elapsed().as_secs_f64() * 1e3;
                if let Some(l) = self.solver.loads.as_mut() {
                    l.topology_version += 1;
                }
                self.solver.rebuild(gpu, 0.0)?;
                if trace {
                    eprintln!("resident split: {} clusters split, sync {t_synced:.1} ms, split {:.1} ms, rebuild {:.1} ms", out.disconnected.len(), t_splits - t_synced, t_split.elapsed().as_secs_f64() * 1e3 - t_splits);
                }
            }
            if let Some(spawned) = spawned {
                self.adopt_live(&engine, spawned)?;
                self.colliders_adopted(gpu);
            }
            if let Some((ids, layout_chunks, gpu_children, bodies, shadow)) = gpu_children {
                self.check_split(&ids, &layout_chunks, &gpu_children);
                if in_place {
                    if let Some((shadow, plan, old_count)) = shadow {
                        self.solver.compare_layout(gpu, &shadow, &plan, old_count);
                    }
                }
                pending_spawn.push((ids, gpu_children.iter().map(|c| c.len()).collect::<Vec<_>>(), bodies));
            }
            split = true;
            first = at + 1;
            if first >= max {
                break;
            }
        }
        if unconverged > 0 {
            self.shell.static_unconverged_frames += 1;
        }
        self.tail_seconds += tail.elapsed().as_secs_f64();
        let t_back = std::time::Instant::now();
        if split {
            // New clusters get their engine bodies (at the engine's end of the frame).
            for cl in &mut self.solver.mirror.clusters {
                cl.driven = true;
            }
            let t_wb = std::time::Instant::now();
            self.with_shell_mut(|w| w.coupled_write_back(&[]));
            for (ids, counts, bodies) in pending_spawn.drain(..) {
                if let Some(bodies) = bodies {
                    self.check_spawn(&engine, &ids, &counts, &bodies);
                }
            }
            if trace {
                eprintln!("resident split: engine bodies {:.1} ms", t_wb.elapsed().as_secs_f64() * 1e3);
            }
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

impl GpuWorld {
    /// M2's check: the GPU's children of the parents `ids` (layout chunk indices into
    /// `layout_chunks`, the layout before the split) against the mirror's, which the host
    /// split made.
    fn check_split(&mut self, ids: &[u64], layout_chunks: &[(usize, usize)], gpu: &[Vec<crate::gpu_split::GpuChild>]) {
        let rel = |a: f64, b: f64, scale: f64| (a - b).abs() / scale.max(1e-30);
        let m = &self.solver.mirror;
        let mut c = self.solver.split_check;
        for (&id, children) in ids.iter().zip(gpu) {
            c.splits += 1;
            // The host's children in creation order (its split event; later splits of the
            // same batch move clusters between slots).
            let made: Vec<u64> = m
                .events
                .iter()
                .rev()
                .find_map(|e| match e {
                    stress_ref::solver::SolverEvent::Split { parent, children, .. } if *parent == id => Some(children.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            let host: Vec<&stress_ref::solver::Cluster> = made.iter().filter_map(|cid| m.clusters.iter().find(|cl| cl.id == *cid)).collect();
            // A disconnection that left the island whole: no split on the host, one child
            // on the GPU.
            if host.is_empty() && children.len() == 1 {
                continue;
            }
            let same = host.len() == children.len()
                && host.iter().zip(children).all(|(h, g)| {
                    g.chunks.len() == h.chunks.len() && g.chunks.iter().zip(&h.chunks).all(|(&k, &hc)| layout_chunks[k as usize] == (h.structure, hc))
                });
            if !same {
                c.topology_mismatch += 1;
                if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
                    let gs: Vec<usize> = children.iter().map(|g| g.chunks.len()).collect();
                    let hs: Vec<usize> = host.iter().map(|h| h.chunks.len()).collect();
                    eprintln!("split check: parent {id}: GPU children {gs:?}, host {hs:?}");
                }
                continue;
            }
            for (h, g) in host.iter().zip(children) {
                c.children += 1;
                if h.anchored != g.anchored {
                    c.anchored_mismatch += 1;
                }
                c.mass = c.mass.max(rel(g.mass as f64, h.mass, h.mass));
                let size = h.inertia.m.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs()));
                for k in 0..3 {
                    c.com = c.com.max((g.com[k] as f64 - h.com.to_array()[k]).abs());
                    for j in 0..3 {
                        c.inertia = c.inertia.max(rel(g.inertia[k][j] as f64, h.inertia.m[k][j], size));
                    }
                }
                let (hv, hw) = (h.velocity.to_array(), h.angular_velocity.to_array());
                let (sv, sw) = (h.velocity.norm().max(1e-3), h.angular_velocity.norm().max(1e-3));
                if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
                    eprintln!("split check child of {id} (exact_frame_transfer {}): {} chunks, anchored {}, v gpu {:?} host {:?}, w gpu {:?} host {:?}", m.config.methods.exact_frame_transfer, g.chunks.len(), h.anchored, g.velocity, hv, g.angular_velocity, hw);
                }
                for k in 0..3 {
                    c.velocity = c.velocity.max(rel(g.velocity[k] as f64, hv[k], sv));
                    c.angular_velocity = c.angular_velocity.max(rel(g.angular_velocity[k] as f64, hw[k], sw));
                }
            }
        }
        self.solver.split_check = c;
    }
}

impl GpuWorld {
    /// M3's check: the GPU-made bodies (`bodies`, by parent `ids` in order, `counts`
    /// children each) against those the engine made for the host's children.
    fn check_spawn(&mut self, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, ids: &[u64], counts: &[usize], bodies: &[crate::gpu_split::GpuBody]) {
        let m = &self.solver.mirror;
        let e = engine.borrow();
        let mut c = self.solver.spawn_check;
        let mut at = 0;
        // The engine's slots for new bodies in key order (`sync_bodies`).
        for (&id, &n) in ids.iter().zip(counts) {
            // A disconnection that left the island whole made no body.
            if n < 2 {
                continue;
            }
            let made: Vec<u64> = m
                .events
                .iter()
                .rev()
                .find_map(|ev| match ev {
                    stress_ref::solver::SolverEvent::Split { parent, children, .. } if *parent == id => Some(children.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            if made.len() == n {
                for (k, cid) in made.iter().enumerate() {
                    if let Some(host) = e.body_words(stress_ref::engine::BodyKey::Cluster(*cid)) {
                        let slot = e.body_slot(stress_ref::engine::BodyKey::Cluster(*cid)).unwrap_or(u32::MAX);
                        c.compare(&bodies[at + k], slot, &host);
                    }
                }
            }
            at += n;
        }
        self.solver.spawn_check = c;
    }
}

impl GpuWorld {
    /// The engine's part table on the GPU, built once (`gpu_split::PartTable`).
    fn part_table(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>) -> Option<std::rc::Rc<crate::gpu_split::PartTable>> {
        if self.part_table.is_none() {
            let sizes: Vec<usize> = self.solver.mirror.structures.iter().map(|s| s.chunks.len()).collect();
            let records = engine.borrow().part_records(&sizes)?;
            self.part_table = Some(std::rc::Rc::new(crate::gpu_split::PartTable::new(gpu, &records)));
        }
        self.part_table.clone()
    }
}

/// A live GPU spawn waiting for the host's split to name its children (`spawn_live`).
struct LiveSpawn {
    ids: Vec<u64>,
    /// Per parent: each child's chunks (structure, chunk) and collider chunks.
    children: Vec<Vec<Vec<(usize, usize)>>>,
    bodies: Vec<crate::gpu_split::GpuBody>,
    /// Each parent's children's chunks (structure-local, sorted: the mirror's components).
    comps: Vec<Vec<Vec<usize>>>,
}

impl GpuWorld {
    /// M3, live: the children of `parents` split on the GPU and their engine bodies made
    /// there, in the engine's buffers, the parents retired. None (the host makes them) when
    /// the engine cannot, or a parent has no body.
    fn spawn_live(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, parents: &[usize], layout: bool) -> Option<LiveSpawn> {
        let ids: Vec<u64> = parents.iter().map(|&ci| self.solver.mirror.clusters[ci].id).collect();
        let layout_chunks = self.solver.gpu_chunks().to_vec();
        let table = self.part_table(gpu, engine)?;
        let e = engine.borrow();
        let retire: Option<Vec<[u32; 3]>> = ids.iter().map(|&id| e.retire_record(BodyKey::Cluster(id))).collect();
        let retire = retire?;
        let (bodies_buf, cols_buf, max_bodies) = e.spawn_buffers()?;
        let (first_body, first_collider) = e.body_counts();
        let run = self.solver.gpu_split_children(gpu, parents);
        // M4: only when the layout can follow on the GPU (checked before anything is made).
        if layout {
            let added: usize = run.children.iter().map(|c| c.len().saturating_sub(1)).sum();
            if !self.solver.can_split_in_place(added) {
                return None;
            }
        }
        let target = crate::gpu_split::SpawnTarget { bodies: bodies_buf, colliders: cols_buf, max_bodies };
        let (bodies, _) = self.solver.gpu_spawn_bodies(gpu, &run, &table, first_body, first_collider, Some(target), &retire);
        let comps: Vec<Vec<Vec<usize>>> = run
            .children
            .iter()
            .map(|list| {
                list.iter()
                    .map(|c| {
                        let mut v: Vec<usize> = c.chunks.iter().map(|&k| layout_chunks[k as usize].1).collect();
                        v.sort_unstable();
                        v
                    })
                    .collect()
            })
            .collect();
        let children = run
            .children
            .iter()
            .map(|list| {
                list.iter()
                    .map(|c| c.chunks.iter().flat_map(|&k| { let sc = layout_chunks[k as usize]; std::iter::repeat(sc).take(table.counts[sc.0][sc.1] as usize) }).collect())
                    .collect()
            })
            .collect();
        Some(LiveSpawn { ids, children, bodies, comps })
    }

    /// M4, live: a whole split on the GPU in one submission and one read
    /// (`gpu_split_pass`). None (the host splits) when it cannot: a parent with no engine
    /// body, or no room (which the GPU checks before it makes anything).
    fn split_pass_live(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, parents: &[usize]) -> Option<LiveSpawn> {
        if !self.solver.can_split_in_place(0) {
            return None;
        }
        let ids: Vec<u64> = parents.iter().map(|&ci| self.solver.mirror.clusters[ci].id).collect();
        let layout_chunks = self.solver.gpu_chunks().to_vec();
        let table = self.part_table(gpu, engine)?;
        let e = engine.borrow();
        let retire: Vec<[u32; 3]> = ids.iter().map(|&id| e.retire_record(BodyKey::Cluster(id))).collect::<Option<_>>()?;
        let parent_bodies: Vec<[[f32; 4]; 2]> = ids
            .iter()
            .map(|&id| e.frame_record(BodyKey::Cluster(id)).map(|(b, c, p)| [[f32::from_bits(b), c[0], c[1], c[2]], p]))
            .collect::<Option<_>>()?;
        let (bodies_buf, cols_buf, max_bodies) = e.spawn_buffers()?;
        let max_colliders = e.collider_capacity();
        let (first_body, first_collider) = e.body_counts();
        let pass_engine = crate::gpu_split::PassEngine {
            parent_bodies,
            target: crate::gpu_split::SpawnTarget { bodies: bodies_buf, colliders: cols_buf, max_bodies },
            max_colliders,
            first_body,
            first_collider,
            retire,
        };
        let resident = self.resident.as_ref().map(|r| r.qs.layout());
        // The collider table refreshed on the GPU too, when it has room for the engine's.
        let colliders = self.resident.as_ref().and_then(|r| r.colliders.as_ref().filter(|c| c.capacity() >= max_colliders).map(|c| (c, &r.ingest.collider_info)));
        let refreshed = colliders.is_some();
        let result = self.solver.gpu_split_pass(gpu, parents, &pass_engine, &table, resident, colliders);
        drop(e);
        if result.aborted {
            return None;
        }
        if let Some(r) = self.resident.as_mut() {
            r.colliders_refreshed = refreshed;
        }
        let children = result
            .children
            .iter()
            .map(|list| {
                list.iter()
                    .map(|c| c.chunks.iter().flat_map(|&k| { let sc = layout_chunks[k as usize]; std::iter::repeat(sc).take(table.counts[sc.0][sc.1] as usize) }).collect())
                    .collect()
            })
            .collect();
        let comps: Vec<Vec<Vec<usize>>> = result
            .children
            .iter()
            .map(|list| {
                list.iter()
                    .map(|c| {
                        let mut v: Vec<usize> = c.chunks.iter().map(|&k| layout_chunks[k as usize].1).collect();
                        v.sort_unstable();
                        v
                    })
                    .collect()
            })
            .collect();
        Some(LiveSpawn { ids, children, bodies: result.bodies, comps })
    }

    /// The engine adopts a live spawn's bodies, named by the host's split (its events).
    fn adopt_live(&mut self, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, s: LiveSpawn) -> Result<(), String> {
        let m = &self.solver.mirror;
        let mut adopted = Vec::new();
        let mut retired = Vec::new();
        let mut at = 0;
        for (&id, children) in s.ids.iter().zip(&s.children) {
            if children.len() < 2 {
                continue;
            }
            let made: Vec<u64> = m
                .events
                .iter()
                .rev()
                .find_map(|ev| match ev {
                    stress_ref::solver::SolverEvent::Split { parent, children, .. } if *parent == id => Some(children.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            if made.len() != children.len() {
                return Err(format!("GPU split of cluster {id}: {} children, the host's {}", children.len(), made.len()));
            }
            retired.push(BodyKey::Cluster(id));
            for (cid, chunks) in made.iter().zip(children) {
                let g = &s.bodies[at];
                at += 1;
                let cl = m.clusters.iter().find(|c| c.id == *cid).ok_or("adopted child not in the mirror")?;
                adopted.push(crate::resident::AdoptedBody {
                    key: BodyKey::Cluster(*cid),
                    signature: stress_ref::world::cluster_signature(cl),
                    fixed: cl.anchored,
                    slot: g.slot,
                    first_collider: g.first_collider,
                    body: g.body,
                    colliders: g.colliders.clone(),
                    chunks: chunks.clone(),
                    com: [g.com[0] as f64, g.com[1] as f64, g.com[2] as f64],
                    principal: [g.principal[0] as f64, g.principal[1] as f64, g.principal[2] as f64, g.principal[3] as f64],
                });
            }
        }
        engine.borrow_mut().adopt_spawned(adopted, &retired)
    }
}

impl ResidentState {
    fn ingest_chunks(&self) -> u32 {
        self.ingest.chunk_count()
    }
}
