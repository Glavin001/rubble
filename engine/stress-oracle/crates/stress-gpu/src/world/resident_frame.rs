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

/// The collider table's lookups of a layout (`host_collider_table`), kept while it stands.
pub(super) struct TableLookup {
    generation: u64,
    chunk_index: HashMap<(usize, usize), u32>,
    by_id: HashMap<u64, usize>,
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
        if self.resident_engine.is_none() || !self.qs_frame_conditions() {
            return false;
        }
        if !self.tiered {
            return !self.shell.qs_islands;
        }
        // Routed (M5): unless the last frame's check found a possible island seed.
        self.routed_resident && !self.route_next && self.promoted.is_empty()
    }

    /// The routed frame's check, made and given this frame's impactors (engine body and
    /// bounding radius each) when the tiered coupling runs.
    fn prepare_routing(&mut self, gpu: &Gpu) {
        if !self.tiered {
            return;
        }
        let engine = self.resident_engine.clone().expect("resident engine");
        let e = engine.borrow();
        let list: Vec<(u32, f32)> = self
            .solver
            .impactors
            .iter()
            .enumerate()
            .filter_map(|(i, imp)| {
                let slot = e.body_slot(BodyKey::Impactor(i))?;
                let r = match imp.shape {
                    stress_ref::scene::ImpactorShape::Box { half_extents } => stress_ref::math::Vec3::from_array(half_extents).norm(),
                    stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
                };
                // The engine's centre may sit off the shape's (its cooked centre of mass).
                Some((slot, (1.01 * r + 1e-3) as f32))
            })
            .collect();
        drop(e);
        let routing = self.routing.get_or_insert_with(|| crate::routing::Routing::new(gpu));
        routing.set_impactors(gpu, &list);
    }

    /// The engine's collider table as the coupling sees it (`ColliderInfo`), against the
    /// solver's current layout and mirror (only the colliders `range`, if given); and the
    /// static geometry's modulus.
    fn host_collider_table(&mut self, e: &dyn ResidentEngine, range: Option<std::ops::Range<u32>>) -> (Vec<crate::ingest::ColliderInfo>, f64) {
        let generation = self.solver.layout_generation;
        if self.table_lookup.as_ref().is_none_or(|l| l.generation != generation) {
            let m = &self.solver.mirror;
            self.table_lookup = Some(TableLookup {
                generation,
                chunk_index: self.solver.gpu_chunks().iter().enumerate().map(|(k, &sc)| (sc, k as u32)).collect(),
                by_id: m.clusters.iter().enumerate().map(|(ci, cl)| (cl.id, ci)).collect(),
            });
        }
        let lookup = self.table_lookup.as_ref().expect("table lookup");
        let (chunk_index, by_id) = (&lookup.chunk_index, &lookup.by_id);
        let m = &self.solver.mirror;
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
        let chunk_of = |s: usize, c: usize| chunk_index.get(&(s, c)).copied();
        let table = match range {
            Some(r) => e.collider_table_range(r, &chunk_of, &other),
            None => e.collider_table(&chunk_of, &other),
        };
        (table, ground_modulus)
    }

    /// Only some colliders changed since the table was built (an impactor added, a
    /// body retired), under the same layout: their entries rewritten in place, when the
    /// engine knows which and the table has room. False: rebuild it.
    fn patch_collider_table(&mut self, gpu: &Gpu, e: &dyn ResidentEngine, version: u64) -> bool {
        let Some(r) = self.resident.as_ref() else { return false };
        if std::env::var("STRESS_COLLIDER_PATCH").is_ok_and(|v| v == "0") {
            return false;
        }
        let Some(changes) = e.collider_changes_since(r.engine_version) else { return false };
        let count = e.body_counts().1;
        let entry = std::mem::size_of::<crate::ingest::ColliderInfo>() as u64;
        if r.ingest.collider_info.size() < entry * count as u64 || r.colliders.as_ref().is_some_and(|c| c.capacity() < count) {
            return false;
        }
        for range in changes {
            let (part, _) = self.host_collider_table(e, Some(range.clone()));
            let r = self.resident.as_mut().expect("resident tables");
            gpu.queue.write_buffer(&r.ingest.collider_info, entry * range.start as u64, bytemuck::cast_slice(&part));
            if let Some(c) = r.colliders.as_mut() {
                c.set_keys_range(gpu, range.start, &part, self.solver.gpu_chunks());
            }
        }
        self.resident.as_mut().expect("resident tables").engine_version = version;
        // `STRESS_GPU_COLLIDER_CHECK=1`: the patched table against the whole one.
        if std::env::var("STRESS_GPU_COLLIDER_CHECK").is_ok_and(|v| v == "1") {
            let (table, _) = self.host_collider_table(e, None);
            let r = self.resident.as_ref().expect("resident tables");
            let read = gpu.read_ranges(&[(&r.ingest.collider_info, 0, entry * table.len() as u64)]);
            let gpu_table: Vec<crate::ingest::ColliderInfo> = bytemuck::pod_collect_to_vec(&read[0]);
            self.solver.collider_check.compare(&gpu_table, &table);
            self.solver.collider_check.patches += 1;
        }
        true
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
        // More substeps than the ingest has room for, the engine as it was made for: only
        // its per-substep buffers grow (`STRESS_INGEST_GROW=0`: the whole frame's tables
        // made anew, as before).
        let (rows, bodies) = e.engine_capacity();
        if let Some(r) = self.resident.as_mut() {
            if r.substeps < substeps && r.ingest.capacity() == (rows, bodies) && std::env::var("STRESS_INGEST_GROW").map_or(true, |v| v != "0") {
                let cap = substeps.max(1).next_power_of_two();
                r.ingest.grow_substeps(gpu, cap);
                r.substeps = cap;
                if r.engine_version == version && r.layout_generation == generation {
                    return;
                }
            }
        }
        // Only the engine's colliders changed, under the same layout: patched in place.
        if let Some(r) = &self.resident {
            if r.layout_generation == generation && r.substeps >= substeps && r.ingest_chunks() == self.solver.gpu_chunks().len().max(1) as u32 && !r.gpu_laid_out && self.patch_collider_table(gpu, &*e, version) {
                return;
            }
        }
        let (table, ground_modulus) = self.host_collider_table(&*e, None);
        let m = &self.solver.mirror;
        let gpu_colliders = crate::gpu_layout();
        let collider_cap = e.collider_capacity();
        let frames: Vec<Option<EngineFrame>> = m.clusters.iter().map(|cl| if cl.anchored { None } else { e.engine_frame(BodyKey::Cluster(cl.id)) }).collect();
        let body_records: Vec<Option<[u32; 3]>> = m.clusters.iter().map(|cl| e.retire_record(BodyKey::Cluster(cl.id))).collect();
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
                r.qs.rebuild(gpu, &self.solver, &frames, &body_records, dt);
                r.engine_version = version;
                r.layout_generation = generation;
            }
            Some(r) => {
                let cap = substeps.max(r.substeps).max(1).next_power_of_two();
                r.ingest = EngineIngest::new(gpu, rows, cap, bodies, chunks, &table);
                r.substeps = cap;
                r.qs.rebuild(gpu, &self.solver, &frames, &body_records, dt);
                r.engine_version = version;
                r.layout_generation = generation;
            }
            None => {
                // Room for a few substeps from the start (`STRESS_INGEST_SUBSTEPS`, 4): growing
                // the ingest's buffers later costs their zero fill on the frame that needs
                // them, a shot's.
                let first = std::env::var("STRESS_INGEST_SUBSTEPS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(4);
                let cap = substeps.max(first).max(1).next_power_of_two();
                self.resident = Some(ResidentState {
                    ingest: EngineIngest::new(gpu, rows, cap, bodies, chunks, &table),
                    substeps: cap,
                    qs: ResidentQs::new(gpu, &self.solver, &frames, &body_records, dt),
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
    /// `check`: compared now (the host's mirror and layout have followed every split the
    /// GPU's table holds; a frame's log checks once, after its last).
    fn colliders_adopted(&mut self, gpu: &Gpu, check: bool) {
        let engine = self.resident_engine.clone().expect("resident engine");
        let e = engine.borrow();
        let Some(r) = self.resident.as_ref() else { return };
        if !r.colliders_refreshed {
            return;
        }
        drop(e);
        if check {
            self.check_colliders(gpu);
        }
        let e = engine.borrow();
        let r = self.resident.as_mut().expect("resident tables");
        r.engine_version = e.layout_version();
        r.gpu_laid_out = false;
        r.colliders_refreshed = false;
    }

    /// `STRESS_GPU_COLLIDER_CHECK=1`: the GPU's collider table against the host's, as the
    /// host's mirror and layout stand.
    fn check_colliders(&mut self, gpu: &Gpu) {
        if !std::env::var("STRESS_GPU_COLLIDER_CHECK").is_ok_and(|v| v == "1") {
            return;
        }
        let engine = self.resident_engine.clone().expect("resident engine");
        let e = engine.borrow();
        let (table, _) = self.host_collider_table(&*e, None);
        let n = table.len() as u64 * std::mem::size_of::<crate::ingest::ColliderInfo>() as u64;
        let r = self.resident.as_ref().expect("resident tables");
        let read = gpu.read_ranges(&[(&r.ingest.collider_info, 0, n)]);
        let gpu_table: Vec<crate::ingest::ColliderInfo> = bytemuck::pod_collect_to_vec(&read[0]);
        self.solver.collider_check.compare(&gpu_table, &table);
    }

    /// The mirror current: the solver's state and, after resident frames, the engine's
    /// motion of the bodies it drives (which those frames leave on the GPU).
    pub fn sync(&mut self, gpu: &Gpu) {
        self.flush_pending(gpu).expect("the frame's GPU splits followed");
        self.solver.sync(gpu);
        if self.resident_frames > 0 {
            self.resident_sync_motion();
        }
    }

    /// A copy of the mirror and the impactors with every body at the engine's pose now, for
    /// a view (a renderer): the world's own mirror keeps its state, so a viewed run steps as
    /// an unviewed one does. Follows the frame's pending GPU splits first (lossless: their
    /// deferral is).
    pub fn view_state(&mut self, gpu: &Gpu) -> (stress_ref::solver::ReferenceSolver, Vec<stress_ref::world::Impactor>) {
        self.flush_pending(gpu).expect("the frame's GPU splits followed");
        self.solver.sync(gpu);
        let mut m = self.solver.mirror.clone();
        let mut imps = self.solver.impactors.clone();
        if let Some(engine) = self.resident_engine.clone() {
            let motion: HashMap<BodyKey, stress_ref::engine::BodyMotion> = engine.borrow().motion().into_iter().map(|m| (m.key, m)).collect();
            for cl in m.clusters.iter_mut().filter(|c| !c.anchored) {
                if let Some(mm) = motion.get(&BodyKey::Cluster(cl.id)) {
                    (cl.pose, cl.velocity, cl.angular_velocity) = (mm.pose, mm.velocity, mm.angular_velocity);
                }
            }
            for (i, imp) in imps.iter_mut().enumerate() {
                if let Some(mm) = motion.get(&BodyKey::Impactor(i)) {
                    (imp.pose, imp.velocity, imp.angular_velocity) = (mm.pose, mm.velocity, mm.angular_velocity);
                }
            }
        }
        (m, imps)
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
        self.prepare_routing(gpu);
        if phases {
            eprintln!("resident phases: tables {:.2} ms", t_tables.elapsed().as_secs_f64() * 1e3);
        }
        let t_engine = std::time::Instant::now();
        {
            let r = self.resident.as_ref().expect("resident tables");
            engine.borrow_mut().step_resident(gpu, fdt, n, &r.ingest)?;
        }
        // The last frame's GPU splits, followed by the host while the engine's substeps run.
        let t_flush = std::time::Instant::now();
        let flushed = self.pending_splits.len();
        self.flush_pending(gpu)?;
        if flushed > 0 && std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
            eprintln!("resident frame: {flushed} GPU splits followed by the host in {:.1} ms (the engine's substeps running)", t_flush.elapsed().as_secs_f64() * 1e3);
        }
        if phases {
            let encode = t_engine.elapsed().as_secs_f64() * 1e3;
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("resident phases: engine {n} substeps + ingest: encode {encode:.1} ms, done {:.1} ms", t_engine.elapsed().as_secs_f64() * 1e3);
        }
        // `STRESS_GPU_DET_STAGE=<frame>`: that frame's stage outputs hashed (reads of their
        // own), to find which stage two runs part in.
        let det_stage = std::env::var("STRESS_GPU_DET_STAGE").ok().and_then(|v| v.parse::<u64>().ok()) == Some(self.frame as u64);
        let hash = |bytes: &[u8]| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        // `STRESS_GPU_DET_BODIES=1`: the engine's bodies after its substeps, hashed every
        // frame (a read of their own).
        if std::env::var("STRESS_GPU_DET_BODIES").is_ok() {
            let e = engine.borrow();
            let bodies: Vec<u8> = gpu.read(e.engine_buffers().bodies);
            // The ingest's output by its physical content: per chunk, each manifold's chunk,
            // other body, point, impulse and other side (not its slot or key).
            let r = self.resident.as_ref().expect("resident tables");
            let per_chunk = r.ingest.read_manifolds(gpu);
            let mut flat: Vec<u32> = Vec::new();
            for m in &per_chunk {
                flat.push(m.len() as u32);
                for rec in m {
                    flat.extend([rec.ids[0], rec.ids[1]]);
                    flat.extend(rec.point.iter().chain(&rec.impulse).chain(&rec.other).map(|v| v.to_bits()));
                }
            }
            eprintln!("det bodies frame {}: {n} substeps, {:x}, ingest {:x}", self.frame, hash(&bodies), hash(bytemuck::cast_slice(&flat)));
        }
        if det_stage {
            let e = engine.borrow();
            let b = e.engine_buffers();
            let r = self.resident.as_ref().expect("resident tables");
            let bodies: Vec<u8> = gpu.read(b.bodies);
            let counters: Vec<u8> = gpu.read(b.counters);
            let manifolds: Vec<u8> = gpu.read(&r.ingest.manifolds);
            let sort: Vec<u8> = gpu.read(&r.ingest.sort);
            let records: Vec<u8> = gpu.read(&r.ingest.records);
            eprintln!("det stage frame {} engine: {n} substeps, bodies {:x}, counters {:x}, records {:x}, sort {:x}, manifolds {:x}", self.frame, hash(&bodies), hash(&counters), hash(&records), hash(&sort), hash(&manifolds));
            // The ingest's output as the stress side reads it: each chunk's manifolds in order.
            let per_chunk = r.ingest.read_manifolds(gpu);
            let flat: Vec<u8> = per_chunk.iter().flat_map(|m| bytemuck::cast_slice::<_, u8>(m).to_vec().into_iter().chain((m.len() as u32).to_le_bytes())).collect();
            eprintln!("det stage frame {} ingest: {} manifolds, content {:x}", self.frame, per_chunk.iter().map(|m| m.len()).sum::<usize>(), hash(&flat));
            // Records sharing a merge key (`record_before` then falls back to slot order).
            let words: &[[u32; 4]] = bytemuck::cast_slice(&records[..records.len() / 80 * 80]);
            let mut keys: HashMap<([u32; 4], u32), u32> = HashMap::new();
            for rec in words.chunks_exact(5) {
                if rec[0][3] != 0 {
                    *keys.entry((rec[4], rec[0][0])).or_insert(0) += 1;
                }
            }
            let dup = keys.values().filter(|&&c| c > 1).count();
            let worst = keys.values().copied().max().unwrap_or(0);
            eprintln!("det stage frame {} records: {} valid, {} keys, {dup} keys shared, at most {worst} records per key", self.frame, keys.values().sum::<u32>(), keys.len());
        }
        // `STRESS_GPU_DET_INGEST=<frame>` (a one-substep frame): its ingest run again and
        // again from the engine's buffers as they are, its manifolds' physical content hashed.
        if n == 1 && std::env::var("STRESS_GPU_DET_INGEST").ok().and_then(|v| v.parse::<u64>().ok()) == Some(self.frame as u64) {
            let e = engine.borrow();
            let r = self.resident.as_ref().expect("resident tables");
            let offset = e.last_row_offset();
            let runs: usize = std::env::var("STRESS_GPU_DET_REPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
            for k in 0..runs {
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                r.ingest.substep(gpu, &mut enc, &e.engine_buffers(), offset, 0, fdt);
                r.ingest.merge(gpu, &mut enc, &e.engine_buffers(), 1);
                gpu.queue.submit([enc.finish()]);
                let per_chunk = r.ingest.read_manifolds(gpu);
                let mut flat: Vec<u32> = Vec::new();
                for m in &per_chunk {
                    flat.push(m.len() as u32);
                    for rec in m {
                        flat.extend([rec.ids[0], rec.ids[1]]);
                        flat.extend(rec.point.iter().chain(&rec.impulse).chain(&rec.other).map(|v| v.to_bits()));
                    }
                }
                eprintln!("det ingest frame {} run {k}: {} manifolds, {:x}", self.frame, per_chunk.iter().map(|m| m.len()).sum::<usize>(), hash(bytemuck::cast_slice(&flat)));
            }
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
        // After a frame-mode window, any later windows (an aborted GPU split) run as before.
        let mut frame_fallback = false;
        loop {
            // The cascade from pass `first` on, under GPU control; then the frame's one
            // read: the cascade's status, the engine's counters, the pulse slots' status.
            self.resident_tables(gpu, n);
            let t_pass = std::time::Instant::now();
            // `STRESS_GPU_DET_REPLAY=<frame>`: that frame's first window run again and again
            // from the same buffers (saved, restored), each run's status and state hashed:
            // whether the stress side alone repeats.
            if first == 0 && std::env::var("STRESS_GPU_DET_REPLAY").ok().and_then(|v| v.parse::<u64>().ok()) == Some(self.frame as u64) {
                self.det_replay(gpu, fdt, max);
            }
            // `STRESS_GPU_FRAME=1` (M4): the frame's passes, and the splits between them, in
            // one submission; the host follows the splits from the frame's one read.
            let frame_mode = first == 0 && !frame_fallback && crate::gpu_frame() && self.frame_splits_ready(gpu, &engine);
            let part_table = if frame_mode { self.part_table(gpu, &engine) } else { None };
            let frame_mode = frame_mode && part_table.is_some();
            let r = self.resident.take().expect("resident tables");
            let window = if frame_mode { max - first } else { CASCADE_WINDOW.min(max - first) };
            // The routed frame's check (M5), after the window's passes in its submission.
            let routing = if self.tiered { self.routing.as_ref() } else { None };
            let engine_base = r.qs.layout().engine_base;
            let ctab = r.qs.layout().ctab.clone();
            let route_post = |s: &crate::solver::GpuSolver, enc: &mut wgpu::CommandEncoder| {
                if let Some(rt) = routing {
                    let e = engine.borrow();
                    rt.encode(gpu, enc, &s.routing_inputs(), &ctab, engine_base, e.engine_buffers().bodies, fdt, stress_ref::world::tiered_impact_speed());
                }
            };
            let post: Option<&dyn Fn(&crate::solver::GpuSolver, &mut wgpu::CommandEncoder)> = if routing.is_some() { Some(&route_post) } else { None };
            let pre = |s: &crate::solver::GpuSolver, enc: &mut wgpu::CommandEncoder, pass: usize| {
                r.qs.encode_poses(gpu, enc, &r.ingest, s);
                if pass == 0 {
                    r.qs.encode_filter(gpu, enc, &r.ingest, s);
                }
                r.qs.encode_static_loads(gpu, enc, &r.ingest, s);
            };
            let run = if frame_mode {
                let e = engine.borrow();
                let (bodies, colliders, max_bodies) = e.spawn_buffers().expect("spawn buffers");
                let max_colliders = e.collider_capacity();
                let fs = crate::solver::FrameSplits {
                    target: crate::gpu_split::SpawnTarget { bodies, colliders, max_bodies },
                    max_colliders,
                    table: part_table.as_ref().expect("part table"),
                    resident: r.qs.layout(),
                    colliders: r.colliders.as_ref().filter(|c| c.capacity() >= max_colliders).map(|c| (c, &r.ingest.collider_info)),
                    layout_params: self.solver.frame_layout_params(),
                };
                self.solver.resident_cascade_with(gpu, fdt, first, window, &pre, Some(&fs), post)
            } else {
                self.solver.resident_cascade_with(gpu, fdt, first, window, &pre, None, post)
            };
            if phases {
                let encode = t_pass.elapsed().as_secs_f64() * 1e3;
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                eprintln!("resident phases: cascade encode {encode:.1} ms, done {:.1} ms", t_pass.elapsed().as_secs_f64() * 1e3);
            }
            if det_stage && first == 0 {
                let (hs, hb) = self.solver.state_hashes(gpu);
                let ctab: Vec<u8> = gpu.read(r.qs.layout().ctab);
                eprintln!("det stage frame {} cascade: state {hs:x}, bonds {hb:x}, ctab {:x}", self.frame, hash(&ctab));
            }
            let e = engine.borrow();
            let status = e.resident_status();
            let engine_parts = status.len();
            let mut parts = vec![run.range()];
            parts.extend(status);
            parts.push(r.qs.status_range());
            let route_part = parts.len();
            if let Some(rt) = routing {
                parts.push(rt.range());
            }
            // `STRESS_GPU_DET_END=1`: the frame's ingest output and the engine's bodies in
            // this same read (no sync of their own), hashed below.
            let det_end = std::env::var("STRESS_GPU_DET_END").is_ok() && first == 0;
            let base_parts = parts.len();
            if det_end {
                let n_chunks = r.ingest.chunk_count() as u64;
                parts.push((&r.ingest.sort, 0, 4 * (2 * n_chunks + 1)));
                parts.push((&r.ingest.manifolds, 0, r.ingest.manifolds.size()));
                parts.push((e.engine_buffers().bodies, 0, e.engine_buffers().bodies.size()));
            }
            let log_part = parts.len();
            if frame_mode {
                parts.extend(self.solver.frame_log_range());
            }
            let t_read = std::time::Instant::now();
            let host_before_read = t_prepare.elapsed().as_secs_f64() * 1e3;
            let read = gpu.read_ranges(&parts);
            let read_ms = t_read.elapsed().as_secs_f64() * 1e3;
            if std::env::var("STRESS_FRAME_HOST_TRACE").is_ok() {
                eprintln!("frame {} host: to the read {host_before_read:.2} ms (substeps {:.2}, tables {:.2}, engine {:.2}, cascade {:.2}), read {read_ms:.2}", self.frame, (t_tables - t_substeps).as_secs_f64() * 1e3, (t_engine - t_tables).as_secs_f64() * 1e3, (t_pass - t_engine).as_secs_f64() * 1e3, (t_read - t_pass).as_secs_f64() * 1e3);
            }
            if det_end {
                use std::hash::{Hash, Hasher};
                let h = |b: &[u8]| {
                    let mut x = std::collections::hash_map::DefaultHasher::new();
                    b.hash(&mut x);
                    x.finish()
                };
                let n = r.ingest.chunk_count() as usize;
                let sort: &[u32] = bytemuck::cast_slice(&read[base_parts]);
                let man: &[crate::ingest::Record] = bytemuck::cast_slice(&read[base_parts + 1][..read[base_parts + 1].len() / 80 * 80]);
                let mut flat: Vec<u32> = Vec::new();
                for k in 0..n {
                    let (b, c) = (sort[k] as usize, sort[n + 1 + k] as usize);
                    flat.push(c as u32);
                    for rec in &man[b..b + c] {
                        flat.extend([rec.ids[0], rec.ids[1]]);
                        flat.extend(rec.point.iter().chain(&rec.impulse).chain(&rec.other).map(|v| v.to_bits()));
                    }
                }
                eprintln!("det end frame {}: ingest {:x}, bodies {:x}", self.frame, h(bytemuck::cast_slice(&flat)), h(&read[base_parts + 2]));
            }
            drop(e);
            self.resident = Some(r);
            let mut out = run.outcome(&read[0]);
            if std::env::var("STRESS_GPU_SPLIT_HASH").is_ok() {
                use std::hash::{Hash, Hasher};
                let (hs, hb) = self.solver.state_hashes(gpu);
                eprintln!("frame hash: pass {first} stress state {hs:x} bonds {hb:x} ingest {:x}", { let mut h = std::collections::hash_map::DefaultHasher::new(); read[1].hash(&mut h); h.finish() });
            }
            engine.borrow_mut().resident_finish(&read[1..1 + engine_parts])?;
            self.resident.as_ref().expect("resident tables").qs.check(&read[1 + engine_parts])?;
            if routing.is_some() {
                // The last window's check stands (each runs after its own splits).
                self.route_next = crate::routing::Routing::flagged(&read[route_part]);
            }
            if frame_mode {
                // The frame's GPU splits, followed in order (an aborted one ends the log: the
                // cascade stopped there, and the code below splits on the host).
                let t_apply = std::time::Instant::now();
                let log = self.solver.frame_log(gpu, &read[log_part]);
                let applied = self.apply_frame_log(gpu, &engine, &log, part_table.as_ref().expect("part table"))?;
                let apply_ms = t_apply.elapsed().as_secs_f64() * 1e3;
                if applied > 0 {
                    split = true;
                    self.solver.adopt_gpu_tables();
                }
                if trace {
                    eprintln!("resident frame splits: {applied} on the GPU, log {} words, followed in {apply_ms:.1} ms", log.len());
                }
                if log.get(2).copied().unwrap_or(0) != 0 {
                    return Err("frame split log overflowed".into());
                }
                frame_fallback = true;
                // The status over the islands as the GPU's splits left them.
                out = crate::resident::CascadeOutcome::from_status(bytemuck::cast_slice(&read[0]), self.solver.mirror.clusters.len());
            }
            unconverged += out.unconverged;
            for cl in self.solver.mirror.clusters.iter_mut().filter(|c| !c.bonds.is_empty()) {
                cl.activity = Activity::Settled;
            }
            if std::env::var("STRESS_GPU_QS_LOADS").is_ok() {
                let loads = self.resident.as_ref().expect("resident tables").qs.read_frame_loads(gpu);
                let total: f64 = loads.iter().map(|(f, _)| f.norm()).sum();
                let peak = loads.iter().map(|(f, _)| f.norm()).fold(0.0f64, f64::max);
                let loaded = loads.iter().filter(|(f, _)| f.norm() > 0.0).count();
                eprintln!("frame {} loads: {loaded} chunks loaded, sum |f| {total:.4e} N, largest {peak:.4e} N", self.frame);
                let (islands_buf, _, _, _, _) = self.solver.resident_buffers();
                let isl: Vec<[f32; 4]> = gpu.read_range(islands_buf, 0, 25);
                eprintln!("  island 0: range {:?}, info {:?}, com {:?}", bytemuck::cast::<[f32; 4], [u32; 4]>(isl[0]), bytemuck::cast::<[f32; 4], [u32; 4]>(isl[1]), isl[2]);
                let man = self.resident.as_ref().expect("resident tables").ingest.read_manifolds(gpu);
                for (k, list) in man.iter().enumerate().filter(|(_, l)| !l.is_empty()).take(6) {
                    for r in list {
                        let j = (r.impulse[0].powi(2) + r.impulse[1].powi(2) + r.impulse[2].powi(2)).sqrt();
                        eprintln!("  chunk {k}: other {:#x}, |j| {j:.3e}, closing {:.3}, other mass {:.1}", r.ids[1], r.impulse[3], r.other[0]);
                    }
                }
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
            // `STRESS_GPU_DET_TRACE=1`: what the frame's status read says, per window (no
            // read of its own), to find where two runs part.
            if std::env::var("STRESS_GPU_DET_TRACE").is_ok() {
                let ids: Vec<u64> = out.disconnected.iter().map(|&ci| self.solver.mirror.clusters[ci].id).collect();
                eprintln!("det frame {} pass {first}: {} run, {} solved, {} converged, {} unconverged, at {:?}, disconnected {ids:?}, clusters {}, read {:x}", self.frame, out.passes, out.solved.len(), out.converged.len(), out.unconverged, out.disconnected_at, self.solver.mirror.clusters.len(), {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    read.hash(&mut h);
                    h.finish()
                });
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
            if !crate::gpu_layout() {
                self.resident_sync_motion();
            }
            // `STRESS_GPU_SPAWN=1` (M3): the children's engine bodies made on the GPU, in the
            // engine's own buffers, and the parents' retired there; the engine adopts them
            // once the host's split names the children.
            // `STRESS_GPU_LAYOUT=1` (M4, with the spawn): the tables laid out on the GPU too.
            let layout_live = crate::gpu_layout();
            let live = layout_live || std::env::var("STRESS_GPU_SPAWN").is_ok_and(|v| v == "1");
            let spawned = if layout_live && std::env::var("STRESS_GPU_PASS").is_ok_and(|v| v == "1") {
                self.split_pass_gpu(gpu, &engine, &out.disconnected)
            } else if layout_live {
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
                self.colliders_adopted(gpu, true);
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
            if self.pending_splits.is_empty() {
                self.with_shell_mut(|w| w.coupled_write_back(&[]));
            } else {
                // After the host follows the splits (the mirror names the children then).
                self.pending_write_back = true;
            }
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
        if std::env::var("STRESS_FRAME_HOST_TRACE").is_ok() {
            eprintln!("frame {} host: whole {:.2} ms", self.frame, t_prepare.elapsed().as_secs_f64() * 1e3);
        }
        self.frame += 1;
        self.resident_frames += 1;
        Ok(())
    }
}

impl GpuWorld {
    /// Whether a frame's GPU split passes can run (`STRESS_GPU_FRAME=1`): made ready, with
    /// the counters as the host has them.
    fn frame_splits_ready(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>) -> bool {
        if !self.solver.can_split_in_place(0) {
            return false;
        }
        let e = engine.borrow();
        if e.spawn_buffers().is_none() {
            return false;
        }
        let (bodies, colliders) = e.body_counts();
        let Some(r) = self.resident.as_ref() else { return false };
        self.solver.prepare_frame_splits(gpu, r.qs.layout().ctab, bodies, colliders)
    }

    /// The frame's GPU splits, followed by the host in order (`passlog.slang`): the
    /// mirror's split into the GPU's components, the layout's bookkeeping, the engine's
    /// adoption of the GPU-made bodies. Returns how many it followed; it stops at an aborted
    /// one (nothing of it was made, and the cascade stopped there).
    fn apply_frame_log(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, log: &[u32], table: &crate::gpu_split::PartTable) -> Result<usize, String> {
        let mut at = 16usize;
        let mut applied = 0;
        while at < log.len() {
            let size = log[at] as usize;
            if size == 0 {
                break;
            }
            let (n, c, aborted) = (log[at + 1] as usize, log[at + 2] as usize, log[at + 3] != 0);
            if aborted {
                break;
            }
            let (first_body, first_collider) = (log[at + 5], log[at + 6]);
            let nb = (log[at + 7] - first_body) as usize;
            let total = log[at + 11] as usize;
            let h0 = at + 16 + 4 * n;
            let m0 = h0 + 14 * c;
            let b0 = m0 + total;
            let k0 = b0 + 28 * nb;
            // The entry describes itself (the tables before its split): the parents' islands,
            // cluster ids and structures, the members' chunks within their structures.
            let parents: Vec<usize> = (0..n).map(|p| log[at + 16 + 4 * p] as usize).collect();
            let ids: Vec<u64> = (0..n).map(|p| log[at + 16 + 4 * p + 2] as u64).collect();
            let structures: Vec<usize> = (0..n).map(|p| log[at + 16 + 4 * p + 3] as usize).collect();
            let f = |w: u32| f32::from_bits(w);
            let mut children_out: Vec<Vec<Vec<(usize, usize)>>> = Vec::with_capacity(n);
            let mut comps: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n);
            let mut bodies = Vec::new();
            let (mut h, mut mc) = (h0, 0usize);
            for p in 0..parents.len() {
                let z = log[at + 16 + 4 * p + 1] as usize;
                let st = structures[p];
                let mut kids = Vec::with_capacity(z);
                let mut comp = Vec::with_capacity(z);
                for _ in 0..z {
                    let head = &log[h..h + 14];
                    h += 14;
                    let count = head[0] as usize;
                    let chunks: Vec<(usize, usize)> = (0..count).map(|j| (st, log[m0 + mc + j] as usize)).collect();
                    mc += count;
                    let mut v: Vec<usize> = chunks.iter().map(|&(_, ch)| ch).collect();
                    v.sort_unstable();
                    comp.push(v);
                    kids.push(chunks.iter().flat_map(|&sc| std::iter::repeat(sc).take(table.counts[sc.0][sc.1] as usize)).collect());
                    if head[5] != 0 {
                        let (slot, fc, ncols) = (head[2], head[3], head[4] as usize);
                        let bb = b0 + 28 * (slot - first_body) as usize;
                        let mut body = [[0f32; 4]; 7];
                        for (w, v) in body.iter_mut().enumerate() {
                            *v = [f(log[bb + 4 * w]), f(log[bb + 4 * w + 1]), f(log[bb + 4 * w + 2]), f(log[bb + 4 * w + 3])];
                        }
                        let colliders = (0..ncols)
                            .map(|j| {
                                let cc = k0 + 16 * ((fc - first_collider) as usize + j);
                                let mut r = [[0f32; 4]; 4];
                                for (w, v) in r.iter_mut().enumerate() {
                                    *v = [f(log[cc + 4 * w]), f(log[cc + 4 * w + 1]), f(log[cc + 4 * w + 2]), f(log[cc + 4 * w + 3])];
                                }
                                r
                            })
                            .collect();
                        bodies.push(crate::gpu_split::GpuBody { slot, first_collider: fc, body, colliders, com: [f(head[6]), f(head[7]), f(head[8]), f(head[9])], principal: [f(head[10]), f(head[11]), f(head[12]), f(head[13])] });
                    }
                }
                children_out.push(kids);
                comps.push(comp);
            }
            let spawned = LiveSpawn { ids, children: children_out, bodies, comps: comps.clone() };
            // Deferred (`STRESS_GPU_FRAME_DEFER`, on unless 0): the engine adopts the bodies now,
            // named by the GPU's cluster ids; the mirror and layout follow next frame, while
            // the engine's substeps run (`flush_pending`).
            if std::env::var("STRESS_GPU_FRAME_DEFER").map_or(true, |v| v != "0") {
                let next_id = log[at + 10] as u64;
                let t1 = std::time::Instant::now();
                let child_ids = self.adopt_from_log(engine, &structures, &spawned, next_id)?;
                if let Some(r) = self.resident.as_mut() {
                    r.colliders_refreshed = r.colliders.is_some();
                }
                self.colliders_adopted(gpu, false);
                if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
                    eprintln!("frame split {applied}: {n} parents, {c} children, {total} chunks: engine {:.1} ms (the host's mirror follows later)", t1.elapsed().as_secs_f64() * 1e3);
                }
                self.pending_splits.push(PendingSplit { parents, comps: spawned.comps, child_ids });
                applied += 1;
                at += size;
                continue;
            }
            let resident = self.resident.as_ref().map(|r| r.qs.layout());
            let l = crate::solver::LiveSplit { comps, run: None, resident };
            let t0 = std::time::Instant::now();
            self.solver.split_resident_with(gpu, &parents, Some(l))?;
            let t_split = t0.elapsed().as_secs_f64() * 1e3;
            if let Some(r) = self.resident.as_mut() {
                r.qs.adopt_layout(&self.solver);
                r.layout_generation = self.solver.layout_generation;
                r.gpu_laid_out = true;
                r.colliders_refreshed = r.colliders.is_some();
            }
            let t1 = std::time::Instant::now();
            self.adopt_live(engine, spawned)?;
            self.colliders_adopted(gpu, false);
            if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
                eprintln!("frame split {applied}: {n} parents, {c} children, {total} chunks: mirror and layout {t_split:.1} ms, engine {:.1} ms", t1.elapsed().as_secs_f64() * 1e3);
            }
            applied += 1;
            at += size;
        }
        // Followed at once (not deferred): the table against the host's now.
        if applied > 0 && self.pending_splits.is_empty() {
            self.check_colliders(gpu);
        }
        Ok(applied)
    }

    fn det_replay(&mut self, gpu: &Gpu, fdt: f64, max: usize) {
        use std::hash::{Hash, Hasher};
        let hash = |bytes: &[u8]| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            h.finish()
        };
        let mut bufs = self.solver.cascade_buffers();
        bufs.push(self.resident.as_ref().expect("resident tables").qs.layout().ctab.clone());
        let copy_all = |from: &[wgpu::Buffer], to: &[wgpu::Buffer]| {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for (a, b) in from.iter().zip(to) {
                enc.copy_buffer_to_buffer(a, 0, b, 0, a.size());
            }
            gpu.queue.submit([enc.finish()]);
        };
        let saved: Vec<wgpu::Buffer> = bufs
            .iter()
            .map(|b| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("det replay"), size: b.size(), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }))
            .collect();
        copy_all(&bufs, &saved);
        let runs: usize = std::env::var("STRESS_GPU_DET_REPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        for k in 0..runs {
            copy_all(&saved, &bufs);
            let r = self.resident.take().expect("resident tables");
            let window = CASCADE_WINDOW.min(max);
            let run = self.solver.resident_cascade(gpu, fdt, 0, window, &|s, enc, pass| {
                r.qs.encode_poses(gpu, enc, &r.ingest, s);
                if pass == 0 {
                    r.qs.encode_filter(gpu, enc, &r.ingest, s);
                }
                r.qs.encode_static_loads(gpu, enc, &r.ingest, s);
            });
            let read = gpu.read_ranges(&[run.range()]);
            self.resident = Some(r);
            let after: Vec<u64> = bufs.iter().map(|b| hash(&gpu.read::<u8>(b))).collect();
            let out = run.outcome(&read[0]);
            eprintln!("det replay frame {} run {k}: status {:x}, {} passes, {} solved, {} converged, {} unconverged, {} repeats, {} fixed, {} certified, buffers {:x?}", self.frame, hash(&read[0]), out.passes, out.solved.len(), out.converged.len(), out.unconverged, out.skipped_unconverged, out.skipped_fixed, out.certified, after);
        }
        copy_all(&saved, &bufs);
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

/// A GPU split of a frame the engine has adopted, for the host's mirror and layout to
/// follow (`flush_pending`): the parents (island indices then), each one's children's chunks
/// (sorted), and the children's cluster ids as the GPU gave them.
pub(super) struct PendingSplit {
    parents: Vec<usize>,
    comps: Vec<Vec<Vec<usize>>>,
    child_ids: Vec<Vec<u64>>,
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
        Some(Self::live_spawn(ids, &layout_chunks, &table, result))
    }

    /// A live spawn from a split pass's result (`split_pass_live`, `split_pass_gpu`).
    fn live_spawn(ids: Vec<u64>, layout_chunks: &[(usize, usize)], table: &crate::gpu_split::PartTable, result: crate::gpu_split::PassResult) -> LiveSpawn {
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
        LiveSpawn { ids, children, bodies: result.bodies, comps }
    }

    /// M4, `STRESS_GPU_PASS=1`: the split pass with GPU-made inputs (`gpu_pass.rs`): its
    /// parents from the cascade's status, its counts from the GPU's counters (set from the
    /// host here), checked against the host's parents.
    fn split_pass_gpu(&mut self, gpu: &Gpu, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, parents: &[usize]) -> Option<LiveSpawn> {
        if !self.solver.can_split_in_place(0) {
            return None;
        }
        let ids: Vec<u64> = parents.iter().map(|&ci| self.solver.mirror.clusters[ci].id).collect();
        let layout_chunks = self.solver.gpu_chunks().to_vec();
        let table = self.part_table(gpu, engine)?;
        let e = engine.borrow();
        let (bodies_buf, cols_buf, max_bodies) = e.spawn_buffers()?;
        let max_colliders = e.collider_capacity();
        let (first_body, first_collider) = e.body_counts();
        let r = self.resident.as_ref()?;
        let colliders = r.colliders.as_ref().filter(|c| c.capacity() >= max_colliders).map(|c| (c, &r.ingest.collider_info));
        let refreshed = colliders.is_some();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let target = crate::gpu_split::SpawnTarget { bodies: bodies_buf, colliders: cols_buf, max_bodies };
        self.solver.encode_gpu_split_pass(gpu, &mut encoder, target, max_colliders, &table, r.qs.layout(), colliders, false)?;
        self.solver.set_gpu_pass_counters(gpu, first_body, first_collider);
        gpu.queue.submit([encoder.finish()]);
        let target = crate::gpu_split::SpawnTarget { bodies: bodies_buf, colliders: cols_buf, max_bodies };
        let (result, gpu_parents) = self.solver.read_gpu_split_pass(gpu, &target)?;
        drop(e);
        if gpu_parents != parents {
            eprintln!("gpu split pass: parents {gpu_parents:?}, the host's {parents:?}");
            return None;
        }
        if result.aborted {
            return None;
        }
        if let Some(r) = self.resident.as_mut() {
            r.colliders_refreshed = refreshed;
        }
        Some(Self::live_spawn(ids, &layout_chunks, &table, result))
    }

    /// The engine adopts a frame log's split (`apply_frame_log`) before the host's mirror
    /// follows it: the children's cluster ids as the GPU gave them (`layout_plan`: from the
    /// last parent split to the first, `next_id` on), their signatures from their chunks.
    fn adopt_from_log(&mut self, engine: &std::rc::Rc<std::cell::RefCell<dyn crate::resident::ResidentEngine>>, structures: &[usize], s: &LiveSpawn, next_id: u64) -> Result<Vec<Vec<u64>>, String> {
        let m = &self.solver.mirror;
        let mut child_ids: Vec<Vec<u64>> = vec![Vec::new(); structures.len()];
        let mut next = next_id;
        for p in (0..structures.len()).rev() {
            if s.comps[p].len() < 2 {
                continue;
            }
            child_ids[p] = (0..s.comps[p].len() as u64).map(|k| next + k).collect();
            next += s.comps[p].len() as u64;
        }
        let mut adopted = Vec::new();
        let mut retired = Vec::new();
        let mut at = 0;
        for (p, &structure) in structures.iter().enumerate() {
            if s.comps[p].len() < 2 {
                continue;
            }
            retired.push(BodyKey::Cluster(s.ids[p]));
            for (k, chunks) in s.children[p].iter().enumerate() {
                let g = &s.bodies[at];
                at += 1;
                let st = &m.structures[structure];
                let anchored = s.comps[p][k].iter().any(|&c| st.chunks[c].support != stress_ref::scene::Support::None);
                let signature = {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    (structure, &s.comps[p][k], anchored).hash(&mut h);
                    h.finish()
                };
                adopted.push(crate::resident::AdoptedBody {
                    key: BodyKey::Cluster(child_ids[p][k]),
                    signature,
                    fixed: anchored,
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
        engine.borrow_mut().adopt_spawned(adopted, &retired)?;
        Ok(child_ids)
    }

    /// The host follows the GPU splits the engine has adopted (`adopt_from_log`), in order:
    /// the mirror split into the GPU's components (the children's ids checked against the
    /// GPU's), the layout's bookkeeping; then any engine write-back that waited for them.
    pub(super) fn flush_pending(&mut self, gpu: &Gpu) -> Result<(), String> {
        if self.pending_splits.is_empty() && !self.pending_write_back {
            return Ok(());
        }
        let trace = std::env::var("STRESS_GPU_QS_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        for p in std::mem::take(&mut self.pending_splits) {
            let resident = self.resident.as_ref().map(|r| r.qs.layout());
            let l = crate::solver::LiveSplit { comps: p.comps, run: None, resident };
            self.solver.split_resident_with(gpu, &p.parents, Some(l))?;
            // The mirror named the children as the GPU did.
            for ids in p.child_ids.iter().filter(|ids| !ids.is_empty()) {
                if !ids.iter().all(|id| self.solver.mirror.clusters.iter().any(|c| c.id == *id)) {
                    return Err(format!("GPU split: children {ids:?} not in the host's mirror"));
                }
            }
            if let Some(r) = self.resident.as_mut() {
                r.qs.adopt_layout(&self.solver);
                r.layout_generation = self.solver.layout_generation;
                r.gpu_laid_out = true;
            }
            self.solver.adopt_gpu_tables();
        }
        let t1 = std::time::Instant::now();
        if std::mem::take(&mut self.pending_write_back) {
            for cl in &mut self.solver.mirror.clusters {
                cl.driven = true;
            }
            self.with_shell_mut(|w| w.coupled_write_back(&[]));
        }
        if trace {
            eprintln!("flush: splits {:.1} ms, write-back {:.1} ms", (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
        }
        self.check_colliders(gpu);
        Ok(())
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
