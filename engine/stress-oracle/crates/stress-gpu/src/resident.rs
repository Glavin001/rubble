//! The resident quasi-static frame's load path (`shaders/resident.slang`,
//! docs/stress-fracture-gpu-resident.md phases 2b and 4): the engine's contact manifolds
//! (`ingest`) to the static solve's loads on the GPU. Driven clusters take their pose
//! from the engine's bodies (`resident_poses`), the load filter runs per chunk
//! (`resident_filter`, coupled.rs `LoadFilter` with `frame_average_terms`, pulses carried
//! between frames in a fixed number of slots per chunk) and each solved island's loads
//! are formed in its body frame (`resident_static_loads`, statics.rs `static_loads`).

use std::collections::HashMap;

use stress_ref::math::{Quat, Vec3};
use stress_ref::solver::ReferenceSolver;

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::ingest::{EngineIngest, NONE};
use crate::shaders;
use crate::solver::GpuSolver;

/// Pulses carried past their frame per chunk (`STRESS_RESIDENT_PULSES`, default 4). A
/// chunk with more fails the frame (`ResidentQs::check`): an explicit limit, never a
/// dropped load.
pub fn pulse_capacity() -> u32 {
    std::env::var("STRESS_RESIDENT_PULSES").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ResidentParams {
    chunk_count: u32,
    island_count: u32,
    sload_base: u32,
    rml: u32,
    gravity: [f32; 4],
    dt: f32,
    props_base: u32,
    engine_base: u32,
    eload_base: u32,
    pulse_base: u32,
    pulse_count_base: u32,
    pulse_cap: u32,
    status_base: u32,
    /// 1: an impact loads its frame with its Hertz peak force, not its mean over the frame
    /// (`STRESS_QS_IMPACT_PEAK=1`, a measurement).
    impact_peak: u32,
    pad: [u32; 2],
}

/// An engine body standing for a cluster: its index, its principal frame and centre of
/// mass in the cluster's body frame (the engine body is at the centre of mass, along the
/// principal axes).
#[derive(Clone, Copy, Debug)]
pub struct EngineFrame {
    pub body: u32,
    pub principal: Quat,
    pub com: Vec3,
}

const BINDINGS: [Binding; 9] = [
    Binding::Uniform,   // params
    Binding::Storage,   // manifolds
    Binding::Storage,   // sort
    Binding::StorageRw, // islands
    Binding::Storage,   // chunk statics
    Binding::Storage,   // state
    Binding::StorageRw, // scratch
    Binding::StorageRw, // ctab
    Binding::Storage,   // engine start bodies
];

pub struct ResidentQs {
    poses: Kernel,
    filter: Kernel,
    static_loads: Kernel,
    params: ResidentParams,
    pub ctab: wgpu::Buffer,
    /// The GPU chunks the tables were built for (pulses follow chunks across rebuilds).
    chunks: Vec<(usize, usize)>,
}

fn f4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w]
}

impl ResidentQs {
    /// The tables of the solver's current layout; `frames[ci]`: cluster ci's engine body
    /// (None: anchored, or not the engine's). `dt` is the frame.
    pub fn new(gpu: &Gpu, solver: &GpuSolver, frames: &[Option<EngineFrame>], bodies: &[Option<[u32; 3]>], dt: f64) -> ResidentQs {
        let k = |e: &str| gpu.compute(&shaders::RESIDENT, e, &BINDINGS);
        let mut r = ResidentQs {
            poses: k("resident_poses"),
            filter: k("resident_filter"),
            static_loads: k("resident_static_loads"),
            params: ResidentParams::default(),
            ctab: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("resident ctab"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            chunks: Vec::new(),
        };
        r.rebuild(gpu, solver, frames, bodies, dt);
        r
    }

    /// New tables after the solver's layout changed (a split, a rebuild): the carried
    /// pulses move with their chunks.
    /// `bodies[ci]`: cluster ci's engine body and collider range, fixed bodies included
    /// (`ResidentEngine::retire_record`), which the GPU's split retires.
    pub fn rebuild(&mut self, gpu: &Gpu, solver: &GpuSolver, frames: &[Option<EngineFrame>], bodies: &[Option<[u32; 3]>], dt: f64) {
        let m: &ReferenceSolver = &solver.mirror;
        let chunks = solver.gpu_chunks().to_vec();
        let n = chunks.len();
        let islands = m.clusters.len();
        let cap = pulse_capacity();
        let props_base = 0u32;
        let engine_base = props_base + n as u32;
        // Per island up to the capacity (a split laid out on the GPU adds islands).
        let eload_base = engine_base + 3 * islands.max(solver.island_cap()) as u32;
        let pulse_base = eload_base + 2 * n as u32;
        let pulse_count_base = pulse_base + 3 * cap * n as u32;
        let status_base = pulse_count_base + n as u32;
        let total = status_base as usize + 1;
        let mut t = vec![[0f32; 4]; total];
        for (k, &(s, c)) in chunks.iter().enumerate() {
            let ch = &m.structures[s].chunks[c];
            t[props_base as usize + k] = [ch.youngs_modulus as f32, ch.poisson_ratio as f32, ch.volume().cbrt() as f32, 0.0];
        }
        for ci in 0..islands {
            let at = engine_base as usize + 3 * ci;
            // y-w: its engine body (fixed ones too) and collider range, for the GPU's split.
            let b = bodies.get(ci).copied().flatten().unwrap_or([NONE, 0, 0]);
            let (by, bz, bw) = (f32::from_bits(b[0]), f32::from_bits(b[1]), f32::from_bits(b[2]));
            match frames.get(ci).copied().flatten() {
                Some(f) => {
                    let p = f.principal.conjugate();
                    t[at] = [f32::from_bits(f.body), by, bz, bw];
                    t[at + 1] = [p.x as f32, p.y as f32, p.z as f32, p.w as f32];
                    t[at + 2] = f4(f.com, 0.0);
                }
                None => t[at] = [f32::from_bits(NONE), by, bz, bw],
            }
        }
        // The capacity past the islands: no body (a blank island takes no pose).
        for ci in islands..islands.max(solver.island_cap()) {
            t[engine_base as usize + 3 * ci] = [f32::from_bits(NONE), f32::from_bits(NONE), 0.0, 0.0];
        }
        // Carried pulses and this frame's filtered loads (a rebuild within the frame's
        // cascade keeps them: the filter runs once a frame), by (structure, chunk).
        if self.params.chunk_count > 0 && self.params.pulse_cap == cap {
            let old: Vec<[f32; 4]> = gpu.read(&self.ctab);
            let at: HashMap<(usize, usize), usize> = chunks.iter().enumerate().map(|(k, &sc)| (sc, k)).collect();
            for (ok, sc) in self.chunks.iter().enumerate() {
                let count = old[self.params.pulse_count_base as usize + ok][0].to_bits() as usize;
                let Some(&k) = at.get(sc) else { continue };
                for j in 0..2 {
                    t[eload_base as usize + 2 * k + j] = old[self.params.eload_base as usize + 2 * ok + j];
                }
                let from = self.params.pulse_base as usize + 3 * cap as usize * ok;
                let to = pulse_base as usize + 3 * cap as usize * k;
                t[to..to + 3 * count].copy_from_slice(&old[from..from + 3 * count]);
                t[pulse_count_base as usize + k] = [f32::from_bits(count as u32), 0.0, 0.0, 0.0];
            }
        } else {
            for k in 0..n {
                t[pulse_count_base as usize + k] = [f32::from_bits(0), 0.0, 0.0, 0.0];
            }
        }
        let g = m.config.gravity;
        let (_, _, _, _, sload_base) = solver.resident_buffers();
        self.params = ResidentParams {
            chunk_count: n as u32,
            island_count: if crate::solver::capacity_dispatch() { islands.max(solver.island_cap()) as u32 } else { islands as u32 },
            sload_base,
            rml: m.config.features.rigid_motion_loads as u32,
            gravity: f4(g, (g.norm() * dt) as f32),
            dt: dt as f32,
            props_base,
            engine_base,
            eload_base,
            pulse_base,
            pulse_count_base,
            pulse_cap: cap,
            status_base,
            impact_peak: std::env::var("STRESS_QS_IMPACT_PEAK").is_ok_and(|v| v == "1") as u32,
            pad: [0; 2],
        };
        self.ctab = gpu.storage("resident ctab", &t);
        self.chunks = chunks;
    }

    /// The tables' layout, for a split the GPU lays out (`GpuSolver::split_resident_with`).
    pub fn layout(&self) -> crate::solver::ResidentLayout<'_> {
        let p = &self.params;
        crate::solver::ResidentLayout { ctab: &self.ctab, props_base: p.props_base, eload_base: p.eload_base, pulse_base: p.pulse_base, pulse_count_base: p.pulse_count_base, pulse_cap: p.pulse_cap, engine_base: p.engine_base }
    }

    /// After a split laid out on the GPU (which moved the tables' chunk and island entries
    /// itself): the tables' island count follows, nothing rebuilt.
    pub fn adopt_layout(&mut self, solver: &GpuSolver) {
        let islands = solver.mirror.clusters.len();
        self.params.island_count = if crate::solver::capacity_dispatch() { islands.max(solver.island_cap()) as u32 } else { islands as u32 };
        self.chunks = solver.gpu_chunks().to_vec();
    }

    fn bind(&self, gpu: &Gpu, k: &Kernel, params: &wgpu::Buffer, ingest: &EngineIngest, solver: &GpuSolver) -> wgpu::BindGroup {
        let (islands, chunks, state, scratch, _) = solver.resident_buffers();
        gpu.bind(&k.group, &[params, &ingest.manifolds, &ingest.sort, islands, chunks, state, scratch, &self.ctab, &ingest.start_bodies])
    }

    fn dispatch(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, k: &Kernel, size: u32, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch_with(gpu, encoder, k, size, None, ingest, solver);
    }

    /// `dispatch`, or indirect from `args` at `offset` (the live islands: `island_args`).
    fn dispatch_with(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, k: &Kernel, size: u32, indirect: Option<u64>, ingest: &EngineIngest, solver: &GpuSolver) {
        let params = gpu.uniform("resident params", &self.params);
        let bind = self.bind(gpu, k, &params, ingest, solver);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&k.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        match (indirect, solver.island_args()) {
            (Some(offset), Some(args)) => pass.dispatch_workgroups_indirect(args, offset),
            _ => pass.dispatch_workgroups(size, 1, 1),
        }
    }

    /// Driven clusters' poses from the engine's frame-start bodies (after any write of
    /// the island records from the host).
    pub fn encode_poses(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch_with(gpu, encoder, &self.poses, groups(self.params.island_count as usize, 64), Some(16), ingest, solver);
    }

    /// The frame's load filter (once per frame: it advances the carried pulses).
    pub fn encode_filter(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch(gpu, encoder, &self.filter, groups(self.params.chunk_count as usize, 64), ingest, solver);
    }

    /// The static loads of the islands flagged for a solve.
    pub fn encode_static_loads(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch_with(gpu, encoder, &self.static_loads, self.params.island_count.max(1), Some(0), ingest, solver);
    }

    /// Diagnostic read-back: each GPU chunk's frame loads (world force, moment about its
    /// displaced centre).
    pub fn read_frame_loads(&self, gpu: &Gpu) -> Vec<(Vec3, Vec3)> {
        let n = self.params.chunk_count as usize;
        let v: Vec<[f32; 4]> = gpu.read_range(&self.ctab, self.params.eload_base as usize, 2 * n);
        let p = |a: [f32; 4]| Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        (0..n).map(|k| (p(v[2 * k]), p(v[2 * k + 1]))).collect()
    }

    /// The status word's range (for the frame's one read).
    pub fn status_range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.ctab, 16 * self.params.status_base as u64, 16)
    }

    /// Err when a chunk's carried pulses overflowed their slots (the status word read).
    pub fn check(&self, status: &[u8]) -> Result<(), String> {
        let s: &[u32] = bytemuck::cast_slice(status);
        if s[0] != 0 {
            return Err(format!("resident frame: a chunk carries more than {} impact pulses past their frame (STRESS_RESIDENT_PULSES)", self.params.pulse_cap));
        }
        Ok(())
    }
}

/// What the other side of a contact is, as the load filter sees it: (mass, Young's
/// modulus, radius) (world_coupling.rs `engine_contacts_to_impulses`).
pub type OtherProps<'a> = &'a dyn Fn(Option<stress_ref::engine::BodyKey>) -> [f32; 3];

/// An engine the resident frame runs without host round trips: on the stress solver's
/// device, its buffers read where they are.
pub trait ResidentEngine: stress_ref::engine::RigidEngine {
    fn engine_buffers(&self) -> crate::ingest::EngineBuffers<'_>;
    /// (contact rows per step, bodies) the engine holds at most.
    fn engine_capacity(&self) -> (u32, u32);
    /// Changes whenever the engine's bodies or colliders do (the collider table follows).
    fn layout_version(&self) -> u64;
    /// The collider table: each collider's GPU chunk (`chunk_of`), body and the other
    /// side's properties (`other`).
    fn collider_table(&self, chunk_of: &dyn Fn(usize, usize) -> Option<u32>, other: OtherProps<'_>) -> Vec<crate::ingest::ColliderInfo>;
    /// The collider table's entries for the colliders `range` only.
    fn collider_table_range(&self, range: std::ops::Range<u32>, chunk_of: &dyn Fn(usize, usize) -> Option<u32>, other: OtherProps<'_>) -> Vec<crate::ingest::ColliderInfo> {
        self.collider_table(chunk_of, other)[range.start as usize..range.end as usize].to_vec()
    }
    /// The colliders whose table entries changed since `layout_version` was `version`
    /// (added or retired), when the engine still knows (None: rebuild the whole table).
    fn collider_changes_since(&self, _version: u64) -> Option<Vec<std::ops::Range<u32>>> {
        None
    }
    /// The contact row offset of the engine's last substep (diagnostic replays).
    fn last_row_offset(&self) -> u32 {
        0
    }
    /// The engine body standing for `key` (None: none, or fixed).
    fn engine_frame(&self, key: stress_ref::engine::BodyKey) -> Option<EngineFrame>;
    /// The substeps a frame of `dt` takes (the engine's CCD bound).
    fn resident_substeps(&mut self, dt: f64) -> u32;
    /// One frame of `dt` in `n` substeps (nothing frozen, no impulses), each substep's
    /// contact rows into `ingest`, then merged; no read-back.
    fn step_resident(&mut self, gpu: &Gpu, dt: f64, n: u32, ingest: &EngineIngest) -> Result<(), String>;
    /// The engine's status for the frame's one read: its step counters first, then
    /// anything the next frame needs from the device (the CCD bodies' state), so that
    /// costs no read of its own.
    fn resident_status(&self) -> Vec<(&wgpu::Buffer, u64, u64)>;
    /// The frame's end, with the status read (`resident_status`'s ranges, in order): the
    /// engine's overflow, if any.
    fn resident_finish(&mut self, status: &[Vec<u8>]) -> Result<(), String>;
    /// The part records of every chunk of every structure (`sizes[s]` chunks each) for
    /// the GPU's body creation (`spawn.slang`), 8 float4 per part. None when a chunk was
    /// never cooked.
    fn part_records(&self, _sizes: &[usize]) -> Option<Vec<Vec<Vec<[[f32; 4]; 8]>>>> {
        None
    }
    /// A body's `BodyState` and its colliders' records, as raw words (for checks).
    fn body_words(&self, _key: stress_ref::engine::BodyKey) -> Option<([[f32; 4]; 7], Vec<[[f32; 4]; 4]>)> {
        None
    }
    /// A body's slot.
    fn body_slot(&self, _key: stress_ref::engine::BodyKey) -> Option<u32> {
        None
    }
    /// The engine's body and collider buffers and body capacity, for bodies made on the
    /// GPU (`spawn.slang`): bodies in two regions, the second at the capacity.
    fn spawn_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        None
    }
    /// A body's slot, centre of mass (its oracle frame) and principal rotation (x, y, z,
    /// w), for the GPU to take its frame from the engine's buffer (`split_frames`).
    fn frame_record(&self, _key: stress_ref::engine::BodyKey) -> Option<(u32, [f32; 3], [f32; 4])> {
        None
    }
    /// A body's slot and collider range, for the GPU to retire it (`spawn_retire`).
    fn retire_record(&self, _key: stress_ref::engine::BodyKey) -> Option<[u32; 3]> {
        None
    }
    /// Bodies the GPU made (`adopted`, at the engine's next slots, in order) and retired
    /// (`retired`): the engine's own bookkeeping follows, with no upload.
    fn adopt_spawned(&mut self, _adopted: Vec<AdoptedBody>, _retired: &[stress_ref::engine::BodyKey]) -> Result<(), String> {
        Err("this engine makes no bodies on the GPU".into())
    }
    /// The engine's collider capacity.
    fn collider_capacity(&self) -> u32 {
        0
    }
    /// The engine's body and collider counts (the next slots).
    fn body_counts(&self) -> (u32, u32) {
        (0, 0)
    }
}

/// A body the GPU made for an engine (`spawn.slang`), for the engine's bookkeeping.
#[derive(Clone, Debug)]
pub struct AdoptedBody {
    pub key: stress_ref::engine::BodyKey,
    pub signature: u64,
    pub fixed: bool,
    pub slot: u32,
    pub first_collider: u32,
    /// Raw `BodyState` and `Collider` words, as written.
    pub body: [[f32; 4]; 7],
    pub colliders: Vec<[[f32; 4]; 4]>,
    /// Each collider's chunk (structure, chunk).
    pub chunks: Vec<(usize, usize)>,
    /// Centre of mass in the body's oracle frame, and the principal rotation (oracle body
    /// frame to the engine body's), as `cook_with` reports them.
    pub com: [f64; 3],
    pub principal: [f64; 4],
}

/// A resident engine shared between the world's shell (as its `RigidEngine`) and the
/// resident frame.
pub struct SharedEngine<T: ResidentEngine>(pub std::rc::Rc<std::cell::RefCell<T>>);

impl<T: ResidentEngine> stress_ref::engine::RigidEngine for SharedEngine<T> {
    fn name(&self) -> &str {
        "shared"
    }
    fn add_ground(&mut self, height: f64) {
        self.0.borrow_mut().add_ground(height)
    }
    fn sync_bodies(&mut self, bodies: &[stress_ref::engine::EngineBody]) {
        self.0.borrow_mut().sync_bodies(bodies)
    }
    fn step(&mut self, dt: f64, frozen: &[stress_ref::engine::BodyKey], impulses: &[(stress_ref::engine::BodyKey, Vec3, Vec3)]) -> Vec<stress_ref::engine::EngineContact> {
        self.0.borrow_mut().step(dt, frozen, impulses)
    }
    fn motion(&self) -> Vec<stress_ref::engine::BodyMotion> {
        self.0.borrow().motion()
    }
    fn set_motion(&mut self, states: &[stress_ref::engine::BodyMotion]) {
        self.0.borrow_mut().set_motion(states)
    }
    fn save(&mut self) {
        self.0.borrow_mut().save()
    }
    fn restore(&mut self) {
        self.0.borrow_mut().restore()
    }
    fn discard(&mut self) {
        self.0.borrow_mut().discard()
    }
    fn set_rest_overlaps(&mut self, overlaps: &[stress_ref::engine::RestOverlap]) -> bool {
        self.0.borrow_mut().set_rest_overlaps(overlaps)
    }
    fn rest_overlaps(&self) -> Vec<stress_ref::engine::RestOverlap> {
        self.0.borrow().rest_overlaps()
    }
}

/// The resident cascade's kernels (`shaders/cascade.slang`).
pub struct CascadeKernels {
    pub begin: Kernel,
    pub gate: Kernel,
    pub fixed: Kernel,
    pub restore: Kernel,
    pub settle: Kernel,
    pub end: Kernel,
    pub backup: Kernel,
}

const CASCADE_BINDINGS: [Binding; 9] = [
    Binding::Uniform,   // params
    Binding::StorageRw, // islands
    Binding::Storage,   // scratch
    Binding::StorageRw, // state
    Binding::StorageRw, // state backup
    Binding::StorageRw, // status
    Binding::StorageRw, // repeat gate
    Binding::Storage,   // bond records
    Binding::StorageRw, // bond records backup
];

impl CascadeKernels {
    pub fn new(gpu: &Gpu) -> CascadeKernels {
        let k = |e: &str| gpu.compute(&shaders::CASCADE, e, &CASCADE_BINDINGS);
        CascadeKernels { begin: k("cascade_begin"), gate: k("cascade_gate"), fixed: k("cascade_fixed"), restore: k("cascade_restore"), settle: k("cascade_settle"), end: k("cascade_end"), backup: k("cascade_backup") }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CascadeParams {
    pub island_count: u32,
    pub result_base: u32,
    pub pass_index: u32,
    pub sload_base: u32,
    pub gate_flag_base: u32,
    pub bond_words: u32,
    pub certify: u32,
    pub unconverged: u32,
}

/// First per-island word of the cascade's status record (cascade.slang `ST_ISLANDS`).
pub const CASCADE_ST_ISLANDS: usize = 16;

/// A submitted resident cascade: its status record to read.
pub struct CascadeRun {
    pub status: wgpu::Buffer,
    pub words: usize,
    pub islands: usize,
}

impl CascadeRun {
    pub fn range(&self) -> (&wgpu::Buffer, u64, u64) {
        (&self.status, 0, 4 * self.words as u64)
    }

    pub fn outcome(&self, bytes: &[u8]) -> CascadeOutcome {
        CascadeOutcome::from_status(bytemuck::cast_slice(bytes), self.islands)
    }
}

/// What a resident cascade did.
#[derive(Clone, Debug, Default)]
pub struct CascadeOutcome {
    pub passes: usize,
    pub unconverged: usize,
    /// The pass at which a bond disconnected (the cascade stopped there), and the
    /// islands (clusters) that disconnected in it.
    pub disconnected_at: Option<usize>,
    pub disconnected: Vec<usize>,
    /// Damage grew in the last pass run (the cascade would continue).
    pub damaged: bool,
    /// The clusters solved in the last pass run, and those that converged.
    pub solved: Vec<usize>,
    pub converged: Vec<usize>,
    /// Islands the repeat gate skipped: repeats of an unconverged solve, of a fixed
    /// point, and converged repeats certified without a solve.
    pub skipped_unconverged: usize,
    pub skipped_fixed: usize,
    pub certified: usize,
}

impl CascadeOutcome {
    pub fn from_status(st: &[u32], islands: usize) -> CascadeOutcome {
        let per = &st[CASCADE_ST_ISLANDS..CASCADE_ST_ISLANDS + islands];
        CascadeOutcome {
            passes: st[2] as usize,
            unconverged: st[3] as usize,
            disconnected_at: (st[1] != 0).then(|| st[1] as usize - 1),
            disconnected: (0..islands).filter(|&i| per[i] & 8 != 0).collect(),
            damaged: st[4] != 0,
            solved: (0..islands).filter(|&i| per[i] & 1 != 0).collect(),
            converged: (0..islands).filter(|&i| per[i] & 3 == 3 || per[i] & 32 != 0).collect(),
            skipped_unconverged: per.iter().filter(|&&p| p & 16 != 0).count(),
            skipped_fixed: per.iter().filter(|&&p| p & 32 != 0).count(),
            certified: per.iter().filter(|&&p| p & 64 != 0).count(),
        }
    }
}
