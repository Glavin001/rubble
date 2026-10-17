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
    pub fn new(gpu: &Gpu, solver: &GpuSolver, frames: &[Option<EngineFrame>], dt: f64) -> ResidentQs {
        let k = |e: &str| gpu.compute(&shaders::RESIDENT, e, &BINDINGS);
        let mut r = ResidentQs {
            poses: k("resident_poses"),
            filter: k("resident_filter"),
            static_loads: k("resident_static_loads"),
            params: ResidentParams::default(),
            ctab: gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("resident ctab"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            chunks: Vec::new(),
        };
        r.rebuild(gpu, solver, frames, dt);
        r
    }

    /// New tables after the solver's layout changed (a split, a rebuild): the carried
    /// pulses move with their chunks.
    pub fn rebuild(&mut self, gpu: &Gpu, solver: &GpuSolver, frames: &[Option<EngineFrame>], dt: f64) {
        let m: &ReferenceSolver = &solver.mirror;
        let chunks = solver.gpu_chunks().to_vec();
        let n = chunks.len();
        let islands = m.clusters.len();
        let cap = pulse_capacity();
        let props_base = 0u32;
        let engine_base = props_base + n as u32;
        let eload_base = engine_base + 3 * islands as u32;
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
            match frames.get(ci).copied().flatten() {
                Some(f) => {
                    let p = f.principal.conjugate();
                    t[at] = [f32::from_bits(f.body), 0.0, 0.0, 0.0];
                    t[at + 1] = [p.x as f32, p.y as f32, p.z as f32, p.w as f32];
                    t[at + 2] = f4(f.com, 0.0);
                }
                None => t[at] = [f32::from_bits(NONE), 0.0, 0.0, 0.0],
            }
        }
        // Carried pulses, by (structure, chunk).
        if self.params.chunk_count > 0 && self.params.pulse_cap == cap {
            let old: Vec<[f32; 4]> = gpu.read(&self.ctab);
            let at: HashMap<(usize, usize), usize> = chunks.iter().enumerate().map(|(k, &sc)| (sc, k)).collect();
            for (ok, sc) in self.chunks.iter().enumerate() {
                let count = old[self.params.pulse_count_base as usize + ok][0].to_bits() as usize;
                let Some(&k) = at.get(sc) else { continue };
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
            island_count: islands as u32,
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
        };
        self.ctab = gpu.storage("resident ctab", &t);
        self.chunks = chunks;
    }

    fn bind(&self, gpu: &Gpu, k: &Kernel, params: &wgpu::Buffer, ingest: &EngineIngest, solver: &GpuSolver) -> wgpu::BindGroup {
        let (islands, chunks, state, scratch, _) = solver.resident_buffers();
        gpu.bind(&k.group, &[params, &ingest.manifolds, &ingest.sort, islands, chunks, state, scratch, &self.ctab, &ingest.start_bodies])
    }

    fn dispatch(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, k: &Kernel, size: u32, ingest: &EngineIngest, solver: &GpuSolver) {
        let params = gpu.uniform("resident params", &self.params);
        let bind = self.bind(gpu, k, &params, ingest, solver);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&k.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(size, 1, 1);
    }

    /// Driven clusters' poses from the engine's frame-start bodies (after any write of
    /// the island records from the host).
    pub fn encode_poses(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch(gpu, encoder, &self.poses, groups(self.params.island_count as usize, 64), ingest, solver);
    }

    /// The frame's load filter (once per frame: it advances the carried pulses).
    pub fn encode_filter(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch(gpu, encoder, &self.filter, groups(self.params.chunk_count as usize, 64), ingest, solver);
    }

    /// The static loads of the islands flagged for a solve.
    pub fn encode_static_loads(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, ingest: &EngineIngest, solver: &GpuSolver) {
        self.dispatch(gpu, encoder, &self.static_loads, self.params.island_count.max(1), ingest, solver);
    }

    /// Diagnostic read-back: each GPU chunk's frame loads (world force, moment about its
    /// displaced centre).
    pub fn read_frame_loads(&self, gpu: &Gpu) -> Vec<(Vec3, Vec3)> {
        let n = self.params.chunk_count as usize;
        let v: Vec<[f32; 4]> = gpu.read_range(&self.ctab, self.params.eload_base as usize, 2 * n);
        let p = |a: [f32; 4]| Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        (0..n).map(|k| (p(v[2 * k]), p(v[2 * k + 1]))).collect()
    }

    /// Err when a chunk's carried pulses overflowed their slots (read: one float4).
    pub fn check(&self, gpu: &Gpu) -> Result<(), String> {
        let s: Vec<[f32; 4]> = gpu.read_range(&self.ctab, self.params.status_base as usize, 1);
        if s[0][0].to_bits() != 0 {
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
    /// The engine body standing for `key` (None: none, or fixed).
    fn engine_frame(&self, key: stress_ref::engine::BodyKey) -> Option<EngineFrame>;
    /// The substeps a frame of `dt` takes (the engine's CCD bound).
    fn resident_substeps(&mut self, dt: f64) -> u32;
    /// One frame of `dt` in `n` substeps (nothing frozen, no impulses), each substep's
    /// contact rows into `ingest`, then merged; no read-back.
    fn step_resident(&mut self, gpu: &Gpu, dt: f64, n: u32, ingest: &EngineIngest) -> Result<(), String>;
    /// The frame's end: wait for the engine's work and report its overflow, if any (the
    /// frame's one engine read).
    fn resident_finish(&mut self) -> Result<(), String>;
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
