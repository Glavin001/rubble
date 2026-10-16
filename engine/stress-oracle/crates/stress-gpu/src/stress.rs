//! The GPU stress solver's explicit substep (`shaders/stress_step.slang`), built from a
//! `stress_ref::solver::ReferenceSolver` so both solve the identical model data.
//!
//! Scope so far: explicit mode, intact bonds (no fracture yet), anchored clusters (their
//! frame is the static world frame), constant external loads (gravity). Anything else
//! is refused at construction rather than solved differently.

use stress_ref::math::{Mat3, Vec3};
use stress_ref::scene::Support;
use stress_ref::solver::ReferenceSolver;

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    chunk_count: u32,
    bond_count: u32,
    dt: f32,
    substeps: u32,
}

/// `Bond` of stress_step.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBond {
    chunks: [u32; 4],
    t1: [f32; 4],
    t2: [f32; 4],
    normal: [f32; 4],
    ra: [f32; 4],
    rb: [f32; 4],
    k_lin: [f32; 4],
    k_ang: [f32; 4],
    c_lin: [f32; 4],
    c_ang: [f32; 4],
    section: [f32; 4],
}

/// `Chunk` of stress_step.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuChunk {
    inv_inertia: [[f32; 4]; 3],
    force: [f32; 4],
    moment: [f32; 4],
    support: [u32; 4],
}

/// `RenderChunk` of stress_render.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderChunk {
    pub center: [f32; 4],
    pub half: [f32; 4],
    pub rot: [[f32; 4]; 3],
    pub pose_pos: [f32; 4],
    pub pose_rot: [[f32; 4]; 3],
}

const STEP_BINDINGS: [Binding; 8] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
    Binding::StorageRw,
    Binding::Storage,
];

fn v4(v: Vec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

fn rows(m: &Mat3) -> [[f32; 4]; 3] {
    std::array::from_fn(|r| [m.m[r][0] as f32, m.m[r][1] as f32, m.m[r][2] as f32, 0.0])
}

/// The explicit stress solve on the GPU.
pub struct GpuStress {
    /// `(structure, chunk)` of every GPU chunk, in GPU order.
    pub chunks: Vec<(usize, usize)>,
    pub bond_count: usize,
    /// The substep (s), the reference solver's stable step in f32.
    pub dt: f32,
    pub render_chunks: Vec<RenderChunk>,
    /// Per island (cluster): chunk range and bond range in GPU order.
    pub islands: Vec<[u32; 4]>,
    params: wgpu::Buffer,
    step: StepParams,
    /// 4 per chunk: u (w: largest bond stress), theta, v, w.
    pub state: wgpu::Buffer,
    pub bond_loads: wgpu::Buffer,
    bond_forces: Kernel,
    chunk_integrate: Kernel,
    island_substeps: Kernel,
    island_shared_substeps: Kernel,
    bind_bonds: wgpu::BindGroup,
    bind_chunks: wgpu::BindGroup,
    /// Islands that fit threadgroup memory, and the rest: (bind group, island count).
    bind_shared: (wgpu::BindGroup, u32),
    bind_device: (wgpu::BindGroup, u32),
}

/// Island capacity of `island_shared_substeps` (stress_step.slang).
pub const SHARED_CHUNKS: u32 = 160;
pub const SHARED_BONDS: u32 = 400;

impl GpuStress {
    pub fn new(gpu: &Gpu, solver: &ReferenceSolver) -> Result<GpuStress, String> {
        if solver.config.fracture {
            return Err("fracture is not on the GPU yet: run with sim.fracture = false".into());
        }
        // Chunks stored by cluster (island), each in its cluster's order.
        let mut index: Vec<Vec<u32>> = solver.structures.iter().map(|st| vec![u32::MAX; st.chunks.len()]).collect();
        let mut chunks = Vec::new();
        let mut islands = Vec::new();
        for cl in &solver.clusters {
            let begin = chunks.len() as u32;
            for &c in &cl.chunks {
                index[cl.structure][c] = chunks.len() as u32;
                chunks.push((cl.structure, c));
            }
            islands.push([begin, chunks.len() as u32, 0, 0]);
        }
        let g = solver.config.gravity;
        let mut gpu_chunks = vec![GpuChunk::default(); chunks.len()];
        let mut render_chunks = vec![RenderChunk::default(); chunks.len()];
        let mut state = vec![[0f32; 4]; 4 * chunks.len()];
        for (k, &(s, c)) in chunks.iter().enumerate() {
            let data = &solver.structures[s].chunks[c];
            let st = &solver.chunks[s][c];
            let cl = &solver.clusters[st.cluster];
            if !cl.anchored {
                return Err(format!("structure {s} chunk {c}: free (non-anchored) clusters are not on the GPU yet"));
            }
            let f = cl.pose.inverse_transform_vector(g * data.mass);
            gpu_chunks[k] = GpuChunk {
                inv_inertia: rows(&data.inv_inertia),
                force: v4(f, 1.0 / (data.mass * st.inertia_scale)),
                moment: [0.0, 0.0, 0.0, (1.0 / st.inertia_scale) as f32],
                support: [match data.support {
                    Support::None => 0,
                    Support::Fixed => 1,
                    Support::Pinned => 2,
                }, 0, 0, 0],
            };
            render_chunks[k] = RenderChunk {
                center: v4(data.center, 0.0),
                half: v4(data.half_extents, 0.0),
                rot: rows(&data.rotation),
                pose_pos: v4(cl.pose.position, 0.0),
                pose_rot: rows(&cl.pose.rotation.to_mat3()),
            };
            for (field, x) in [st.u, st.th, st.v, st.w].into_iter().enumerate() {
                state[4 * k + field] = v4(x, 0.0);
            }
        }
        // Bonds in cluster order, each cluster's in its own order (the reference sums
        // a chunk's internal loads in this order).
        let mut bonds = Vec::new();
        let mut incident = vec![Vec::new(); chunks.len()];
        for (ci, cl) in solver.clusters.iter().enumerate() {
            let s = cl.structure;
            islands[ci][2] = bonds.len() as u32;
            for &bi in &cl.bonds {
                let b = &solver.bonds[s][bi];
                if b.rebar.is_some() {
                    return Err(format!("structure {s} bond {bi}: rebar is not on the GPU yet"));
                }
                if b.joint.damage != 0.0 || b.joint.crush != 0.0 {
                    return Err(format!("structure {s} bond {bi}: damaged joints are not on the GPU yet"));
                }
                let geo = &b.geometry;
                let (a, bb) = (index[s][geo.a], index[s][geo.b]);
                let i = bonds.len() as u32;
                incident[a as usize].push(2 * i);
                incident[bb as usize].push(2 * i + 1);
                let k = &b.stiffness;
                bonds.push(GpuBond {
                    chunks: [a, bb, 0, 0],
                    t1: v4(geo.t1, 0.0),
                    t2: v4(geo.t2, 0.0),
                    normal: v4(geo.normal, 0.0),
                    ra: v4(geo.ra, 0.0),
                    rb: v4(geo.rb, 0.0),
                    k_lin: v4(Vec3::new(k.ks, k.ks, k.kn), 0.0),
                    k_ang: v4(Vec3::new(k.kb_t1, k.kb_t2, k.kt), 0.0),
                    // Intact joint: every secant factor is 1.
                    c_lin: v4(b.damping.lin, 0.0),
                    c_ang: v4(b.damping.ang, 0.0),
                    section: [geo.area as f32, geo.s_t1 as f32, geo.s_t2 as f32, 0.0],
                });
            }
            islands[ci][3] = bonds.len() as u32;
        }
        let mut start = vec![0u32];
        let mut list = Vec::new();
        for inc in &incident {
            list.extend_from_slice(inc);
            start.push(list.len() as u32);
        }
        let dt = solver.stable_dt() as f32;
        let step = StepParams { chunk_count: chunks.len() as u32, bond_count: bonds.len() as u32, dt, substeps: 1 };
        let params = gpu.uniform("stress params", &step);
        let bonds_buf = gpu.storage("bonds", &bonds);
        let chunks_buf = gpu.storage("chunks", &gpu_chunks);
        let start = gpu.storage("chunk bond start", &start);
        let list = gpu.storage("chunk bonds", &list);
        let state = gpu.storage("state", &state);
        let bond_loads = gpu.storage("bond loads", &vec![[0f32; 4]; 3 * bonds.len()]);
        let bond_forces = gpu.compute(&shaders::STRESS_STEP, "bond_forces", &STEP_BINDINGS);
        let chunk_integrate = gpu.compute(&shaders::STRESS_STEP, "chunk_integrate", &STEP_BINDINGS);
        let island_substeps = gpu.compute(&shaders::STRESS_STEP, "island_substeps", &STEP_BINDINGS);
        let island_shared_substeps = gpu.compute(&shaders::STRESS_STEP, "island_shared_substeps", &STEP_BINDINGS);
        let fits = |i: &&[u32; 4]| i[1] - i[0] <= SHARED_CHUNKS && i[3] - i[2] <= SHARED_BONDS;
        let shared_islands: Vec<[u32; 4]> = islands.iter().filter(fits).copied().collect();
        let device_islands: Vec<[u32; 4]> = islands.iter().filter(|i| !fits(i)).copied().collect();
        let (shared_buf, device_buf) = (gpu.storage("shared islands", &shared_islands), gpu.storage("device islands", &device_islands));
        let buffers = [&params, &bonds_buf, &chunks_buf, &start, &list, &state, &bond_loads, &shared_buf];
        let bind_bonds = gpu.bind(&bond_forces.group, &buffers);
        let bind_chunks = gpu.bind(&chunk_integrate.group, &buffers);
        let bind_shared = (gpu.bind(&island_shared_substeps.group, &buffers), shared_islands.len() as u32);
        let buffers = [&params, &bonds_buf, &chunks_buf, &start, &list, &state, &bond_loads, &device_buf];
        let bind_device = (gpu.bind(&island_substeps.group, &buffers), device_islands.len() as u32);
        Ok(GpuStress {
            chunks,
            bond_count: bonds.len(),
            dt,
            render_chunks,
            islands,
            params,
            step,
            state,
            bond_loads,
            bond_forces,
            chunk_integrate,
            island_substeps,
            island_shared_substeps,
            bind_bonds,
            bind_chunks,
            bind_shared,
            bind_device,
        })
    }

    /// Record `substeps` explicit substeps (two dispatches each) into one compute pass.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, substeps: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("stress substeps"), timestamp_writes: None });
        for _ in 0..substeps {
            self.bond_forces.dispatch(&mut pass, &self.bind_bonds, self.bond_count);
            self.chunk_integrate.dispatch(&mut pass, &self.bind_chunks, self.chunks.len());
        }
    }

    /// Run `substeps` substeps, one dispatch per phase per substep.
    pub fn run(&self, gpu: &Gpu, substeps: usize) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        self.record(&mut encoder, substeps);
        gpu.queue.submit([encoder.finish()]);
    }

    /// Run `substeps` substeps in one dispatch per island kind: one threadgroup per
    /// island, looping over the substeps with a barrier between phases; islands that
    /// fit threadgroup memory keep their state there.
    pub fn run_islands(&mut self, gpu: &Gpu, substeps: usize) {
        self.step.substeps = substeps as u32;
        gpu.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.step));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("island substeps"), timestamp_writes: None });
            for (kernel, (bind, count)) in [(&self.island_shared_substeps, &self.bind_shared), (&self.island_substeps, &self.bind_device)] {
                if *count > 0 {
                    pass.set_pipeline(&kernel.pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(*count, 1, 1);
                }
            }
        }
        gpu.queue.submit([encoder.finish()]);
    }

    /// The hidden state, 4 per chunk: u (w: largest bond stress), theta, v, w.
    pub fn read_state(&self, gpu: &Gpu) -> Vec<[f32; 4]> {
        gpu.read(&self.state)
    }
}
