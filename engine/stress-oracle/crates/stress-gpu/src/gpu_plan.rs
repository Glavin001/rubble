//! The contact plan on the GPU (`shaders/plan.slang`, docs/stress-fracture-gpu-resident.md
//! phase 6a): `contacts::plan` from the GPU's chunk and island state, so an island segment
//! can plan without a download. The owned clusters (those an impact island holds, not the
//! engine) come from the host's routing until that moves to the GPU too (phase 6d).
//!
//! `STRESS_GPU_PLAN_CHECK=1` runs it beside every host plan (`GpuSolver::plan_for`) and
//! records how the two differ (`PlanCheck`).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PlanParams {
    chunk_count: u32,
    island_count: u32,
    job_count: u32,
    impactor_count: u32,
    duration: f32,
    g: f32,
    gz: f32,
    ground_height: f32,
    has_ground: u32,
    reach_base: u32,
    cp_base: u32,
    job_base: u32,
    count_base: u32,
    pair_base: u32,
    flag_base: u32,
    owned_base: u32,
}

const BINDINGS: [Binding; 7] = [
    Binding::Uniform,   // params
    Binding::Storage,   // chunk statics
    Binding::Storage,   // islands
    Binding::Storage,   // state
    Binding::Storage,   // impactors
    Binding::StorageRw, // pm (motion, reach)
    Binding::StorageRw, // pu (pairs, jobs, counts, flags)
];

/// What a GPU plan holds (read back).
#[derive(Clone, Debug, Default)]
pub struct GpuPlanOut {
    pub pairs: Vec<(u32, u32)>,
    pub ground: Vec<bool>,
    pub impactor: Vec<Vec<bool>>,
    pub rotation_checked: Vec<bool>,
    pub budget: Vec<f32>,
}

/// How GPU plans compared with the host's (`STRESS_GPU_PLAN_CHECK=1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlanCheck {
    pub plans: u64,
    pub host_pairs: u64,
    /// Host pairs the GPU plan missed (it must hold every one).
    pub missed_pairs: u64,
    /// GPU pairs beyond the host's (borderline, conservative).
    pub extra_pairs: u64,
    pub missed_ground: u64,
    pub extra_ground: u64,
    pub missed_impactor: u64,
    pub extra_impactor: u64,
    pub rotation_mismatch: u64,
    /// Plans whose pairs (as sets equal) came in another order.
    pub order_mismatch: u64,
    /// Largest relative budget difference.
    pub budget: f64,
}

pub struct GpuPlanner {
    motion: Kernel,
    reach: Kernel,
    count: Kernel,
    scan: Kernel,
    write: Kernel,
    side: Kernel,
}

impl GpuPlanner {
    pub fn new(gpu: &Gpu) -> GpuPlanner {
        let k = |e: &str| gpu.compute(&shaders::PLAN, e, &BINDINGS);
        GpuPlanner { motion: k("plan_motion"), reach: k("plan_reach"), count: k("plan_pair_count"), scan: k("plan_scan"), write: k("plan_pair_write"), side: k("plan_side") }
    }

    /// A plan over `duration` from the GPU state: `ranges[i]` island i's chunk range,
    /// `owned[i]` whether an impact island holds it, `anchored[i]`; `sorted[i]` its chunks
    /// in the host's order (by structure, chunk). Read back.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &self,
        gpu: &Gpu,
        buffers: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        chunk_count: usize,
        owned: &[bool],
        anchored: &[bool],
        sorted: &[Vec<u32>],
        impactor_count: usize,
        duration: f64,
        gravity: [f64; 3],
        ground: Option<f64>,
    ) -> GpuPlanOut {
        let islands = owned.len();
        // Owned cluster pairs in the host loop's order (both anchored: none), and a job per
        // pair and chunk of its first cluster.
        let own: Vec<usize> = (0..islands).filter(|&i| owned[i]).collect();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (x, &a) in own.iter().enumerate() {
            for &b in &own[x + 1..] {
                if !(anchored[a] && anchored[b]) {
                    pairs.push((a as u32, b as u32));
                }
            }
        }
        let mut jobs: Vec<(u32, u32)> = Vec::new();
        for (p, &(a, _)) in pairs.iter().enumerate() {
            for &k in &sorted[a as usize] {
                jobs.push((p as u32, k));
            }
        }
        let n = chunk_count;
        let reach_base = 3 * n as u32;
        let cp_base = 0u32;
        let job_base = cp_base + 2 * pairs.len() as u32;
        let count_base = job_base + 2 * jobs.len() as u32;
        let flag_base = count_base + 2 * jobs.len() as u32 + 1;
        let owned_base = flag_base + n as u32;
        let pair_base = owned_base + islands as u32;
        // Room for every pair a job could keep: its chunk against the whole other cluster.
        let max_pairs: usize = jobs.iter().map(|&(p, _)| sorted[pairs[p as usize].1 as usize].len()).sum();
        let mut pu = vec![0u32; pair_base as usize + 2 * max_pairs + 1];
        for (p, &(a, b)) in pairs.iter().enumerate() {
            pu[cp_base as usize + 2 * p] = a;
            pu[cp_base as usize + 2 * p + 1] = b;
        }
        for (j, &(p, k)) in jobs.iter().enumerate() {
            pu[job_base as usize + 2 * j] = p;
            pu[job_base as usize + 2 * j + 1] = k;
        }
        for i in 0..islands {
            pu[owned_base as usize + i] = owned[i] as u32;
        }
        let pm = gpu.storage("plan pm", &vec![[0f32; 4]; 3 * n + 2 * islands + 1]);
        let pu_buf = gpu.storage("plan pu", &pu);
        let g = (gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]).sqrt();
        let params = PlanParams {
            chunk_count: n as u32,
            island_count: islands as u32,
            job_count: jobs.len() as u32,
            impactor_count: impactor_count as u32,
            duration: duration as f32,
            g: g as f32,
            gz: gravity[2] as f32,
            ground_height: ground.unwrap_or(0.0) as f32,
            has_ground: ground.is_some() as u32,
            reach_base,
            cp_base,
            job_base,
            count_base,
            pair_base,
            flag_base,
            owned_base,
        };
        let pbuf = gpu.uniform("plan params", &params);
        let (chunks, islands_buf, state, impactors) = buffers;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, size) in [
            (&self.motion, groups(n, 64)),
            (&self.reach, islands.max(1) as u32),
            (&self.count, groups(jobs.len(), 64)),
            (&self.scan, 1),
            (&self.write, groups(jobs.len(), 64)),
            (&self.side, groups(n, 64)),
        ] {
            let bind = gpu.bind(&k.group, &[&pbuf, chunks, islands_buf, state, impactors, &pm, &pu_buf]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let u: Vec<u32> = gpu.read(&pu_buf);
        let m: Vec<[f32; 4]> = gpu.read_range(&pm, 0, 3 * n);
        let total = u[count_base as usize + jobs.len()] as usize;
        let flags = &u[flag_base as usize..flag_base as usize + n];
        GpuPlanOut {
            pairs: (0..total).map(|t| (u[pair_base as usize + 2 * t], u[pair_base as usize + 2 * t + 1])).collect(),
            ground: flags.iter().map(|f| f & 2 != 0).collect(),
            impactor: (0..impactor_count).map(|i| flags.iter().map(|f| f & (4 << i) != 0).collect()).collect(),
            rotation_checked: flags.iter().map(|f| f & 1 != 0).collect(),
            budget: (0..n).map(|k| m[3 * k + 1][3]).collect(),
        }
    }
}
