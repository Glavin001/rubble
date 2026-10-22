//! P3: the executor of P2's impact-outcome model on the GPU. The program (the op list, the
//! packed weights, the gather and CSR table, the initial arena) is built by
//! `stress_learned::outcome_program::compile`; this module only runs it with the Slang
//! kernels of `shaders/outcome_gnn.slang`, every op in one compute pass (WebGPU orders the
//! dispatches, each seeing the last one's writes). Op `k`'s 64 parameter words select the
//! kernel (`params[0]`: 0 MLP, 1 CSR, 2 fibre) and are its uniform.

use crate::gpu::{Binding, Gpu, Kernel};

pub const PARAM_WORDS: usize = 64;

/// The model's three kernels on a device.
pub struct OutcomeKernels {
    kernels: [Kernel; 3],
}

impl OutcomeKernels {
    pub fn new(gpu: &Gpu) -> OutcomeKernels {
        let set = &crate::shaders::OUTCOME_GNN;
        let layout = [Binding::Uniform, Binding::Storage, Binding::Storage, Binding::StorageRw];
        OutcomeKernels { kernels: [gpu.compute(set, "gnn_mlp", &layout), gpu.compute(set, "gnn_csr", &layout), gpu.compute(set, "gnn_fibre", &layout)] }
    }
}

/// One uploaded program: its buffers and a bind group per op.
pub struct OutcomeRun {
    arena: wgpu::Buffer,
    _weights: wgpu::Buffer,
    _table: wgpu::Buffer,
    _uniforms: Vec<wgpu::Buffer>,
    steps: Vec<(usize, wgpu::BindGroup, u32, u32)>,
    arena_len: usize,
}

impl OutcomeRun {
    /// Upload a program. `ops`: each op's parameter words and its workgroup grid.
    pub fn new(gpu: &Gpu, k: &OutcomeKernels, arena: &[f32], weights: &[f32], table: &[u32], ops: &[([u32; PARAM_WORDS], (u32, u32))]) -> OutcomeRun {
        let a = gpu.storage("outcome arena", arena);
        let w = gpu.storage("outcome weights", weights);
        let t = gpu.storage("outcome table", table);
        let mut uniforms = Vec::with_capacity(ops.len());
        let mut steps = Vec::with_capacity(ops.len());
        for (p, (gx, gy)) in ops {
            let kernel = p[0] as usize;
            assert!(kernel < 3, "op kernel {kernel}");
            let u = gpu.uniform("outcome op", p);
            let bind = gpu.bind(&k.kernels[kernel].group, &[&u, &w, &t, &a]);
            uniforms.push(u);
            steps.push((kernel, bind, *gx, *gy));
        }
        OutcomeRun { arena: a, _weights: w, _table: t, _uniforms: uniforms, steps, arena_len: arena.len() }
    }

    /// Record the whole forward pass.
    pub fn encode(&self, k: &OutcomeKernels, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("outcome gnn"), timestamp_writes: None });
        for (kernel, bind, gx, gy) in &self.steps {
            pass.set_pipeline(&k.kernels[*kernel].pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(*gx, *gy, 1);
        }
    }

    /// Run once and read the final arena back (blocking).
    pub fn run(&self, gpu: &Gpu, k: &OutcomeKernels) -> Vec<f32> {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        self.encode(k, &mut enc);
        gpu.queue.submit([enc.finish()]);
        gpu.read_range(&self.arena, 0, self.arena_len)
    }

    pub fn dispatches(&self) -> usize {
        self.steps.len()
    }
}
