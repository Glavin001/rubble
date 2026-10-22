//! P3: the executor of P2's impact-outcome model on the GPU. The program (the op list, the
//! packed weights, the gather and CSR table, the initial arena) is built by
//! `stress_learned::outcome_program::compile`; this module only runs it with the Slang
//! kernels of `shaders/outcome_gnn.slang`, every op in one compute pass (WebGPU orders the
//! dispatches, each seeing the last one's writes). Op `k`'s 64 parameter words select the
//! kernel (`params[0]`: 0 MLP, 1 CSR, 2 fibre, 3 tiled MLP) and are its uniform.

use crate::gpu::{Binding, Gpu, Kernel};

pub const PARAM_WORDS: usize = 64;

/// The model's three kernels on a device.
pub struct OutcomeKernels {
    kernels: [Kernel; 4],
}

impl OutcomeKernels {
    pub fn new(gpu: &Gpu) -> OutcomeKernels {
        let set = &crate::shaders::OUTCOME_GNN;
        let layout = [Binding::Uniform, Binding::Storage, Binding::Storage, Binding::StorageRw];
        OutcomeKernels { kernels: [gpu.compute(set, "gnn_mlp", &layout), gpu.compute(set, "gnn_csr", &layout), gpu.compute(set, "gnn_fibre", &layout), gpu.compute(set, "gnn_mlp_tiled", &layout)] }
    }
}

/// One uploaded program: its buffers and a bind group per op.
pub struct OutcomeRun {
    timestamps: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
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
            assert!(kernel < 4, "op kernel {kernel}");
            let u = gpu.uniform("outcome op", p);
            let bind = gpu.bind(&k.kernels[kernel].group, &[&u, &w, &t, &a]);
            uniforms.push(u);
            steps.push((kernel, bind, *gx, *gy));
        }
        let timestamps = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let q = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("outcome time"), ty: wgpu::QueryType::Timestamp, count: 2 });
            let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("outcome resolve"), size: 16, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("outcome read"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            (q, resolve, read)
        });
        OutcomeRun { timestamps, arena: a, _weights: w, _table: t, _uniforms: uniforms, steps, arena_len: arena.len() }
    }

    /// Record the whole forward pass.
    pub fn encode(&self, k: &OutcomeKernels, encoder: &mut wgpu::CommandEncoder) {
        let tw = self.timestamps.as_ref().map(|(q, _, _)| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("outcome gnn"), timestamp_writes: tw });
            for (kernel, bind, gx, gy) in &self.steps {
                pass.set_pipeline(&k.kernels[*kernel].pipeline);
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        if let Some((q, resolve, read)) = &self.timestamps {
            encoder.resolve_query_set(q, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 16);
        }
    }

    /// The last run's GPU time (ms) from its timestamps, after the device is idle.
    fn gpu_ms(&self, gpu: &Gpu) -> Option<f64> {
        let (_, _, read) = self.timestamps.as_ref()?;
        let slice = read.slice(..);
        let mapped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |_| flag.store(true, std::sync::atomic::Ordering::Release));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        engine_gpu_wait::until(|| mapped.load(std::sync::atomic::Ordering::Acquire));
        let t: [u64; 2] = {
            let data = slice.get_mapped_range().expect("mapped timestamps");
            let v: &[u64] = bytemuck::cast_slice(&data);
            [v[0], v[1]]
        };
        read.unmap();
        Some(t[1].saturating_sub(t[0]) as f64 * gpu.queue.get_timestamp_period() as f64 * 1e-6)
    }

    /// The forward pass `repeats` times (after `warmup`), each submitted alone and waited
    /// for: wall times (ms, submit to the work-done callback) and GPU times (ms, the pass's
    /// timestamps, when the device has them). In the browser, under `WebAssembly.promising`.
    pub fn time(&self, gpu: &Gpu, k: &OutcomeKernels, warmup: usize, repeats: usize) -> (Vec<f64>, Vec<f64>) {
        let (mut wall, mut gpu_t) = (Vec::new(), Vec::new());
        for r in 0..warmup + repeats {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            self.encode(k, &mut enc);
            let t = web_time::Instant::now();
            gpu.queue.submit([enc.finish()]);
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = done.clone();
            gpu.queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            engine_gpu_wait::until(|| done.load(std::sync::atomic::Ordering::Acquire));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let g = self.gpu_ms(gpu);
            if r >= warmup {
                wall.push(ms);
                if let Some(g) = g {
                    gpu_t.push(g);
                }
            }
        }
        (wall, gpu_t)
    }

    /// Steady state: `n` forward passes in one submission (as consecutive frames would
    /// keep the GPU busy), the wall time per pass (ms), `samples` times after a warm-up.
    pub fn time_steady(&self, gpu: &Gpu, k: &OutcomeKernels, n: usize, samples: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for r in 0..samples + 1 {
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            for _ in 0..n {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("outcome gnn steady"), timestamp_writes: None });
                for (kernel, bind, gx, gy) in &self.steps {
                    pass.set_pipeline(&k.kernels[*kernel].pipeline);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(*gx, *gy, 1);
                }
            }
            let t = web_time::Instant::now();
            gpu.queue.submit([enc.finish()]);
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = done.clone();
            gpu.queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            engine_gpu_wait::until(|| done.load(std::sync::atomic::Ordering::Acquire));
            if r > 0 {
                out.push(t.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
        }
        out
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

/// A program as one file (`OGNN1`, little-endian), so a page can time the same program the
/// native bench compiled: magic, the counts (arena, weights, table, ops) as u32, the arena,
/// weights and table, then per op its 64 parameter words and its (x, y) grid.
pub fn encode_program(arena: &[f32], weights: &[f32], table: &[u32], ops: &[([u32; PARAM_WORDS], (u32, u32))]) -> Vec<u8> {
    let mut b = b"OGNN1\0\0\0".to_vec();
    for n in [arena.len(), weights.len(), table.len(), ops.len()] {
        b.extend_from_slice(&(n as u32).to_le_bytes());
    }
    b.extend_from_slice(bytemuck::cast_slice(arena));
    b.extend_from_slice(bytemuck::cast_slice(weights));
    b.extend_from_slice(bytemuck::cast_slice(table));
    for (p, (gx, gy)) in ops {
        b.extend_from_slice(bytemuck::cast_slice(p));
        b.extend_from_slice(&gx.to_le_bytes());
        b.extend_from_slice(&gy.to_le_bytes());
    }
    b
}

/// The inverse of `encode_program`.
#[allow(clippy::type_complexity)]
pub fn decode_program(b: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>, Vec<([u32; PARAM_WORDS], (u32, u32))>), String> {
    if b.len() < 24 || &b[..5] != b"OGNN1" {
        return Err("not an OGNN1 program".into());
    }
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let (na, nw, nt, no) = (u(8) as usize, u(12) as usize, u(16) as usize, u(20) as usize);
    let need = 24 + 4 * (na + nw + nt) + no * 4 * (PARAM_WORDS + 2);
    if b.len() != need {
        return Err(format!("program of {} bytes, expected {need}", b.len()));
    }
    let words: Vec<u32> = b[24..].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let arena = words[..na].iter().map(|&x| f32::from_bits(x)).collect();
    let weights = words[na..na + nw].iter().map(|&x| f32::from_bits(x)).collect();
    let table = words[na + nw..na + nw + nt].to_vec();
    let ops = words[na + nw + nt..]
        .chunks_exact(PARAM_WORDS + 2)
        .map(|c| {
            let mut p = [0u32; PARAM_WORDS];
            p.copy_from_slice(&c[..PARAM_WORDS]);
            (p, (c[PARAM_WORDS], c[PARAM_WORDS + 1]))
        })
        .collect();
    Ok((arena, weights, table, ops))
}
