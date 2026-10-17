//! Where a substep's time goes on this GPU.
//!
//! 1. The island kernel's time per substep against bonds per thread (one island, one
//!    threadgroup of 256): the floor, and the cost of each further joint-law
//!    evaluation a thread does in sequence.
//! 2. Primitive latencies of the device, from one 256-thread workgroup running long
//!    dependent loops (dispatch cost amortised away): a dependent FMA, a dependent
//!    sqrt + divide, a threadgroup barrier with a shared-memory exchange, and a
//!    device-memory barrier with a device-memory exchange (what `AllMemoryBarrier`
//!    compiles to).
//!
//!   stress-gpu-latency

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::scenes;
use stress_gpu::solver::GpuSolver;
use stress_ref::scene::SolveMode;
use stress_ref::solver::ReferenceSolver;

const MICRO: &str = r#"
struct P { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn empty(@builtin(local_invocation_index) t: u32) {
    var x = buf[t];
    for (var i = 0u; i < p.n; i++) { x = x + 1.0; }
    buf[t] = x;
}

@compute @workgroup_size(256)
fn fma_chain(@builtin(local_invocation_index) t: u32) {
    var x = buf[t];
    for (var i = 0u; i < p.n; i++) {
        x = fma(x, 1.0000001, 1e-7); x = fma(x, 0.9999999, 1e-7);
        x = fma(x, 1.0000001, 1e-7); x = fma(x, 0.9999999, 1e-7);
        x = fma(x, 1.0000001, 1e-7); x = fma(x, 0.9999999, 1e-7);
        x = fma(x, 1.0000001, 1e-7); x = fma(x, 0.9999999, 1e-7);
    }
    buf[t] = x;
}

@compute @workgroup_size(256)
fn sqrt_div_chain(@builtin(local_invocation_index) t: u32) {
    var x = buf[t] + 1.0;
    for (var i = 0u; i < p.n; i++) {
        x = sqrt(x + 1.0) / (x + 0.5); x = sqrt(x + 1.0) / (x + 0.5);
        x = sqrt(x + 1.0) / (x + 0.5); x = sqrt(x + 1.0) / (x + 0.5);
    }
    buf[t] = x;
}

@compute @workgroup_size(256)
fn group_barrier(@builtin(local_invocation_index) t: u32) {
    var x = buf[t];
    for (var i = 0u; i < p.n; i++) {
        sh[t] = x;
        workgroupBarrier();
        x = sh[(t + 1u) & 255u] * 0.5 + 0.25;
        workgroupBarrier();
    }
    buf[t] = x;
}

@compute @workgroup_size(256)
fn device_barrier(@builtin(local_invocation_index) t: u32) {
    var x = buf[t];
    for (var i = 0u; i < p.n; i++) {
        buf[256u + t] = x;
        storageBarrier();
        x = buf[256u + ((t + 1u) & 255u)] * 0.5 + 0.25;
        storageBarrier();
    }
    buf[t] = x;
}
"#;

fn main() {
    let gpu = Gpu::new().expect("GPU");
    println!("{} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);

    println!("\n## Island kernel: one island, one threadgroup of 256 threads, 2000 substeps");
    println!("| slab | chunks | bonds | bonds/thread | us/substep | batches | gpu wait us/substep | readback us/substep |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for n in [[2, 2, 1], [4, 2, 2], [8, 6, 2], [12, 6, 2], [16, 8, 2], [16, 12, 2], [20, 16, 2], [32, 16, 2], [40, 20, 4]] {
        let mut scene = scenes::slab(n);
        scene.sim.solve_mode = SolveMode::Explicit;
        let reference = ReferenceSolver::new(&scene);
        let dt = reference.stable_dt();
        let bonds: usize = reference.clusters.iter().map(|c| c.bonds.len()).sum();
        let chunks: usize = reference.clusters.iter().map(|c| c.chunks.len()).sum();
        let mut solver = GpuSolver::new(&gpu, reference, None, Vec::new()).expect("GPU solver");
        let substeps = 2000;
        solver.step(&gpu, dt, substeps).expect("warm-up");
        // The GPU is shared with other work: the minimum over repeats is the
        // uncontended cost (`STRESS_GPU_REPEAT`, default 5).
        let repeats: usize = std::env::var("STRESS_GPU_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let before = solver.profile.clone();
        let mut best = f64::INFINITY;
        for _ in 0..repeats {
            let t = Instant::now();
            solver.step(&gpu, dt, substeps).expect("step");
            best = best.min(t.elapsed().as_secs_f64());
        }
        let us = best / substeps as f64 * 1e6;
        let substeps = substeps * repeats;
        let p = &solver.profile;
        let per = |x: f64| x / substeps as f64 * 1e6;
        println!(
            "| {}x{}x{} | {chunks} | {bonds} | {:.2} | {us:.1} | {} | {:.1} | {:.1} |",
            n[0],
            n[1],
            n[2],
            bonds as f64 / 256.0,
            p.batches - before.batches,
            per(p.gpu - before.gpu),
            per(p.readback - before.readback)
        );
    }

    println!("\n## Primitive latencies (one workgroup of 256, dependent loop)");
    let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("latency"), source: wgpu::ShaderSource::Wgsl(MICRO.into()) });
    let bindings = [stress_gpu::gpu::Binding::StorageRw, stress_gpu::gpu::Binding::Uniform];
    let (group, layout) = gpu.layout("latency", &bindings, wgpu::ShaderStages::COMPUTE);
    let buf = gpu.storage("latency buf", &vec![1.0f32; 512]);
    let n: u32 = 200_000;
    let uni = gpu.uniform("latency params", &[n, 0, 0, 0]);
    let bind = gpu.bind(&group, &[&buf, &uni]);
    let time = |entry: &str| -> f64 {
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let run = || {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        run();
        let t = Instant::now();
        run();
        t.elapsed().as_secs_f64()
    };
    let base = time("empty");
    println!("| primitive | ns each |");
    println!("|---|---:|");
    println!("| loop iteration (baseline) | {:.2} |", base / n as f64 * 1e9);
    for (entry, per_iter, what) in [
        ("fma_chain", 8.0, "dependent FMA"),
        ("sqrt_div_chain", 4.0, "dependent sqrt + divide"),
        ("group_barrier", 2.0, "threadgroup barrier + shared exchange"),
        ("device_barrier", 2.0, "device barrier + device exchange"),
    ] {
        let secs = time(entry) - base;
        println!("| {what} | {:.2} |", secs / n as f64 / per_iter * 1e9);
    }
}
