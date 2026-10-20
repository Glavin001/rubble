//! P0 step 4 of the learned impact-outcome plan: the GPU cost envelope of a random-weight
//! MeshGraphNet of the target shape (`stress_gpu::gnn_probe`). One JSON line per
//! configuration on stdout, a progress line per configuration on stderr. The browser runs
//! the same sweep through `stress-web`'s `gnn_probe_run`.
//!
//!   cargo run --release -p stress-gpu --bin stress_gpu_gnn_probe -- [--sweep size|batch|all]
//!       [--kernel fused|split|tiled|all] [--repeats 30] [--warmup 5]
//!   STRESS_GPU_SHADERS=wgsl   the WGSL path (Naga) instead of the Metal library

use stress_gpu::gnn_probe::{run, sweep, Precision, RoundKernel};
use stress_gpu::gpu::{Gpu, ShaderPath};

/// The default device with half precision (`SHADER_F16`) and timestamps where available.
fn gpu() -> Gpu {
    let mut g = Gpu::new().expect("GPU");
    let wanted = wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY | (g.adapter.features() & wgpu::Features::PASSTHROUGH_SHADERS);
    let features = g.adapter.features() & wanted;
    let (device, queue) = pollster::block_on(g.adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gnn probe"),
        required_features: features,
        required_limits: g.adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    g.device = device;
    g.queue = queue;
    g
}

fn main() {
    let mut name = "all".to_string();
    let (mut repeats, mut warmup) = (30usize, 5usize);
    let mut kernels = vec![RoundKernel::Fused, RoundKernel::Split, RoundKernel::Tiled];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sweep" => name = args.next().expect("--sweep size|batch|all"),
            "--repeats" => repeats = args.next().and_then(|v| v.parse().ok()).expect("--repeats N"),
            "--warmup" => warmup = args.next().and_then(|v| v.parse().ok()).expect("--warmup N"),
            "--kernel" => {
                kernels = match args.next().as_deref() {
                    Some("fused") => vec![RoundKernel::Fused],
                    Some("split") => vec![RoundKernel::Split],
                    Some("tiled") => vec![RoundKernel::Tiled],
                    Some("all") => vec![RoundKernel::Fused, RoundKernel::Split, RoundKernel::Tiled],
                    other => panic!("--kernel fused|split|tiled|all, not {other:?}"),
                }
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let gpu = gpu();
    let f16 = gpu.device.features().contains(wgpu::Features::SHADER_F16);
    let timestamps = gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let path = match gpu.shader_path {
        ShaderPath::MetalLib => "metallib",
        ShaderPath::MetalSource => "msl",
        ShaderPath::Wgsl => "wgsl",
    };
    eprintln!("[gnn] adapter {} ({:?}), shaders {path}, f16 {f16}, timestamps {timestamps}", gpu.adapter_info.name, gpu.adapter_info.backend);
    let precisions: Vec<Precision> = if f16 { vec![Precision::F32, Precision::F16] } else { vec![Precision::F32] };
    let configs = sweep(&name, &precisions, &kernels);
    let total = configs.len();
    let t0 = std::time::Instant::now();
    for (i, c) in configs.iter().enumerate() {
        let line = run(&gpu, c, warmup, repeats, path);
        println!("{line}");
        let med = |v: &serde_json::Value| v.get("median").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        eprintln!(
            "[gnn] {}/{total} | {} x {} chunks ({} nodes, {} edges) W {} {} {} {}: wall {:.3} ms, gpu {:.3} ms | {:.0} s",
            i + 1,
            c.islands,
            c.chunks,
            line["nodes"],
            line["edges"],
            c.width,
            c.precision.name(),
            c.schedule_name,
            c.kernel.name(),
            med(&line["wall_ms"]),
            med(&line["gpu_ms"]),
            t0.elapsed().as_secs_f64()
        );
    }
}
