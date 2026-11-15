//! The adapter `default_device` picks (the one every wgpu test and example uses), and
//! whether it is real hardware: `engine/tools/modal/gpu_run.sh` runs this first on the
//! remote GPU and stops when it exits non-zero.
//!
//! Exit codes: 0 a hardware adapter; 2 no adapter; 3 a software rasteriser (llvmpipe,
//! lavapipe, SwiftShader or any CPU device type).

fn main() {
    let (info, device, _queue) = match engine_webgpu::default_device() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("gpu_adapter: {e}");
            std::process::exit(2);
        }
    };
    let limits = device.limits();
    println!(
        "adapter: {} | backend {:?} | type {:?} | driver {} {} | vendor {:#06x} device {:#06x}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info, info.vendor, info.device
    );
    println!(
        "limits: storage buffers/stage {}, max storage binding {} MB, workgroup storage {} B, invocations/workgroup {}",
        limits.max_storage_buffers_per_shader_stage,
        u64::from(limits.max_storage_buffer_binding_size) >> 20,
        limits.max_compute_workgroup_storage_size,
        limits.max_compute_invocations_per_workgroup
    );
    let name = info.name.to_lowercase();
    let software = info.device_type == wgpu::DeviceType::Cpu
        || ["llvmpipe", "lavapipe", "swiftshader", "softpipe"]
            .iter()
            .any(|s| name.contains(s) || info.driver.to_lowercase().contains(s));
    if software {
        eprintln!("gpu_adapter: {} is a software adapter, not the GPU", info.name);
        std::process::exit(3);
    }
    // `--compile NAME [NAME...]` (diagnostics): the time to create the shader module of
    // each generated `NAME.wgsl` (EXPLICIT_BEAMS_WGSL_DIR) and a pipeline of each of
    // its compute entry points: where a cold start's CPU time goes on this driver.
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--compile") {
        for name in &args[1..] {
            compile_timing(&device, name);
        }
    }
}

fn compile_timing(device: &wgpu::Device, name: &str) {
    let path = format!("{}/{name}.wgsl", engine_webgpu::EXPLICIT_BEAMS_WGSL_DIR);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let entries: Vec<String> = src
        .split("@compute")
        .skip(1)
        .filter_map(|s| {
            let at = s.find("fn ")? + 3;
            Some(s[at..].split('(').next()?.trim().to_string())
        })
        .collect();
    let t = std::time::Instant::now();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
    });
    println!(
        "compile {name}: {} kB WGSL, module {:.2} s, {} entry points",
        src.len() >> 10,
        t.elapsed().as_secs_f64(),
        entries.len()
    );
    let all = std::time::Instant::now();
    for e in &entries {
        let t = std::time::Instant::now();
        let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(e),
            layout: None,
            module: &module,
            entry_point: Some(e),
            compilation_options: Default::default(),
            cache: None,
        });
        println!("  pipeline {e}: {:.2} s", t.elapsed().as_secs_f64());
    }
    println!("compile {name}: pipelines {:.2} s", all.elapsed().as_secs_f64());
}
