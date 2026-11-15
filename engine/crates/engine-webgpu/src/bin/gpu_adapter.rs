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
    // `--compile [--jobs N] NAME...`: N modules at once on N threads (default 1), to
    // see whether the driver compiles pipelines in parallel.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--warm") {
        warm(&device);
        return;
    }
    if args.first().map(String::as_str) == Some("--compile") {
        args.remove(0);
        // RUBBLE_COMPILE_JOBS sets it too (gpu_run.sh --variant).
        let mut jobs: usize = std::env::var("RUBBLE_COMPILE_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        if args.first().map(String::as_str) == Some("--jobs") {
            jobs = args[1].parse().expect("--jobs N");
            args.drain(..2);
        }
        let t = std::time::Instant::now();
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..jobs.max(1) {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(name) = args.get(i) else { break };
                    compile_timing(&device, name);
                });
            }
        });
        println!("compile: {} modules on {jobs} threads in {:.2} s", args.len(), t.elapsed().as_secs_f64());
    }
}

fn compile_timing(device: &wgpu::Device, spec: &str) {
    // `NAME@MODE` (measurement): wgpu's runtime checks off for that module, to see
    // what they cost the driver's compiler. MODE: `noloop` (no forced loop bounding),
    // `nobounds` (no injected bounds checks), `nodiv` (no integer-division checks),
    // `unchecked` (none of them). Plain NAME: the checks every host uses.
    let (name, mode) = spec.split_once('@').unwrap_or((spec, "checked"));
    let mut checks = wgpu::ShaderRuntimeChecks::checked();
    match mode {
        "checked" => {}
        "noloop" => checks.force_loop_bounding = false,
        "nobounds" => checks.bounds_checks = false,
        "nodiv" => checks.int_div_checks = false,
        "unchecked" => checks = wgpu::ShaderRuntimeChecks::unchecked(),
        m => panic!("unknown check mode {m}"),
    }
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
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
    };
    // SAFETY: measurement only; these modules are the engine's own kernels.
    let module = unsafe { device.create_shader_module_trusted(desc, checks) };
    let name = spec;
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

/// `--warm` (gpu_run.sh, before the selection): create, on every core at once, the
/// pipelines the explicit-beams host and the rigid World create one after another, so
/// the driver's shader disk cache holds them when the tests start. NVIDIA's Vulkan
/// driver compiles each pipeline on one core (the current explicit modules take minutes
/// each) but compiles different pipelines in parallel, so a cold kernel version costs
/// about its slowest module instead of the sum. The pipelines are
/// `ExplicitBeamsGpu::compute_specs` (the WGSL and layouts the host creates) for the
/// repository sedan with the shipped settings variants, alone, two and a group-sized
/// fleet, with and without the World coupling, plus the generated sedan when
/// `SEDAN` names its export; then the rigid World's kernels (`WgpuSim::prewarm`).
/// Kernels outside these (the bitwise tests' frozen and `_det` variants, other
/// layouts) still compile when first used.
fn warm(device: &wgpu::Device) {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
    use std::collections::BTreeMap;
    let t0 = std::time::Instant::now();
    let truck = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("the repo sedan");
    let settings = [
        ExplicitBeamsSettings::default(),
        ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() },
        ExplicitBeamsSettings { fast_inv_sqrt: true, ..Default::default() },
        ExplicitBeamsSettings { fast_inv_sqrt: true, drag: true, turbulence: true, ..Default::default() },
    ];
    let mut cars: Vec<ExplicitBeams> = settings
        .iter()
        .filter_map(|s| ExplicitBeams::from_truck(&truck, glam::Vec3::ZERO, glam::Vec3::ZERO, *s).ok())
        .collect();
    if let Some(car) = std::env::var("SEDAN").ok().and_then(|d| sedan_export(&d)) {
        cars.push(car);
    }
    let coupling = engine_webgpu::CouplingCapacity { bodies: 16, shapes: 64, planes: 1024 };
    let mut specs: BTreeMap<String, engine_webgpu::ComputeSpec> = BTreeMap::new();
    for car in &cars {
        for n in [1usize, 2, engine_exec::explicit_plan::VV_PART + 1] {
            let fleet: Vec<ExplicitBeams> = (0..n)
                .map(|i| {
                    let mut c = car.clone();
                    for node in &mut c.nodes {
                        node.pos.x += 6.0 * i as f32;
                    }
                    c
                })
                .collect();
            for cap in [None, Some(coupling)] {
                for spec in engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, cap) {
                    let key = format!("{}|{}|{:?}|{}", spec.module, spec.entry, spec.layout, spec.wgsl.len());
                    specs.entry(key).or_insert(spec);
                }
            }
        }
    }
    let mut todo: Vec<engine_webgpu::ComputeSpec> = specs.into_values().collect();
    todo.sort_by(|a, b| b.wgsl.len().cmp(&a.wgsl.len()));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    println!("warm: {} explicit pipelines on {threads} threads", todo.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(todo.len()) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(spec) = todo.get(i) else { break };
                let t = std::time::Instant::now();
                build_spec(device, spec);
                let secs = t.elapsed().as_secs_f64();
                if secs > 1.0 {
                    println!("  warm {} ({}): {secs:.1} s", spec.module, spec.entry);
                }
            });
        }
    });
    println!("warm: explicit pipelines in {:.1} s", t0.elapsed().as_secs_f64());
    let t = std::time::Instant::now();
    match engine_webgpu::WgpuSim::new(engine_schema::WorldCapacity::default()) {
        Ok(sim) => {
            let p = sim.prewarm_async(engine_webgpu::KernelGroups::ALL);
            while !p.is_done() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            println!("warm: {} rigid World kernels in {:.1} s", p.progress().1, t.elapsed().as_secs_f64());
        }
        Err(e) => println!("warm: no rigid World ({e})"),
    }
    println!("warm: done in {:.1} s", t0.elapsed().as_secs_f64());
}

fn build_spec(device: &wgpu::Device, spec: &engine_webgpu::ComputeSpec) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(spec.label),
        source: wgpu::ShaderSource::Wgsl(spec.wgsl.as_str().into()),
    });
    let layout = spec.layout.as_ref().map(|entries| {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(spec.label),
            entries,
        });
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(spec.label),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        })
    });
    let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(spec.label),
        layout: layout.as_ref(),
        module: &module,
        entry_point: Some(spec.entry),
        compilation_options: Default::default(),
        cache: None,
    });
}

/// The generated sedan from its JBeam export (`files.json`, `*.jbeam`, `*.pc`), with
/// the vehicle lab's settings.
fn sedan_export(dir: &str) -> Option<engine_softbody::explicit_beams::ExplicitBeams> {
    use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
    let dir = std::path::Path::new(dir);
    let list: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("files.json")).ok()?).ok()?;
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((f.clone(), std::fs::read_to_string(dir.join(f)).ok()?));
        } else if f.ends_with(".pc") {
            pc = std::fs::read_to_string(dir.join(f)).ok();
        }
    }
    let refs: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).ok()?;
    let settings = ExplicitBeamsSettings { fast_inv_sqrt: true, drag: true, ..Default::default() };
    ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, glam::Vec3::ZERO, glam::Vec3::ZERO, settings)
        .ok()
        .map(|(sim, _)| sim)
}
