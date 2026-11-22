//! Native pipeline compile times of the contact modules and the split group path's
//! (contact-groups-design.md): wall time of `create_compute_pipeline` for each, with
//! workgroup zeroing on and off. No dispatch. Local measurement tool.
//!
//!   cargo run --release -p engine-webgpu-lab --example contact_compile_time
fn main() {
    let (_, device, _queue) = engine_webgpu::default_device().unwrap();
    let all = engine_webgpu::explicit_beams_shipped_wgsl();
    // Module names from the arguments (default: the contact modules); each compiled with
    // zeroing on, off, and off again (a second compile in the process: the driver's cache).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default = [
        "explicit_beams_contact_wide_nosurf",
        "explicit_beams_solo_wide_nosurf",
        "explicit_beams_split_member_wide",
        "explicit_beams_split_group_wide",
        "explicit_beams_split_tests_wide",
    ];
    let names: Vec<&str> = if args.is_empty() { default.to_vec() } else { args.iter().map(String::as_str).collect() };
    for name in names {
        let (_, entry, wgsl) = all.iter().find(|(n, _, _)| n == name).expect("module");
        for zero in [true, false, false] {
            let t = std::time::Instant::now();
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(wgsl.as_str().into()),
            });
            let _p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: zero,
                    ..Default::default()
                },
                cache: None,
            });
            println!("{name} ({} KB WGSL), zeroing {zero}: {:.1} s", wgsl.len() / 1024, t.elapsed().as_secs_f64());
        }
    }
}
