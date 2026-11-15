//! Native pipeline compile times of the contact modules and the split group path's
//! (contact-groups-design.md): wall time of `create_compute_pipeline` for each, with
//! workgroup zeroing on and off. No dispatch. Local measurement tool.
//!
//!   cargo run --release -p engine-webgpu --example contact_compile_time
fn main() {
    let (_, device, _queue) = engine_webgpu::default_device().unwrap();
    let all = engine_webgpu::explicit_beams_shipped_wgsl();
    for name in [
        "explicit_beams_contact_wide_nosurf",
        "explicit_beams_solo_wide_nosurf",
        "explicit_beams_split_member_wide",
        "explicit_beams_split_group_wide",
        "explicit_beams_split_tests_wide",
    ] {
        let (_, entry, wgsl) = all.iter().find(|(n, _, _)| n == name).expect("module");
        for zero in [true, false] {
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
