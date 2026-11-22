//! What this machine's wgpu adapters offer for GPU timing (no GPU work, no lock needed):
//! `cargo run --release -p engine-webgpu --example timestamp_features`.
fn main() {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL;
    desc.flags = engine_webgpu::instance_flags();
    println!("instance flags (this build's validation and debug): {:?}", desc.flags);
    let instance = wgpu::Instance::new(desc);
    for a in pollster::block_on(instance.enumerate_adapters(desc_backends())) {
        let i = a.get_info();
        let f = a.features();
        println!("{} ({:?}, {:?})", i.name, i.backend, i.device_type);
        for (name, bit) in [
            ("TIMESTAMP_QUERY", wgpu::Features::TIMESTAMP_QUERY),
            ("TIMESTAMP_QUERY_INSIDE_ENCODERS", wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            ("TIMESTAMP_QUERY_INSIDE_PASSES", wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            ("PIPELINE_STATISTICS_QUERY", wgpu::Features::PIPELINE_STATISTICS_QUERY),
        ] {
            println!("  {name:34} {}", if f.contains(bit) { "yes" } else { "no" });
        }
    }
}

fn desc_backends() -> wgpu::Backends {
    wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL
}
