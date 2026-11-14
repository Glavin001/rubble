//! Measurement: the cold compile time of every shipped explicit-beams module on this
//! machine's native wgpu backend (Metal on macOS): shader module and compute pipeline
//! (layout auto), one at a time. Each module's entry point is renamed with a nonce
//! first, so the driver's shader cache (keyed by source) cannot serve it: the time is
//! a user's first load. Diagnostic only.
//!
//!   cargo run --release -p engine-webgpu --example explicit_compile_time [name filter...]
//!
//! (bounded: perl -e 'alarm 1200; exec @ARGV'; one module at a time.)
fn main() -> anyhow::Result<()> {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let (info, device, _queue) = engine_webgpu::default_device()?;
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
    println!("{} ({:?}), nonce {nonce:x}", info.name, info.backend);
    let mut total = 0.0;
    for (name, entry, wgsl) in engine_webgpu::explicit_beams_shipped_wgsl() {
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let renamed = format!("{entry}_n{nonce:x}");
        let src = wgsl.replace(&format!("fn {entry}("), &format!("fn {renamed}("));
        let t = std::time::Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let _pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&name),
            layout: None,
            module: &module,
            entry_point: Some(&renamed),
            compilation_options: Default::default(),
            cache: None,
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let s = t.elapsed().as_secs_f64();
        total += s;
        println!("{name:40} {:>7} KB  {s:8.2} s", wgsl.len() / 1024);
    }
    println!("total {total:.2} s");
    Ok(())
}
