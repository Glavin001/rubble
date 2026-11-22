//! Measurement: the cold compile time of every shipped explicit-beams module on this
//! machine's native wgpu backend (Metal on macOS): shader module and compute pipeline
//! (layout auto), one at a time. Each module's entry point is renamed with a nonce
//! first, so the driver's shader cache (keyed by source) cannot serve it: the time is
//! a user's first load. Diagnostic only.
//!
//!   cargo run --release -p engine-webgpu-lab --example explicit_compile_time [name filter...]
//!
//! (bounded: perl -e 'alarm 1200; exec @ARGV'; one module at a time.)
//!
//! EXPLICIT_WGSL_DIR=DIR: the modules of DIR's manifest.json instead (for example the
//! fast setting's, from `explicit_wgsl_dump -- --fast`), as the pages create them.
fn main() -> anyhow::Result<()> {
    let filters: Vec<String> = std::env::args().skip(1).collect();
    let (info, device, _queue) = engine_webgpu::default_device()?;
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u64;
    println!("{} ({:?}), nonce {nonce:x}", info.name, info.backend);
    let mut total = 0.0;
    let modules: Vec<(String, String, String)> = match std::env::var("EXPLICIT_WGSL_DIR") {
        Ok(dir) => {
            let dir = std::path::PathBuf::from(dir);
            let list: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
            let mut v = Vec::new();
            for m in list.as_array().into_iter().flatten() {
                let name = m["name"].as_str().unwrap_or_default().to_string();
                let entry = m["entry"].as_str().unwrap_or_default().to_string();
                let wgsl = std::fs::read_to_string(dir.join(format!("{name}.wgsl")))?;
                v.push((name, entry, wgsl));
            }
            v
        }
        Err(_) => engine_webgpu::explicit_beams_shipped_wgsl()
            .into_iter()
            .map(|(n, e, w)| (n.to_string(), e.to_string(), w.to_string()))
            .collect(),
    };
    for (name, entry, wgsl) in modules {
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
