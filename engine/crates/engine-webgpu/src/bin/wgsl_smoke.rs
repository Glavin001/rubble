//! Report Slang→SPIR-V→WGSL translation coverage for the fair Chrome WebGPU bar.

use engine_webgpu::WebGpuWorld;

fn main() {
    env_logger::init();
    let report = WebGpuWorld::compile_shaders_report();
    let mut ok = 0u32;
    let mut bad = 0u32;
    for (name, r) in &report {
        match r {
            Ok(wgsl) => {
                ok += 1;
                println!("OK   {name:28} {} bytes", wgsl.len());
            }
            Err(e) => {
                bad += 1;
                let short = e.lines().next().unwrap_or(e.as_str());
                println!("FAIL {name:28} {short}");
            }
        }
    }
    println!("\nSPIR-V→WGSL: {ok} ok, {bad} failed / {}", report.len());
    match WebGpuWorld::smoke_device() {
        Ok(s) => println!("wgpu device: {s}"),
        Err(e) => println!("wgpu device: ERR {e:#}"),
    }
    if bad > 0 {
        std::process::exit(1);
    }
}
