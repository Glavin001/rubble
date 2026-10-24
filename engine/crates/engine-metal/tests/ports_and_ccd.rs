use engine_webgpu::WebGpuWorld;

#[test]
fn webgpu_spirv_to_wgsl() {
    // Full graph must translate for the fair Chrome bar; report all failures.
    let report = WebGpuWorld::compile_shaders_report();
    assert!(report.len() >= 30, "expected full world_step SPIR-V set, got {}", report.len());
    let mut fails = Vec::new();
    for (name, r) in &report {
        match r {
            Ok(wgsl) => {
                assert!(
                    wgsl.contains("fn ") || wgsl.contains("@compute"),
                    "{name} empty/invalid"
                );
            }
            Err(e) => fails.push(format!("{name}: {e}")),
        }
    }
    assert!(
        fails.is_empty(),
        "SPIR-V→WGSL failed for {} entries:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
