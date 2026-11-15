//! The shared explicit-beams plan's module table (`engine_exec::explicit_plan::MODULES`,
//! which the Metal and CUDA builds compile) names exactly the WGSL modules this
//! crate ships (`explicit_beams_shipped_wgsl`, which Chrome's check compiles), with
//! each module's entry point: one list for every target.
use engine_exec::explicit_plan::MODULES;

#[test]
fn plan_modules_are_the_shipped_wgsl() {
    let mut shipped: Vec<(String, &str)> = engine_webgpu::explicit_beams_shipped_wgsl()
        .into_iter()
        .map(|(n, e, _)| (n, e))
        .collect();
    let mut plan: Vec<(String, &str)> = MODULES.iter().map(|m| (m.name.to_string(), m.entry)).collect();
    shipped.sort();
    plan.sort();
    assert_eq!(plan, shipped);
}
