//! The shared explicit-beams plan's module table (`engine_exec::explicit_plan::MODULES`,
//! which the Metal and CUDA builds compile) names exactly the WGSL modules this
//! crate ships (`explicit_beams_shipped_wgsl`, which Chrome's check compiles), with
//! each module's entry point: one list for every target. The WebGPU-only tables are
//! left out: the modules without the World's coupling
//! (`engine_webgpu::EXPLICIT_BEAMS_NOCPL`, twins a browser compiles faster) and
//! surface contact split from the frame (`engine_webgpu::EXPLICIT_BEAMS_SURF_SPLIT`:
//! the `*_seg` frames and the `explicit_surface_*` pipelines), which Metal and CUDA
//! do not run (known-bugs P1).
use engine_exec::explicit_plan::MODULES;

#[test]
fn plan_modules_are_the_shipped_wgsl() {
    let webgpu_only: Vec<&str> = engine_webgpu::EXPLICIT_BEAMS_NOCPL
        .iter()
        .chain(engine_webgpu::EXPLICIT_BEAMS_SURF_SPLIT.iter())
        .map(|(n, _)| *n)
        .collect();
    let mut shipped: Vec<(String, &str)> = engine_webgpu::explicit_beams_shipped_wgsl()
        .into_iter()
        .map(|(n, e, _)| (n, e))
        .filter(|(n, _)| !webgpu_only.contains(&n.as_str()))
        .collect();
    let mut plan: Vec<(String, &str)> = MODULES
        .iter()
        .map(|m| (m.name.to_string(), m.entry))
        .collect();
    shipped.sort();
    plan.sort();
    assert_eq!(plan, shipped);
    // The WebGPU-only modules are shipped, and none is in the shared table.
    let all = engine_webgpu::explicit_beams_shipped_wgsl();
    for n in &webgpu_only {
        assert!(all.iter().any(|(s, _, _)| s == n), "{n} is not shipped");
        assert!(
            MODULES.iter().all(|m| m.name != *n),
            "{n} is in the shared table"
        );
    }
}
