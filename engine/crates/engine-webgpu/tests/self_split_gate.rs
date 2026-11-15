//! Self-collision split waits for its kernels' async compiles in a browser host
//! (`engine_webgpu::set_pipeline_gate`, `mark_pipeline_compiled`): with the gate on and
//! its modules not yet marked, frames keep the tests in the workgroup (the same
//! results) and never create the split's pipelines; once every module the frame needs
//! is marked, the split runs. The modules are those `ExplicitBeamsGpu::compute_specs`
//! lists as deferred. Its own test binary: the gate is process-wide.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(at: Vec3) -> ExplicitBeams {
    use engine_softbody::authoring::{build, style, BuildParams, Resolution};
    let text = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Low, ..Default::default() })
        .unwrap()
        .truck;
    let mut s = ExplicitBeams::from_truck_config(&text, None, at, Vec3::ZERO, ExplicitBeamsSettings::default()).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.finish();
    s
}

#[test]
fn the_split_waits_for_its_deferred_compiles() {
    let Ok((_, device, queue)) = default_device() else { return };
    engine_webgpu::set_pipeline_gate(true);
    for fleet in [vec![sedan(Vec3::ZERO)], vec![sedan(Vec3::ZERO), sedan(Vec3::new(30.0, 0.0, 0.0))]] {
        let specs = ExplicitBeamsGpu::compute_specs(&fleet, None);
        let deferred: Vec<&str> = specs.iter().filter(|s| s.deferred && s.module.contains("selfsplit")).map(|s| s.module).collect();
        assert!(!deferred.is_empty(), "{} vehicles: the self split's modules are listed, deferred", fleet.len());
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
        gpu.set_wide(false);
        gpu.set_vehicle_contact(fleet.len() > 1);
        gpu.set_self_split(true);
        gpu.frame(4);
        gpu.wait();
        assert!(!gpu.self_split_used(), "{} vehicles: the split ran before its compiles were marked", fleet.len());
        for s in specs.iter().filter(|s| s.deferred) {
            engine_webgpu::mark_pipeline_compiled(s.module);
        }
        gpu.frame(4);
        gpu.wait();
        assert!(gpu.self_split_used(), "{} vehicles: the split did not run once its modules were marked", fleet.len());
    }
    engine_webgpu::set_pipeline_gate(false);
}
