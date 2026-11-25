//! Edge contact while a browser host's kernels compile (`set_pipeline_gate`). Its own test
//! binary: the gate and the marks are process-wide.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(at: Vec3) -> ExplicitBeams {
    use engine_softbody_authoring::authoring::{build, style, BuildParams, Resolution};
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

/// Edge contact in a browser host: whatever order its modules' compiles end in, the frames
/// run (none panics), and with all of them marked the edge pass acts.
#[test]
fn edge_contact_waits_for_its_deferred_compiles() {
    let Ok((_, device, queue)) = default_device() else { return };
    let fleet = || {
        let mut f = vec![sedan(Vec3::ZERO), sedan(Vec3::new(1.0, 0.0, 0.0))];
        for s in &mut f {
            (s.two_body_contact, s.dissipative_contact, s.fixed_contact, s.edge_contact) = (true, true, true, true);
        }
        f
    };
    let specs = ExplicitBeamsGpu::compute_specs(&fleet(), None);
    let run = |gpu: &mut ExplicitBeamsGpu| {
        gpu.frame(4);
        gpu.wait();
        gpu.contact_status().expect("contact status");
    };
    engine_webgpu::set_pipeline_gate(true);
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet());
    gpu.set_vehicle_contact(true);
    run(&mut gpu);
    // The edge pass first, then the others in turn.
    let deferred: Vec<&str> = specs.iter().filter(|s| s.deferred).map(|s| s.module).collect();
    let (edge, rest): (Vec<&str>, Vec<&str>) = deferred.iter().copied().partition(|m| m.contains("split_edges"));
    assert_eq!(edge.len(), 1, "{deferred:?}");
    for m in edge.iter().chain(&rest) {
        engine_webgpu::mark_pipeline_compiled(m);
        run(&mut gpu);
    }
    engine_webgpu::set_pipeline_gate(false);
}
