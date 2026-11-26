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

/// Edge contact in a browser host: its path's modules are compiled before the first
/// frame, and whatever order the compiles end in no frame panics.
#[test]
fn a_page_starts_with_edge_contact_compiled() {
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
    // Edge contact changes results, so its path's modules are compiled before a page's first
    // frame: never after the crash (it was 111 s late in Chrome).
    // (As the lab orders them: `prefer_self_split` keeps them ahead.)
    let mut specs = specs;
    ExplicitBeamsGpu::prefer_self_split(&mut specs);
    let (later, ahead): (Vec<_>, Vec<_>) = specs.iter().partition(|s| s.deferred);
    // (With contact offsets the levels-once kernels do not serve: the groups' pass with every pass,
    // `group`, is the one the split path runs.)
    for w in ["member", "tests", "group", "groupbegin", "grouplevels", "grouplv", "edges"] {
        let m = format!("explicit_beams_split_{w}");
        assert!(ahead.iter().any(|s| s.module.starts_with(&m)), "{m} compiles after the page starts");
    }
    for s in ahead.iter().chain(&later) {
        engine_webgpu::mark_pipeline_compiled(s.module);
        run(&mut gpu);
    }
    engine_webgpu::set_pipeline_gate(false);
}
