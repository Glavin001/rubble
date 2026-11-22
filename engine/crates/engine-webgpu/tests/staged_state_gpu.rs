//! The browser A/B's readers (`ExplicitBeamsGpu::stage_state`, `stage_snapshot`,
//! `state_differences`; engine-world-wasm `WebWorld::explicitAb`): what they read once
//! mapped is what the blocking readers give natively, so a page's verdict is the native
//! harness's verdict (`vehicle_lab --ab-switch`).

use engine_softbody_authoring::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, state_differences, ExplicitBeamsGpu};
use glam::Vec3;

fn sedan(text: &str, at: Vec3, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings { drag: false, turbulence: false, ..Default::default() };
    let mut s = ExplicitBeams::from_truck_config(text, None, at, vel, settings).expect("spawn");
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(vel);
    s.finish();
    s
}

fn map(device: &wgpu::Device, b: &wgpu::Buffer) {
    let (tx, rx) = std::sync::mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map ok");
}

#[test]
fn staged_reads_match_the_blocking_reads() {
    let Some((_, device, queue)) = default_device().ok() else { return };
    let text = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Low, ..Default::default() }).unwrap().truck;
    let a = sedan(&text, Vec3::ZERO, Vec3::new(0.0, 0.0, 8.0));
    let front = a.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    let b = sedan(&text, Vec3::new(0.0, 0.0, front + 3.0), Vec3::new(0.0, 0.0, -8.0));
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &[a, b]);
    gpu.set_vehicle_contact(true);
    for _ in 0..6 {
        gpu.frame(34);
    }
    gpu.wait();
    let staged = gpu.stage_state();
    map(&device, staged.buffer());
    let got = staged.ab_words();
    let vv = gpu.vv_record_words();
    let mut want = gpu.state_words();
    for (name, v) in &mut want {
        if *name == "misc" {
            v[vv.clone()].fill(0);
        }
    }
    want.extend(gpu.coupling_words());
    assert!(got == want, "staged state words differ from state_words: {:?}", state_differences(&got, &want, 8));
    assert!(state_differences(&got, &want, 8).is_empty());

    let snap = gpu.snapshot();
    let staged = gpu.stage_snapshot(&snap);
    map(&device, staged.buffer());
    assert!(staged.bytes() == gpu.snapshot_to_bytes(&snap), "staged snapshot bytes differ from snapshot_to_bytes");

    // A frame later the state differs, and the differences say where.
    gpu.frame(34);
    gpu.wait();
    let later = gpu.stage_state();
    map(&device, later.buffer());
    let d = state_differences(&got, &later.ab_words(), 8);
    assert!(d.iter().any(|l| l.starts_with("nodes ")), "{d:?}");
}
