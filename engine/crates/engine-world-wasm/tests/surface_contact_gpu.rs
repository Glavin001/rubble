//! Surface contact (jbeam-semantics-spec.md §38) on the GPU against the CPU reference,
//! bit for bit: the crash lab's side pole (`sidepole`, the page's spawn) with the
//! showcase sedan1 (the BeamNG profile: fixed-point contact sums), every node's
//! position and velocity after each frame, surface contact off (the default path
//! unchanged) and on. Then the oblique (75°) pole and the frontal pole with it on.

use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::FrameClock;
use engine_world_wasm::ExplicitSim;
use glam::Vec3;

fn sedan1() -> String {
    build(
        &style("sedan1").unwrap(),
        &BuildParams {
            resolution: Resolution::parse("medium").unwrap(),
            strength: 1.0,
            paint: None,
        },
    )
    .unwrap()
    .truck
}

/// Frames of the scene on both, compared bit for bit after each; the surface hits.
fn run(scene: serde_json::Value, frames: usize) -> Option<u64> {
    let (_, device, queue) = engine_webgpu::default_device().ok()?;
    let mut spec = serde_json::json!({
        "vehicles": [{ "truck": sedan1() }],
        "speed": 32.0 / 3.6,
        "timing": false,
        "turbulence": false,
        "runUp": 0.3,
    });
    for (k, v) in scene.as_object().unwrap() {
        spec[k] = v.clone();
    }
    let mut sim = ExplicitSim::new_native(&device, &queue, &spec).unwrap();
    let mut cpu = sim.spawned(0).clone();
    assert!(cpu.fixed_contact, "sedan1 runs the BeamNG profile");
    let mut clock = FrameClock::default();
    let mut hits = 0u64;
    for f in 0..frames {
        let n = clock.next();
        sim.frame_native(n).unwrap();
        cpu.check_wings();
        cpu.rebase_origin();
        for _ in 0..n {
            cpu.step();
            hits += u64::from(cpu.surface_last);
        }
        sim.gpu_solver().wait();
        let p = sim.gpu_solver().positions().remove(0);
        let v = sim.gpu_solver().velocities().remove(0);
        let pc = cpu.positions();
        let bad = (0..p.len()).find(|&i| p[i] != pc[i] || v[i] != cpu.nodes[i].vel);
        if let Some(i) = bad {
            let worst = (0..p.len())
                .map(|i| (p[i] - pc[i]).length())
                .fold(0.0f32, f32::max);
            panic!(
                "{spec_scene} frame {f}: node {i} gpu {:?} {:?} cpu {:?} {:?}; worst position {worst:e} m ({hits} surface hits so far)",
                p[i],
                v[i],
                pc[i],
                cpu.nodes[i].vel,
                spec_scene = spec["scene"]
            );
        }
    }
    let _ = Vec3::ZERO;
    Some(hits)
}

#[test]
fn side_pole_off_is_the_default_path_bit_for_bit() {
    if run(serde_json::json!({ "scene": "sidepole" }), 36).is_none() {
        eprintln!("no GPU: skipped");
    }
}

#[test]
fn side_pole_surface_contact_steps_like_the_cpu() {
    let Some(hits) = run(
        serde_json::json!({ "scene": "sidepole", "surfaceContact": true }),
        36,
    ) else {
        eprintln!("no GPU: skipped");
        return;
    };
    println!("side pole: {hits} surface hits, bit for bit");
    assert!(hits > 100, "the pole was met by the surface: {hits}");
}

#[test]
fn oblique_and_frontal_poles_with_surface_contact_step_like_the_cpu() {
    for scene in [
        serde_json::json!({ "scene": "sidepole", "angle": 75.0, "surfaceContact": true }),
        serde_json::json!({ "scene": "pole", "surfaceContact": true }),
    ] {
        let Some(hits) = run(scene.clone(), 36) else {
            eprintln!("no GPU: skipped");
            return;
        };
        println!("{scene}: {hits} surface hits, bit for bit");
        assert!(hits > 0);
    }
}
