//! rubble's integrated pipeline (one device, impact islands routed before the engine
//! step, no frame rewind) against the same frame with the reference's rewinds and the
//! standalone oracle.
//!
//!   STRESS_GPU_FIXED_HORIZON=1 cargo test --release -p stress-rubble --features gpu --test integrated
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_ref::scene::{Methods, Scene, SolveMode};
use stress_rubble::RubbleEngine;

fn configure(mut s: Scene, seconds: f64) -> Scene {
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.duration = seconds;
    s
}

struct Outcome {
    ram: f64,
    fragments: f64,
    broken: f64,
    redone: u64,
    momentum: Vec3,
}

fn run(gpu: &Gpu, scene: &Scene, redo: bool) -> Outcome {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let engine = RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble");
    let mut w = GpuWorld::with_engine(gpu, scene, Box::new(engine)).expect("coupled world");
    w.redo_fractured_frames = redo;
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    for _ in 0..frames {
        w.step_frame(gpu).expect("frame");
    }
    w.solver.sync(gpu);
    let o = w.observation();
    let ram = w.solver.impactors.first().map_or(0.0, |i| i.velocity.norm());
    let mut momentum = w.solver.impactors.iter().fold(Vec3::ZERO, |p, i| p + i.velocity * i.mass);
    for cl in &w.solver.mirror.clusters {
        if !cl.anchored {
            momentum = momentum + cl.velocity * cl.mass;
        }
    }
    Outcome { ram, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames(), momentum }
}

fn wall_ram() -> Scene {
    use stress_ref::builders::*;
    use stress_ref::scene::*;
    let mut s = new_scene("wall_ram", "free concrete wall hit by a steel ram");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.1), [10, 2, 11], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.1, 0.1, 0.1] },
        mass: 200.0,
        position: [0.0, -0.15, 0.5],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, 40.0, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s
}

/// A free wall in zero gravity: punch-through without rewinds equals the frame with
/// rewinds (the impact is an island from its first frame), never rewinds, and conserves
/// momentum.
#[test]
fn free_wall_ram_needs_no_rewind() {
    let gpu = Gpu::new().expect("GPU");
    let scene = configure(wall_ram(), 0.25);
    let p0 = Vec3::new(0.0, 200.0 * 40.0, 0.0);
    let a = run(&gpu, &scene, false);
    let b = run(&gpu, &scene, true);
    println!("no rewind: ram {:.3} {} fragments {} broken; with: ram {:.3} {} {} (rewound {})", a.ram, a.fragments, a.broken, b.ram, b.fragments, b.broken, b.redone);
    assert_eq!(a.redone, 0);
    assert!((a.ram - b.ram).abs() < 1e-6 && a.fragments == b.fragments && a.broken == b.broken);
    // The reference's punch-through (REALTIME_ROADMAP A11: 35.05 m/s, 6 fragments).
    assert!((a.ram - 35.05).abs() < 0.1, "ram {}", a.ram);
    assert_eq!(a.fragments, 6.0);
    assert!((a.momentum - p0).norm() < 1e-3 * p0.norm(), "momentum {:?} vs {p0:?}", a.momentum);
}

/// The anchored b5 wall at 20 m/s (the bouncing-projectile case): punch-through without
/// rewinds, the ram leaving at about its speed less the plug's momentum.
#[test]
fn anchored_wall_punch_through_without_rewind() {
    let gpu = Gpu::new().expect("GPU");
    let scene = configure(stress_ref::builders::wall_impact(20.0, "b5_v20"), 0.25);
    let a = run(&gpu, &scene, false);
    let b = run(&gpu, &scene, true);
    println!("no rewind: ram {:.3} {} fragments {} broken; with: ram {:.3} {} {}", a.ram, a.fragments, a.broken, b.ram, b.fragments, b.broken);
    assert_eq!(a.redone, 0);
    assert!((a.ram - b.ram).abs() < 1e-6 && a.fragments == b.fragments && a.broken == b.broken);
    assert!(a.ram > 15.0, "the ram must pass through, not bounce: {}", a.ram);
}

/// b8's sudden column removal: the structure is promoted at its event time, so the
/// release is resolved by the stress dynamics without rewinds and stays near the
/// standalone oracle (11 fragments, 11 bonds at this configuration).
#[test]
fn sudden_support_loss_is_promoted_at_its_event() {
    let gpu = Gpu::new().expect("GPU");
    let named = stress_ref::builders::catalog().into_iter().find(|s| s.name == "b8_frame_sudden").unwrap();
    let scene = configure(named, 0.25);
    let a = run(&gpu, &scene, false);
    println!("no rewind: {} fragments {} broken", a.fragments, a.broken);
    assert_eq!(a.redone, 0);
    assert!((a.fragments - 11.0).abs() <= 2.0 && (a.broken - 11.0).abs() <= 2.0, "{} fragments, {} broken", a.fragments, a.broken);
}
