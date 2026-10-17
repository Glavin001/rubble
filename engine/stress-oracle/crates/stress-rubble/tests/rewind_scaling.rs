//! The rewind-scaling scenes (`findings/rewind-scaling.md`): a ram through a row of free
//! walls, every wall it reaches within a frame routed into the frame's impact island
//! before the engine steps, so rubble's pipeline reaches the oracle's outcome without
//! rewinding a frame, whatever the number of walls.
//!
//!   cargo test --release -p stress-rubble --features gpu --test rewind_scaling
#![cfg(feature = "gpu")]

mod rewind_scenes;

use rewind_scenes::wall_row;
use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_rubble::RubbleEngine;

struct Outcome {
    ram: f64,
    fragments: f64,
    broken: f64,
    redone: u64,
    unverified: u64,
}

fn rubble(gpu: &Gpu, scene: &Scene, redo: bool) -> Outcome {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let engine = RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble");
    let mut w = GpuWorld::with_engine(gpu, scene, Box::new(engine)).expect("coupled world");
    w.redo_fractured_frames = redo;
    for _ in 0..(scene.sim.duration / scene.sim.frame_dt).round() as usize {
        w.step_frame(gpu).expect("frame");
    }
    w.solver.sync(gpu);
    let o = w.observation();
    Outcome { ram: w.solver.impactors[0].velocity.y, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames(), unverified: w.unverified_fractures }
}

fn oracle(scene: &Scene) -> Outcome {
    let mut w = stress_ref::world::World::new(scene);
    let o = w.run();
    Outcome { ram: w.impactors[0].velocity.y, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: 0, unverified: 0 }
}

/// Two walls at 40 m/s: the ram crosses both within 1.5 frames; the first is punched
/// (a plug), the second shattered by the plug and debris. Without a rewind (and with
/// the redo on, none is taken: routing left nothing for it) rubble ends where the oracle
/// does. Measured: oracle 23.62 m/s, 133 fragments, 401 bonds; rubble 23.4-23.9 m/s,
/// 123-137 fragments, 386-406 bonds (fixed horizons: 23.67, 123, 386).
#[test]
fn ram_through_two_walls_without_rewinds() {
    let Ok(gpu) = Gpu::new() else {
        eprintln!("no GPU; skipping");
        return;
    };
    let scene = wall_row(2, 40.0, 0.1, 0.15);
    let truth = oracle(&scene);
    for redo in [false, true] {
        let r = rubble(&gpu, &scene, redo);
        println!(
            "redo {redo}: ram {:.2} m/s ({:.2}), {} fragments ({}), {} bonds ({}), {} redone, {} unverified",
            r.ram, truth.ram, r.fragments, truth.fragments, r.broken, truth.broken, r.redone, r.unverified
        );
        assert_eq!(r.redone, 0, "no frame needs a rewind");
        assert_eq!(r.unverified, 0, "no fracture happens under the engine's loads");
        assert!((r.ram - truth.ram).abs() < 1.5, "ram {:.2} vs oracle {:.2}", r.ram, truth.ram);
        assert!((r.fragments - truth.fragments).abs() <= 0.15 * truth.fragments, "fragments {} vs oracle {}", r.fragments, truth.fragments);
        assert!((r.broken - truth.broken).abs() <= 0.1 * truth.broken, "bonds {} vs oracle {}", r.broken, truth.broken);
    }
}
