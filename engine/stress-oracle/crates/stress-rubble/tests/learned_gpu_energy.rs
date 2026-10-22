//! The learned path's window end must not inject energy when the engine takes the placed
//! bodies on (P0's handover lesson: a body handed to a contact deeper than it would rest
//! fires that overlap). An anchored wall under gravity on the ground, P1's evaluation
//! member, with the punch stub: the mechanical energy of the bodies (kinetic and gravity
//! potential) after each frame following a window end may not rise above the frame
//! before it by more than the plan's 5% of the impact's energy. A file of its own: the
//! switches are read once per process.
//!
//!   cargo test --release -p stress-rubble --features gpu --test learned_gpu_energy -- --nocapture
#![cfg(feature = "gpu")]

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_learned::procedural::{wall_recipes, with_splits_and_ensembles, EnsemblePlan, Ranges};
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_rubble::scenes::{later, realtime};
use stress_rubble::RubbleEngine;

fn energy(w: &mut GpuWorld, gpu: &Gpu) -> f64 {
    w.sync(gpu);
    w.with_shell(|s| {
        let g = s.solver.config.gravity;
        let imp: f64 = s.impactors.iter().map(|i| i.kinetic_energy() - i.mass * g.dot(i.pose.position)).sum();
        let ke: f64 = s.solver.clusters.iter().filter(|c| !c.anchored).map(|c| {
            let r = c.rotation();
            let iw = r * c.inertia * r.transpose();
            0.5 * c.mass * c.velocity.norm2() + 0.5 * c.angular_velocity.dot(iw * c.angular_velocity) - c.mass * g.dot(c.com_world())
        }).sum();
        ke + imp
    })
}

#[test]
fn window_end_injects_no_energy() {
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    std::env::set_var("STRESS_IMPACT_PUNCH", "learned");
    // `STRESS_LEARNED_STUB` may be set by the caller (`punch-overlap`: an adversarial
    // placement, the impactor 5 mm into the plug); the punch stub otherwise
    if std::env::var("STRESS_LEARNED_STUB").is_err() {
        std::env::set_var("STRESS_LEARNED_STUB", "punch");
    }
    let recipes = with_splits_and_ensembles(wall_recipes("anchored_wall", 1, 24, 101, &Ranges::pilot()).expect("recipes"), &EnsemblePlan::default());
    let fast = |r: &&stress_learned::outcome::OutcomeRecipe| r.scene.impactors.iter().any(|i| Vec3::from_array(i.velocity).norm() > 15.0);
    let r = recipes.iter().filter(|r| r.parameters["split"] != "train").find(fast).expect("an evaluation recipe");
    let gpu = Gpu::new().expect("GPU");
    let p = Profile::BlastParityRouted;
    // (its strike moved into the resident frames: the first are coupled ones)
    let mut scene = realtime(later(r.scene.clone(), 4), 0.5);
    p.apply_scene(&mut scene);
    let e0: f64 = scene.impactors.iter().map(|i| 0.5 * i.mass * Vec3::from_array(i.velocity).norm2()).sum();
    let mut eng = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble engine");
    let (iterations, substeps) = p.engine_policy();
    eng.set_policy(iterations, substeps);
    let mut w = GpuWorld::with_resident_engine(&gpu, &scene, eng).expect("resident world");
    w.redo_fractured_frames = false;
    w.set_impact_islands(true);
    w.apply_profile(p);
    w.set_tiered(true).expect("tiered coupling");
    let frames = (scene.sim.duration / scene.sim.frame_dt).round() as usize;
    let mut applied = 0u64;
    let mut after_window_end = false;
    let mut worst = f64::NEG_INFINITY;
    let mut e_prev = energy(&mut w, &gpu);
    for f in 0..frames {
        w.step_frame(&gpu).expect("frame");
        let e = energy(&mut w, &gpu);
        let ram = w.solver.impactors[0].velocity.norm();
        eprintln!("frame {f}: E {e:.1} J, ram {ram:.2} m/s");
        if after_window_end {
            let rise = (e - e_prev) / e0;
            worst = worst.max(rise);
            eprintln!("frame {f}: energy after the frame following a window end {:+.4} E0", rise);
        }
        let now = w.learned.as_ref().map_or(0, |l| l.totals.applied);
        after_window_end = now > applied;
        applied = now;
        e_prev = e;
    }
    eprintln!("{}: E0 {e0:.1} J, {applied} calls applied, worst rise {worst:+.4} E0", r.event_id);
    assert!(applied > 0, "no call applied");
    assert!(worst <= 0.05, "the frame after a window end gained {worst:+.4} E0");
}
