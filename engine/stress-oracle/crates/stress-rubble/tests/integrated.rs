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
    observation: stress_ref::observation::Observation,
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
    Outcome { ram, fragments: o.values["fragments"], broken: o.values["broken_bonds"], redone: w.redone_frames(), momentum, observation: o }
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
/// release is resolved by the stress dynamics without rewinds. The collapse is chaotic
/// (stress-gpu `gate::is_chaotic`: the reference's broken bonds scatter over a wide
/// range under 1e-6..1e-4 input changes), so one run against one reference run is a
/// draw from each distribution. As the accuracy gate judges such a case, the median of
/// the runs on the same input changes must lie within the reference ensemble's range
/// (counts only: this test never judged the probes).
///
/// The reference gives 7-14 broken bonds. rubble gave 11-40 (median 29) once its
/// broadphase kept dynamic-static pairs whose static has the lower index (before, freed
/// pieces fell through the frame's anchored parts): engine shapes built undeformed held a
/// freed top-beam piece flush between the frame's anchored halves, where the stress solve
/// had a widening gap, and it wedged and broke. With the shapes following the chunks'
/// deformation (`World::shape_poses`) it gives 13-17 (median 14).
#[test]
fn sudden_support_loss_is_promoted_at_its_event() {
    use stress_gpu::gate::{ensemble_gate, is_chaotic, perturbations, perturbed_by, reference_set};
    let gpu = Gpu::new().expect("GPU");
    let named = stress_ref::builders::catalog().into_iter().find(|s| s.name == "b8_frame_sudden").unwrap();
    let scene = configure(named, 0.25);
    let refs = reference_set(&scene, true).expect("reference ensemble");
    let mut runs = Vec::new();
    for s in std::iter::once(scene.clone()).chain(perturbations().into_iter().map(|e| perturbed_by(&scene, e))) {
        let a = run(&gpu, &s, false);
        assert_eq!(a.redone, 0);
        runs.push(a.observation);
    }
    let broken = |obs: &[&stress_ref::observation::Observation]| -> Vec<f64> { obs.iter().map(|o| o.values["broken_bonds"]).collect() };
    let reference: Vec<_> = std::iter::once(&refs.reference).chain(std::iter::once(&refs.half)).chain(&refs.spreads).collect();
    println!("broken bonds: reference {:?}, no rewind {:?}", broken(&reference), broken(&runs.iter().collect::<Vec<_>>()));
    assert!(is_chaotic(&refs), "b8 is judged as a chaotic case");
    let fails: Vec<String> = ensemble_gate(&refs, &runs).into_iter().filter(|f| !f.starts_with("median probes")).collect();
    assert!(fails.is_empty(), "{fails:?}");
}

/// A free concrete beam (nine 0.3 m chunks) landing on an anchored block: the coupled
/// pipeline breaks what the standalone oracle breaks, from gentle (0.09 m/s, nothing
/// breaks) to breaking (1 m/s), centred and off centre.
#[test]
fn a_beam_landing_on_an_anchored_block_breaks_as_the_oracle() {
    use stress_ref::builders::*;
    use stress_ref::scene::Support;
    let gpu = Gpu::new().expect("GPU");
    for (offset, speed) in [(4.0, 0.09), (0.0, 0.09), (4.0, 1.0), (1.0, 1.0), (0.0, 1.0)] {
        let mut s = new_scene("beam_landing", "free beam lands on an anchored block");
        s.materials.insert("concrete".into(), oracle_concrete(None));
        let mut block = chunk(Vec3::new(0.0, 0.0, -0.15), Vec3::splat(0.15), "concrete");
        block.support = Support::Fixed;
        s.bodies.push(body("block", vec![block], vec![]));
        // 1 mm above the block, which lies under chunk `offset`.
        let chunks = grid(Vec3::new(-0.15 - offset * 0.3, -0.15, 0.001), [9, 1, 1], Vec3::splat(0.3), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut beam = body("beam", chunks, bonds);
        beam.linear_velocity = [0.0, 0.0, -speed];
        s.bodies.push(beam);
        let scene = configure(s, 0.1);
        let oracle = stress_ref::world::World::new(&scene).run().values["broken_bonds"];
        let ours = run(&gpu, &scene, false).broken;
        assert_eq!(ours, oracle, "offset {offset}, {speed} m/s: {ours} broken, the oracle {oracle}");
    }
}

/// A free concrete beam fitting exactly between two anchored walls (its bonds to them
/// already broken), falling at 0.1 m/s and turning about its width: it slides out as
/// in the oracle (nothing breaks). Without the walls' deformation (none here), a freed
/// piece flush between anchored faces is no harm.
#[test]
fn a_beam_freed_between_anchored_walls_breaks_as_the_oracle() {
    use stress_ref::builders::*;
    use stress_ref::scene::Support;
    let gpu = Gpu::new().expect("GPU");
    for spin in [0.0, 0.08, 0.3] {
        let mut s = new_scene("beam_wedge", "freed beam between two anchored walls");
        s.materials.insert("concrete".into(), oracle_concrete(None));
        for (name, x) in [("left", -0.3), ("right", 2.7)] {
            // Five bonded chunks, 1.5 m tall, the beam level with the middle one.
            let mut wall = grid(Vec3::new(x, -0.15, -0.6), [1, 1, 5], Vec3::splat(0.3), "concrete");
            for c in &mut wall {
                c.support = Support::Fixed;
            }
            let bonds = auto_bonds(&wall, |_, _| "concrete".into());
            s.bodies.push(body(name, wall, bonds));
        }
        let chunks = grid(Vec3::new(0.0, -0.15, 0.0), [9, 1, 1], Vec3::splat(0.3), "concrete");
        let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
        let mut beam = body("beam", chunks, bonds);
        beam.linear_velocity = [0.0, 0.0, -0.1];
        beam.angular_velocity = [0.0, spin, 0.0];
        s.bodies.push(beam);
        let scene = configure(s, 0.1);
        let oracle = stress_ref::world::World::new(&scene).run().values["broken_bonds"];
        let ours = run(&gpu, &scene, false).broken;
        assert_eq!(ours, oracle, "spin {spin} rad/s: {ours} broken, the oracle {oracle}");
    }
}

/// Bit-level summary of the mirror's state: cluster motion and membership, every chunk's
/// hidden state, every bond's history and force.
fn mirror_state(m: &stress_ref::solver::ReferenceSolver) -> String {
    let mut out = String::new();
    for cl in &m.clusters {
        out += &format!("{} {:?} {:?} {:?} {:?} {:?}\n", cl.id, cl.chunks, cl.pose, cl.velocity, cl.angular_velocity, cl.activity);
    }
    for (s, chunks) in m.chunks.iter().enumerate() {
        for st in chunks {
            out += &format!("{s} {:?} {:?} {:?} {:?}\n", st.u, st.th, st.v, st.w);
        }
    }
    for bonds in &m.bonds {
        for b in bonds {
            out += &format!("{:?} {:?} {}\n", b.joint, b.force, b.alive);
        }
    }
    out
}

/// A segment snapshot leaves the structures whose clusters are all quiet (off the GPU)
/// out of its copy; restoring it gives the same mirror as restoring a full snapshot of
/// the same instant, after GPU substeps changed the live state in between.
#[test]
fn segment_snapshot_restores_like_a_full_one() {
    let gpu = Gpu::new().expect("GPU");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let house = stress_ref::scene_pack::import(&path, "house-1story", "structure").expect("house-1story pack");
    let town = stress_ref::scenes::bombarded_exact(stress_ref::scenes::pack_town(&house, 4, 2.0), 1);
    let scene = configure(town, 0.25);
    std::env::set_var("STRESS_GPU_FIXED_HORIZON", "1");
    let engine = RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble");
    let mut w = GpuWorld::with_engine(&gpu, &scene, Box::new(engine)).expect("coupled world");
    w.redo_fractured_frames = false;
    for _ in 0..3 {
        w.step_frame(&gpu).expect("frame");
    }
    assert!(w.solver.quiet_count() > 0, "the untouched houses should be quiet");
    let full = w.solver.snapshot(&gpu);
    let segment = w.solver.snapshot_segment(&gpu);
    let dt = 1e-6;
    w.solver.step(&gpu, dt, 200).expect("substeps");
    w.solver.restore(&gpu, segment).expect("restore segment");
    w.solver.sync(&gpu);
    let after_segment = mirror_state(&w.solver.mirror);
    w.solver.step(&gpu, dt, 200).expect("substeps");
    w.solver.restore(&gpu, full).expect("restore full");
    w.solver.sync(&gpu);
    let after_full = mirror_state(&w.solver.mirror);
    assert!(after_segment == after_full, "segment restore differs from full restore");
}
