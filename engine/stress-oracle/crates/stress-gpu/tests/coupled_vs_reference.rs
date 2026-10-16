//! The engine-coupled GPU stress solver (`coupled::GpuStressSolverApi`) against stress-ref's
//! `EngineCoupledSolver` on the same engine input: frames of contact impulses (impacts,
//! then sustained pushes) and engine-owned motion. Compared per bond (damage,
//! utilization, broken) and by fractures, beside the reference's own spread (the same
//! input with every impulse scaled by 1 + 1e-4).

use stress_gpu::coupled::GpuStressSolverApi;
use stress_gpu::scenes;
use stress_ref::api::{ClusterMotion, ContactImpulse, EngineCoupledSolver, FrameInput, StressSolverApi};
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::scene::{CrushDesc, Scene, Support};

/// The chunk nearest `target` (structure 0, body frame at rest).
fn nearest(scene: &Scene, target: Vec3) -> usize {
    let chunks = &scene.bodies[0].chunks;
    (0..chunks.len()).min_by(|&a, &b| (Vec3::from_array(chunks[a].center) - target).norm().total_cmp(&(Vec3::from_array(chunks[b].center) - target).norm())).unwrap()
}

fn contact(scene: &Scene, chunk: usize, impulse: Vec3, approach: f64, crush: Option<CrushDesc>) -> ContactImpulse {
    let c = &scene.bodies[0].chunks[chunk];
    let center = Vec3::from_array(c.center);
    let dir = impulse * (1.0 / impulse.norm());
    ContactImpulse {
        structure: 0,
        chunk,
        // On the face the impulse pushes into.
        point: center - dir * 0.09,
        impulse,
        other_mass: 800.0,
        approach_speed: approach,
        other_modulus: 30e9,
        other_radius: 0.25,
        crush,
        other: Some(1),
    }
}

/// Frame inputs: `frames` frames of 1/60 s; `impulses(frame)` the contacts of a frame;
/// `motion(frame, solver)` the engine-owned motion.
struct Case {
    name: &'static str,
    scene: Scene,
    /// Run the GPU in adaptive mode (sleeping); the reference stays explicit.
    sleeping: bool,
    frames: usize,
    contacts: Box<dyn Fn(usize, f64) -> Vec<ContactImpulse>>,
    motion: Box<dyn Fn(usize, &dyn StressSolverApi) -> Vec<ClusterMotion>>,
}

fn run(case: &Case, solver: &mut dyn StressSolverApi, scale: f64) -> (Vec<Vec<f64>>, usize) {
    let mut fractures = 0;
    for f in 0..case.frames {
        let input = FrameInput { dt: 1.0 / 60.0, motion: (case.motion)(f, &*solver), contacts: (case.contacts)(f, scale) };
        fractures += solver.step(&input).fractures.len();
    }
    let reports = solver.bond_reports(0);
    let rows = reports.iter().map(|r| vec![r.damage, r.utilization, if r.broken { 1.0 } else { 0.0 }]).collect();
    (rows, fractures)
}

/// Worst damage and utilization differences; bonds broken in one only.
fn compare(a: &[Vec<f64>], b: &[Vec<f64>]) -> (f64, f64, usize) {
    let mut worst = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        worst.0 = worst.0.max((x[0] - y[0]).abs());
        worst.1 = worst.1.max((x[1] - y[1]).abs() / x[1].abs().max(1e-3));
        worst.2 += (x[2] != y[2]) as usize;
    }
    worst
}

#[test]
fn coupled_frames_match_the_reference() {
    let wall = {
        let mut s = scenes::slab([16, 2, 8]);
        // A wall: supported along its base (z = 0) instead of one edge.
        for c in s.bodies[0].chunks.iter_mut() {
            c.support = if c.center[2] < 0.15 { Support::Fixed } else { Support::None };
        }
        s
    };
    let wall_target = nearest(&wall, Vec3::new(1.6, 0.0, 0.8));
    let free = {
        let mut s = scenes::slab([10, 4, 2]);
        for c in s.bodies[0].chunks.iter_mut() {
            c.support = Support::None;
        }
        s
    };
    let free_target = nearest(&free, Vec3::new(0.0, 0.4, 0.2));
    let wall_scene = wall.clone();
    let wall_sleep = wall_scene.clone();
    let free_scene = free.clone();
    let cases = vec![
        Case {
            name: "wall: ram impact, then a sustained push",
            scene: wall.clone(),
            sleeping: false,
            frames: 12,
            contacts: Box::new(move |f, k| match f {
                0..=2 => vec![contact(&wall_scene, wall_target, Vec3::new(0.0, 6000.0 * k / (f as f64 + 1.0), 0.0), 12.0, None)],
                _ => vec![contact(&wall_scene, wall_target, Vec3::new(0.0, 400.0 * k, 0.0), 0.0, None)],
            }),
            motion: Box::new(|_, _| Vec::new()),
        },
        Case {
            name: "free slab: engine-driven motion, crushing impactor",
            scene: free,
            sleeping: false,
            frames: 10,
            contacts: Box::new(move |f, k| {
                if f < 3 {
                    let crush = CrushDesc { max_force: 2.0e5, energy: 2.0e3 };
                    vec![contact(&free_scene, free_target, Vec3::new(0.0, 0.0, 1500.0 * k), 9.0, Some(crush))]
                } else {
                    Vec::new()
                }
            }),
            motion: Box::new(|f, s| {
                // The engine's motion: a slow spin and drift of every cluster.
                let t = (f + 1) as f64 / 60.0;
                s.clusters()
                    .iter()
                    .filter(|c| !c.anchored)
                    .map(|c| {
                        let pose = Pose {
                            position: Vec3::new(c.pose[0], c.pose[1], c.pose[2]) + Vec3::new(0.02, 0.0, -0.01),
                            rotation: Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.3 * t),
                        };
                        ClusterMotion { id: c.id, pose, velocity: Vec3::new(1.2, 0.0, -0.6), angular_velocity: Vec3::new(0.0, 0.0, 0.3) }
                    })
                    .collect()
            }),
        },
        Case {
            name: "wall, sleeping: ram impact, then a sustained push",
            scene: wall.clone(),
            sleeping: true,
            frames: 12,
            contacts: Box::new(move |f, k| match f {
                0..=2 => vec![contact(&wall_sleep, wall_target, Vec3::new(0.0, 6000.0 * k / (f as f64 + 1.0), 0.0), 12.0, None)],
                _ => vec![contact(&wall_sleep, wall_target, Vec3::new(0.0, 400.0 * k, 0.0), 0.0, None)],
            }),
            motion: Box::new(|_, _| Vec::new()),
        },
        Case {
            name: "wall, sleeping: idle (no contacts)",
            scene: wall,
            sleeping: true,
            frames: 12,
            contacts: Box::new(|_, _| Vec::new()),
            motion: Box::new(|_, _| Vec::new()),
        },
    ];
    for case in &cases {
        // The reference runs come from the cache (gate.rs), keyed by the case and its
        // scene; the contact and motion closures are not serializable, so
        // `INPUTS_VERSION` stands for them: bump it when they change.
        const INPUTS_VERSION: u32 = 1;
        let key = |scale: f64| (case.name, &case.scene, case.frames, scale.to_bits(), INPUTS_VERSION);
        let label = format!("coupled-{}", case.name);
        let (r_rows, r_frac) = stress_gpu::gate::cached(&label, &key(1.0), || run(case, &mut EngineCoupledSolver::new(&case.scene), 1.0));
        let (p_rows, p_frac) = stress_gpu::gate::cached(&label, &key(1.0 + 1e-4), || run(case, &mut EngineCoupledSolver::new(&case.scene), 1.0 + 1e-4));
        let gpu_scene = if case.sleeping { case.scene.with_override("sim.solve_mode", "\"adaptive\"").expect("adaptive") } else { case.scene.clone() };
        let mut gpu = GpuStressSolverApi::new(&gpu_scene).expect("GPU coupled solver");
        let (g_rows, g_frac) = run(case, &mut gpu, 1.0);
        let gpu_err = compare(&r_rows, &g_rows);
        let spread = compare(&r_rows, &p_rows);
        let max_damage = r_rows.iter().map(|r| r[0]).fold(0.0, f64::max);
        let broken = r_rows.iter().filter(|r| r[2] != 0.0).count();
        eprintln!(
            "{}: {} bonds, reference max damage {max_damage:.3}, {broken} broken, {r_frac} fractures | gpu: damage {:.1e}, utilization {:.1e}, broken differs on {} ({g_frac} fractures) | reference spread: damage {:.1e}, utilization {:.1e}, broken differs on {} ({p_frac} fractures)",
            case.name,
            r_rows.len(),
            gpu_err.0,
            gpu_err.1,
            gpu_err.2,
            spread.0,
            spread.1,
            spread.2
        );
        assert!(gpu_err.0 <= (2.0 * spread.0).max(1e-3), "{}: damage differs by {:.1e}", case.name, gpu_err.0);
        assert!(gpu_err.2 <= (2 * spread.2).max(1), "{}: {} bonds broken differently", case.name, gpu_err.2);
        assert!(gpu_err.1 <= (2.0 * spread.1).max(1e-3), "{}: utilization differs by {:.1e}", case.name, gpu_err.1);
    }
}
