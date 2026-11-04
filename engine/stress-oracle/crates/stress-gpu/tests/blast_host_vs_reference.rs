//! The GPU port of Blast's stress solver, host path (`stress_gpu::blast`, the Slang kernels
//! blast_cgls and blast_material), against its f64 reference (`stress_ref::blast_bond`):
//! the same iterate frame by frame, capped and warm-started, within f32 rounding; the same
//! iteration counts and convergence; the same stresses and damage; bit-identical reruns.

use stress_gpu::blast::GpuBlastSolver;
use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(g) => Some(g),
        Err(_) => {
            eprintln!("no GPU; skipping");
            None
        }
    }
}

fn house(settings: BlastSettings) -> (BlastGraph, Vec<BlastMaterial>, Vec3) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let options = stress_ref::scene_pack::ImportOptions {
        authored_strengths: true,
        authored_fatal: true,
        pack_masses: true,
        pack_modulus: true,
        blast_authoring: true,
    };
    let scene = stress_ref::scene_pack::import_with(&path, "house", "house", options)
        .expect("house-1story");
    let (graph, materials) = BlastAuthoring::graph(&scene, settings).unwrap();
    (graph, materials, Vec3::from_array(scene.gravity))
}

/// A portal frame on two supports, with a diagonal: redundant, so the answer is the least
/// compliance one.
fn frame(settings: BlastSettings) -> BlastGraph {
    let node = |x: f64, y: f64, mass: f64| BlastNode {
        position: Vec3::new(x, y, 0.0),
        mass,
        inertia: 0.0,
        volume: 0.5,
    };
    let nodes = vec![
        node(0.0, 0.0, 0.0),
        node(4.0, 0.0, 0.0),
        node(0.0, 1.0, 8.0),
        node(4.0, 1.0, 12.0),
        node(0.0, 2.0, 8.0),
        node(2.0, 2.0, 6.0),
        node(4.0, 2.0, 9.0),
    ];
    let pairs = [(0, 2), (2, 4), (1, 3), (3, 6), (4, 5), (5, 6), (2, 5)];
    let bonds = pairs
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| {
            let (pa, pb): (Vec3, Vec3) = (nodes[a].position, nodes[b].position);
            BlastBond {
                nodes: [a, b],
                centroid: (pa + pb) * 0.5,
                normal: (pb - pa).normalized(),
                area: 0.2 + 0.1 * k as f64,
                modulus: 10e9 + 5e9 * k as f64,
                material: 0,
            }
        })
        .collect();
    BlastGraph::new(nodes, bonds, settings).unwrap()
}

fn uniform(n: usize, lin: Vec3) -> Vec<AngLin> {
    vec![
        AngLin {
            ang: Vec3::ZERO,
            lin
        };
        n
    ]
}

/// The largest difference between two impulse sets, relative to the largest impulse.
fn difference(a: &[AngLin], b: &[AngLin]) -> f64 {
    let scale = b
        .iter()
        .map(|j| j.lin.norm().max(j.ang.norm()))
        .fold(0.0, f64::max)
        .max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x.lin - y.lin).norm().max((x.ang - y.ang).norm()))
        .fold(0.0, f64::max)
        / scale
}

fn reference_impulses(r: &BlastSolver) -> Vec<AngLin> {
    (0..r.graph.bond_count()).map(|j| r.impulse(j)).collect()
}

/// The reference in the device's precision (f32): the same recurrence in the same arithmetic,
/// differing from the device only in the order of its sums.
fn single(graph: &BlastGraph) -> BlastSolver {
    let mut s = BlastSolver::new(graph.clone());
    s.device_precision = true;
    s
}

/// How far one ulp moves the f32 reference's iterates over `loads`: the reference against
/// itself with every load nudged by f32's unit roundoff (two sign patterns), frame by frame.
/// A capped CG iterate on an ill-conditioned island amplifies any rounding difference, so this
/// is the spread any f32 implementation of the solve has, Blast's own CUDA one with its
/// run-to-run atomic order included. Returns the worst relative difference.
fn rounding_spread(graph: &BlastGraph, loads: &[Vec<AngLin>], skip: bool) -> f64 {
    let eps = f32::EPSILON as f64;
    let mut worst = 0.0f64;
    for pattern in 0..2usize {
        let mut a = single(graph);
        let mut b = single(graph);
        for accel in loads {
            let nudged: Vec<AngLin> = accel
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let k = 1.0 + eps * if (i + pattern) % 2 == 0 { 1.0 } else { -1.0 };
                    AngLin {
                        ang: v.ang * k,
                        lin: v.lin * k,
                    }
                })
                .collect();
            a.solve(accel, skip);
            b.solve(&nudged, skip);
            worst = worst.max(difference(&reference_impulses(&b), &reference_impulses(&a)));
        }
    }
    worst
}

/// Frame by frame against the f32 reference, within `bound` or (when larger) ten times the
/// spread one ulp of rounding produces; also reports the f64 reference's distance. Returns the
/// worst difference from the f32 reference.
fn track(
    gpu: &Gpu,
    graph: BlastGraph,
    materials: Vec<BlastMaterial>,
    loads: &[Vec<AngLin>],
    skip: bool,
    bound: f64,
) -> f64 {
    let spread = rounding_spread(&graph, loads, skip);
    let bound = bound.max(10.0 * spread);
    let mut reference = single(&graph);
    let mut exact = BlastSolver::new(graph.clone());
    let mut device = GpuBlastSolver::new(gpu, graph, materials);
    let (mut worst, mut worst_exact) = (0.0f64, 0.0f64);
    for (frame, accel) in loads.iter().enumerate() {
        let r = reference.solve(accel, skip);
        exact.solve(accel, skip);
        let d = device.solve(accel, skip);
        if frame == 0 {
            assert_eq!(device.islands(), reference.islands);
        }
        assert_eq!(r.len(), d.len());
        for (k, (r, d)) in r.iter().zip(&d).enumerate() {
            assert_eq!(
                r.skipped, d.skipped,
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
            assert_eq!(
                r.converged, d.converged,
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
            // CG's residual falls by orders of magnitude at its last steps (finite termination
            // on a small island), where the order of sums can move the stop by a few iterations.
            assert!(
                r.iterations.abs_diff(d.iterations) <= 2 + r.iterations / 5,
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
        }
        let impulses = device.impulses();
        let e = difference(&impulses, &reference_impulses(&reference));
        worst = worst.max(e);
        worst_exact = worst_exact.max(difference(&impulses, &reference_impulses(&exact)));
        assert!(e < bound, "frame {frame}: impulses differ by {e:.3e} of the largest (bound {bound:.3e}, one-ulp spread {spread:.3e})");
    }
    println!("worst {worst:.3e} from the f32 reference (bound {bound:.3e}, one-ulp spread {spread:.3e}); {worst_exact:.3e} from the f64 one");
    worst
}

#[test]
fn a_capped_warm_started_frame_sequence_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    // The frame: 32 iterations a frame (vibe-land's), a load that changes each frame.
    let graph = frame(BlastSettings::default());
    let loads: Vec<_> = (0..8)
        .map(|f| {
            let mut a = uniform(graph.node_count(), Vec3::new(0.0, -9.81, 0.0));
            a[5].lin.x = 2.0 * (f as f64 * 0.7).sin();
            a
        })
        .collect();
    let e = track(
        &gpu,
        graph,
        vec![BlastMaterial::CONCRETE],
        &loads,
        false,
        1e-4,
    );
    println!("frame: worst {e:.3e}");

    // 2 iterations a frame: far from converged, every frame continues the last.
    let graph = frame(BlastSettings {
        max_iterations: 2,
        ..Default::default()
    });
    let loads = vec![uniform(graph.node_count(), Vec3::new(0.0, -9.81, 0.0)); 20];
    let e = track(
        &gpu,
        graph,
        vec![BlastMaterial::CONCRETE],
        &loads,
        false,
        1e-4,
    );
    println!("frame, 2 iterations: worst {e:.3e}");
}

#[test]
#[ignore = "diagnostic: the difference by iteration count on a cold solve"]
fn difference_by_iteration() {
    let Some(gpu) = gpu() else { return };
    for its in [1u32, 8, 16, 20, 24, 28, 32, 64] {
        let (graph, materials, g) = house(BlastSettings {
            max_iterations: its,
            ..Default::default()
        });
        let mut reference = BlastSolver::new(graph.clone());
        let mut device = GpuBlastSolver::new(&gpu, graph, materials);
        let accel = uniform(reference.graph.node_count(), g);
        let r = reference.solve(&accel, false);
        let d = device.solve(&accel, false);
        let e = difference(&device.impulses(), &reference_impulses(&reference));
        // The device against itself with every load one f32 ulp off.
        let mut nudged = GpuBlastSolver::new(
            &gpu,
            reference.graph.clone(),
            vec![BlastMaterial::CONCRETE; 32],
        );
        let ulp: Vec<AngLin> = accel
            .iter()
            .enumerate()
            .map(|(i, v)| AngLin {
                ang: v.ang,
                lin: v.lin * (1.0 + f32::EPSILON as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }),
            })
            .collect();
        nudged.solve(&ulp, false);
        let own = difference(&nudged.impulses(), &device.impulses());
        let mut single = BlastSolver::new(reference.graph.clone());
        single.device_precision = true;
        single.solve(&accel, false);
        let f32_ref = difference(&device.impulses(), &reference_impulses(&single));
        println!("{its:3} iterations: {e:.3e} vs f64, {f32_ref:.3e} vs f32 reference (device vs itself, one ulp: {own:.3e})  residual {:.4e} vs {:.4e}", r[0].residual2, d[0].residual2);
    }
}

#[test]
fn a_converged_house_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    for weighting in [MassWeighting::Equalized, MassWeighting::Native] {
        let (graph, materials, g) = house(BlastSettings {
            mass_weighting: weighting,
            max_iterations: 20_000,
            tolerance: 1e-5,
            ..Default::default()
        });
        let mut reference = BlastSolver::new(graph.clone());
        let mut device = GpuBlastSolver::new(&gpu, graph, materials);
        let accel = uniform(reference.graph.node_count(), g);
        let r = reference.solve(&accel, false);
        let d = device.solve(&accel, false);
        let e = difference(&device.impulses(), &reference_impulses(&reference));
        let its = |v: &[IslandReport]| v.iter().map(|x| x.iterations).collect::<Vec<_>>();
        println!(
            "converged house {weighting:?}: {e:.3e}; iterations {:?} vs {:?}",
            its(&r),
            its(&d)
        );
        assert!(r.iter().chain(&d).all(|x| x.converged));
        assert!(e < 1e-3, "{e:.3e}");
    }
}

#[test]
fn a_house_under_gravity_matches_the_reference() {
    let Some(gpu) = gpu() else { return };
    for weighting in [MassWeighting::Equalized, MassWeighting::Native] {
        let (graph, materials, g) = house(BlastSettings {
            mass_weighting: weighting,
            ..Default::default()
        });
        let loads = vec![uniform(graph.node_count(), g); 12];
        let e = track(&gpu, graph, materials, &loads, true, 2e-3);
        println!("house {weighting:?}: worst {e:.3e}");
    }
}

#[test]
fn stresses_and_damage_match_the_reference_formula() {
    let Some(gpu) = gpu() else { return };
    // The house with its limits scaled down so gravity overloads some bonds, and the material
    // kernel against the reference's formula and law applied to the device's own impulses.
    let (graph, mut materials, g) = house(BlastSettings::default());
    for m in &mut materials {
        for v in [
            &mut m.compression_elastic,
            &mut m.compression_fatal,
            &mut m.tension_elastic,
            &mut m.tension_fatal,
            &mut m.shear_elastic,
            &mut m.shear_fatal,
        ] {
            *v *= 1e-3;
        }
    }
    let settings = graph.settings;
    let mut device = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
    let accel = uniform(graph.node_count(), g);
    for _ in 0..4 {
        device.solve(&accel, false);
    }
    let dt = 1.0 / 60.0;
    let results = device.bond_results(dt, false);
    let healths = device.health();
    let (mut commands, mut agree) = (0, 0);
    for (j, d) in results.iter().enumerate() {
        let health = healths[j];
        let r = bond_stress(
            d.impulse,
            graph.normal[j],
            health,
            graph.node_distance[j],
            settings.bend_gain_max,
            settings.fibres,
        );
        let scale = r
            .compression
            .max(r.tension)
            .max(r.shear)
            .max(r.normal.abs())
            .max(1e-30);
        for (a, b) in [
            (d.stress.normal, r.normal),
            (d.stress.shear, r.shear),
            (d.stress.bend, r.bend),
            (d.stress.compression, r.compression),
            (d.stress.tension, r.tension),
        ] {
            assert!(
                (a - b).abs() <= 1e-5 * scale,
                "bond {j}: {:?} vs {r:?}",
                d.stress
            );
        }
        let rd = bond_damage(
            r,
            health,
            graph.bonds[j].area,
            &materials[graph.bonds[j].material],
            dt,
            settings.damage_rate,
        );
        commands += rd.command as usize;
        // A stress within rounding of a limit may fall either side of it in f32.
        agree += (rd.command == d.damage.command) as usize;
        if rd.command && d.damage.command {
            assert!(
                (rd.damage - d.damage.damage).abs() <= 1e-4 * health,
                "bond {j}: {rd:?} vs {:?}",
                d.damage
            );
            assert!(
                (rd.multiplier - d.damage.multiplier).abs() <= 1e-4 * rd.multiplier.max(1.0),
                "bond {j}: {rd:?} vs {:?}",
                d.damage
            );
        }
    }
    println!(
        "{commands} damage commands, {agree} of {} verdicts agree",
        results.len()
    );
    assert!(commands > 0, "the scaled limits overload some bonds");
    assert!(agree + 2 >= results.len(), "{agree} of {}", results.len());

    // Applying the damage: health falls by each verdict's damage, and a bond at zero breaks.
    let before = healths;
    let applied = device.bond_results(dt, true);
    let after = device.health();
    for (j, d) in applied.iter().enumerate() {
        if d.damage.command {
            let expected = (before[j] - d.damage.damage).max(0.0);
            assert!(
                (after[j] - expected).abs() <= 1e-6 * before[j].max(1.0),
                "bond {j}: {} vs {expected}",
                after[j]
            );
        } else {
            assert_eq!(after[j] as f32, before[j] as f32, "bond {j}");
        }
    }
    let broken = after.iter().filter(|h| !(**h > 0.0)).count();
    println!("{broken} bonds broken");
    // The islands the GPU rebuilds from the damaged health, before the next solve.
    device.solve(&accel, false);
    assert_eq!(device.islands(), Islands::compute(&graph, &after));
}

#[test]
fn breaking_bonds_repartitions_as_the_reference() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings::default());
    let accel = uniform(graph.node_count(), g);
    let spread = rounding_spread(&graph, &vec![accel.clone(); 7], false);
    let mut reference = single(&graph);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    for frame in 0..6 {
        reference.solve(&accel, false);
        device.solve(&accel, false);
        let j = frame * 7;
        reference.set_health(j, 0.0);
        device.set_health(j, 0.0);
    }
    reference.solve(&accel, false);
    device.solve(&accel, false);
    assert_eq!(device.islands(), reference.islands);
    let e = difference(&device.impulses(), &reference_impulses(&reference));
    assert!(
        e < 10.0 * spread.max(1e-5),
        "{e:.3e} (one-ulp spread {spread:.3e})"
    );
}

#[test]
fn a_rerun_is_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings::default());
    let accel = uniform(graph.node_count(), g);
    let run = || {
        let mut s = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
        for _ in 0..3 {
            s.solve(&accel, false);
        }
        s.solution()
    };
    let (a, b) = (run(), run());
    assert!(a.iter().zip(&b).all(|(x, y)| x
        .ang
        .to_array()
        .iter()
        .chain(&x.lin.to_array())
        .zip(y.ang.to_array().iter().chain(&y.lin.to_array()))
        .all(|(p, q)| p.to_bits() == q.to_bits())));
}
