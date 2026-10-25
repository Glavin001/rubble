//! The GPU port of PhysX's resident (native) Blast solve (`shaders/blast_native.slang`)
//! against its reference (`stress_ref::blast_bond`, `SolverKind::Native`): converged, the f64
//! forces; capped and warm-started frame after frame, the f32 reference within its own one-ulp
//! spread, iteration counts alike; free pieces, zero loads and the settled certificate as the
//! reference has them; bit-identical reruns.

use stress_gpu::blast::GpuBlastSolver;
use stress_gpu::gpu::Gpu;
use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

fn gpu() -> Option<Gpu> {
    Gpu::new().map_err(|_| eprintln!("no GPU; skipping")).ok()
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

fn uniform(n: usize, lin: Vec3) -> Vec<AngLin> {
    vec![
        AngLin {
            ang: Vec3::ZERO,
            lin
        };
        n
    ]
}

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

fn single(graph: &BlastGraph) -> BlastSolver {
    let mut s = BlastSolver::new(graph.clone());
    s.device_precision = true;
    s
}

/// The f32 reference against itself with every load one ulp off (two patterns), frame by frame.
fn rounding_spread(graph: &BlastGraph, loads: &[Vec<AngLin>]) -> f64 {
    let eps = f32::EPSILON as f64;
    let mut worst = 0.0f64;
    for pattern in 0..2usize {
        let (mut a, mut b) = (single(graph), single(graph));
        for accel in loads {
            let nudged: Vec<AngLin> = accel
                .iter()
                .enumerate()
                .map(|(i, v)| AngLin {
                    ang: v.ang,
                    lin: v.lin * (1.0 + eps * if (i + pattern) % 2 == 0 { 1.0 } else { -1.0 }),
                })
                .collect();
            a.solve(accel, false);
            b.solve(&nudged, false);
            worst = worst.max(difference(&reference_impulses(&b), &reference_impulses(&a)));
        }
    }
    worst
}

#[test]
fn converged_native_solves_are_the_f64_answer() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings {
        max_iterations: 2000,
        tolerance: 1e-4,
        ..BlastSettings::native()
    });
    let accel = uniform(graph.node_count(), g);
    let mut reference = BlastSolver::new(graph.clone());
    let r = reference.solve(&accel, false);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    let d = device.solve(&accel, false);
    let e = difference(&device.impulses(), &reference_impulses(&reference));
    let mut f32_reference = single(&reference.graph);
    let s = f32_reference.solve(&accel, false);
    println!(
        "converged: {e:.3e}; iterations {} vs {}; f32 reference {s:?}",
        r[0].iterations, d[0].iterations
    );
    // The f64 solve converges. In f32 the residual test has a floor on this house far above the
    // tolerance (it reads |B^T r| from a difference of nearly equal terms) while the forces are
    // within 1e-5 of f64: the f32 reference stalls the same way (stressgpu/detail/README.md: "on
    // stiff structures it can read 1e3x over tolerance with forces within 1e-3", the reason for
    // the force tolerance). So the device is judged by its forces.
    assert!(r[0].converged, "{r:?}");
    assert!(e < 1e-4, "{e:.3e}");
    assert_eq!(
        d[0].converged, s[0].converged,
        "the device and the f32 reference agree on convergence: {d:?} vs {s:?}"
    );
}

#[test]
fn capped_native_frames_track_the_f32_reference() {
    let Some(gpu) = gpu() else { return };
    // vibe-land's Mac budget: 16 iterations a frame, warm-started, the settled certificate on.
    let (graph, materials, g) = house(BlastSettings::native());
    let loads: Vec<_> = (0..24)
        .map(|f| uniform(graph.node_count(), g * if f < 12 { 1.0 } else { 1.2 }))
        .collect();
    let spread = rounding_spread(&graph, &loads);
    let bound = (10.0 * spread).max(1e-4);
    let mut reference = single(&graph);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    let mut worst = 0.0f64;
    for (frame, accel) in loads.iter().enumerate() {
        let r = reference.solve(accel, false);
        let d = device.solve(accel, false);
        for (k, (r, d)) in r.iter().zip(&d).enumerate() {
            assert_eq!(
                (r.skipped, r.converged),
                (d.skipped, d.converged),
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
            // A warm frame that converges at the edge of the tolerance may stop a few iterations
            // either side, or many when one sum order lands just under it (the forces, below,
            // agree either way).
            let edge =
                r.converged && d.converged && r.residual2.max(d.residual2) >= 0.5 * r.target2;
            assert!(
                edge || r.iterations.abs_diff(d.iterations) <= 2 + r.iterations / 5,
                "frame {frame} island {k}: {r:?} vs {d:?}"
            );
        }
        let e = difference(&device.impulses(), &reference_impulses(&reference));
        worst = worst.max(e);
        assert!(e < bound, "frame {frame}: {e:.3e} (bound {bound:.3e})");
    }
    println!(
        "capped frames: worst {worst:.3e} from the f32 reference (one-ulp spread {spread:.3e})"
    );
}

#[test]
fn a_free_pair_and_a_zero_load_match_the_reference() {
    let Some(gpu) = gpu() else { return };
    let node = |x: f64, m: f64| BlastNode {
        position: Vec3::new(x, 0.0, 0.0),
        mass: m,
        inertia: 0.0,
        volume: 1.0,
    };
    let nodes = vec![node(0.0, 3.0), node(1.0, 5.0), node(2.0, 2.0)];
    let bond = |a: usize, b: usize| BlastBond {
        nodes: [a, b],
        centroid: (nodes[a].position + nodes[b].position) * 0.5,
        normal: Vec3::X,
        area: 0.5,
        modulus: 0.0,
        material: 0,
    };
    let bonds = vec![bond(0, 1), bond(1, 2)];
    let graph = BlastGraph::new(
        nodes.clone(),
        bonds,
        BlastSettings {
            max_iterations: 200,
            tolerance: 1e-6,
            ..BlastSettings::native()
        },
    )
    .unwrap();
    // Pulled apart at both ends: tension through both bonds, nothing into the rigid modes.
    let f = 60.0;
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(-f / 3.0, 0.0, 0.0),
        },
        AngLin::ZERO,
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(f / 2.0, 0.0, 0.0),
        },
    ];
    let mut reference = BlastSolver::new(graph.clone());
    reference.solve(&accel, false);
    let mut device = GpuBlastSolver::new(&gpu, graph, vec![BlastMaterial::CONCRETE]);
    let d = device.solve(&accel, false);
    assert!(d[0].converged, "{d:?}");
    let e = difference(&device.impulses(), &reference_impulses(&reference));
    assert!(e < 1e-4, "{e:.3e}");
    assert!(
        device
            .impulses()
            .iter()
            .all(|j| (j.lin.x.abs() - f).abs() < 1e-3 * f),
        "{:?}",
        device.impulses()
    );
    // A zero load: zero forces at once.
    let z = device.solve(&uniform(3, Vec3::ZERO), false);
    assert!(z[0].converged && z[0].iterations == 0, "{z:?}");
    assert!(device.impulses().iter().all(|j| j.norm2() == 0.0));
}

#[test]
fn the_settled_certificate_skips_as_the_reference_does() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings {
        max_iterations: 200,
        ..BlastSettings::native()
    });
    let accel = uniform(graph.node_count(), g);
    let mut reference = single(&graph);
    let mut device = GpuBlastSolver::new(&gpu, graph, materials);
    for frame in 0..6 {
        let r = reference.solve(&accel, false);
        let d = device.solve(&accel, false);
        assert_eq!(
            (r[0].skipped, r[0].converged, r[0].iterations == 0),
            (d[0].skipped, d[0].converged, d[0].iterations == 0),
            "frame {frame}: {r:?} vs {d:?}"
        );
    }
    assert!(device.reports()[0].skipped, "settled by frame 6");
}

#[test]
fn a_native_rerun_is_bit_identical() {
    let Some(gpu) = gpu() else { return };
    let (graph, materials, g) = house(BlastSettings::native());
    let accel = uniform(graph.node_count(), g);
    let run = || {
        let mut s = GpuBlastSolver::new(&gpu, graph.clone(), materials.clone());
        for _ in 0..3 {
            s.solve(&accel, false);
        }
        s.solution()
    };
    let (a, b) = (run(), run());
    let bits = |v: &[AngLin]| {
        v.iter()
            .flat_map(|x| x.ang.to_array().into_iter().chain(x.lin.to_array()))
            .map(f64::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&a), bits(&b));
}
