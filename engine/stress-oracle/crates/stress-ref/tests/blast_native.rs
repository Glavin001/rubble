//! The resident (native) solve of PhysX's GPU destruction stage (`SolverKind::Native` in
//! `stress_ref::blast_bond`) against physical invariants and against the host path's CGLS:
//! closed-form forces, node balance, free bodies, convergence to the same least-compliance
//! answer in far fewer iterations, the zero-load retirement, and the settled certificate.

use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

const G: f64 = 9.81;

fn native(max_iterations: u32, tolerance: f64) -> BlastSettings {
    BlastSettings {
        max_iterations,
        tolerance,
        ..BlastSettings::native()
    }
}

fn node(x: f64, y: f64, z: f64, mass: f64) -> BlastNode {
    BlastNode {
        position: Vec3::new(x, y, z),
        mass,
        inertia: 0.0,
        volume: 1.0,
    }
}

fn bond(a: usize, b: usize, nodes: &[BlastNode], area: f64) -> BlastBond {
    let (pa, pb) = (nodes[a].position, nodes[b].position);
    BlastBond {
        nodes: [a, b],
        centroid: (pa + pb) * 0.5,
        normal: (pb - pa).normalized(),
        area,
        modulus: 0.0,
        material: 0,
    }
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

fn impulses(s: &BlastSolver) -> Vec<AngLin> {
    (0..s.graph.bond_count()).map(|j| s.impulse(j)).collect()
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

#[test]
fn a_column_of_unequal_chunks_carries_each_ones_own_weight() {
    // Native weighting: each chunk's own mass loads the column, not the mean.
    let masses = [0.0, 4.0, 9.0, 1.0, 16.0];
    let nodes: Vec<_> = masses
        .iter()
        .enumerate()
        .map(|(i, &m)| node(0.0, i as f64, 0.0, m))
        .collect();
    let bonds: Vec<_> = (0..4).map(|i| bond(i, i + 1, &nodes, 1.0)).collect();
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, native(500, 1e-9)).unwrap());
    let r = solver.solve(&uniform(nodes.len(), Vec3::new(0.0, -G, 0.0)), false);
    assert!(r[0].converged, "{r:?}");
    for j in 0..4 {
        let above: f64 = masses[j + 1..].iter().sum::<f64>() * G;
        let f = solver.impulse(j).lin.y;
        assert!((f + above).abs() < 1e-6 * above, "bond {j}: {f} vs {above}");
    }
}

#[test]
fn a_free_pair_pulled_apart_carries_the_pull() {
    // No support, no gravity: forces +-F on two chunks; the bond carries F in tension and the
    // rigid modes carry nothing.
    let nodes = vec![node(0.0, 0.0, 0.0, 3.0), node(1.0, 0.0, 0.0, 5.0)];
    let bonds = vec![bond(0, 1, &nodes, 0.5)];
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, native(200, 1e-9)).unwrap());
    let f = 100.0;
    let accel = vec![
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(-f / 3.0, 0.0, 0.0),
        },
        AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(f / 5.0, 0.0, 0.0),
        },
    ];
    let r = solver.solve(&accel, false);
    assert!(r[0].converged, "{r:?}");
    let j = solver.impulse(0);
    assert!((j.lin.x.abs() - f).abs() < 1e-6 * f, "{j:?}");
    assert!(
        solver.stress(0).tension > 0.0 && solver.stress(0).compression == 0.0,
        "{:?}",
        solver.stress(0)
    );

    // Falling together, the same pair carries nothing.
    let mut solver = BlastSolver::new(solver.graph.clone());
    let r = solver.solve(&uniform(2, Vec3::new(0.0, -G, 0.0)), false);
    assert!(r[0].converged && r[0].iterations == 0, "{r:?}");
    assert!(solver.impulse(0).norm2() < 1e-20);
}

fn house(settings: BlastSettings) -> (BlastGraph, Vec3) {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/house-1story.json");
    let options = stress_ref::scene_pack::ImportOptions {
        authored_strengths: true,
        authored_fatal: true,
        pack_masses: true,
        pack_modulus: true,
        blast_authoring: true,
    };
    let scene = stress_ref::scene_pack::import_with(&path, "house", "house", options).unwrap();
    let (graph, _) = BlastAuthoring::graph(&scene, settings).unwrap();
    (graph, Vec3::from_array(scene.gravity))
}

#[test]
fn native_converges_to_the_host_paths_answer_in_far_fewer_iterations() {
    // Both solve the same least-compliance problem (native weighting for both, so the loads
    // are the same physical forces): converged, the forces agree; the preconditioned solve
    // gets there in a fraction of the iterations.
    let tol = 1e-7;
    let (graph, g) = house(native(20_000, tol));
    let accel = uniform(graph.node_count(), g);
    let mut pcg = BlastSolver::new(graph.clone());
    let rp = pcg.solve(&accel, false);
    let host = BlastSettings {
        solver: SolverKind::HostCgls,
        ..graph.settings
    };
    let mut cg =
        BlastSolver::new(BlastGraph::new(graph.nodes.clone(), graph.bonds.clone(), host).unwrap());
    let rc = cg.solve(&accel, false);
    assert!(rp.iter().chain(&rc).all(|r| r.converged), "{rp:?} {rc:?}");
    let e = difference(&impulses(&pcg), &impulses(&cg));
    let (ip, ic) = (
        rp.iter().map(|r| r.iterations).sum::<u32>(),
        rc.iter().map(|r| r.iterations).sum::<u32>(),
    );
    println!("native {ip} iterations, host {ic}; forces differ by {e:.2e}");
    assert!(e < 1e-4, "{e:.2e}");
    assert!(2 * ip <= ic, "native {ip} vs host {ic}");
}

#[test]
fn capped_native_frames_reach_the_answer_the_host_path_stalls_short_of() {
    // vibe-land's budgets: native 16 iterations a frame, host 32. Over 30 frames of gravity the
    // native solve settles on the converged forces; the host path is still far off.
    let (graph, g) = house(native(16, 1e-3));
    let accel = uniform(graph.node_count(), g);
    let (exact_graph, _) = house(native(20_000, 1e-9));
    let mut exact = BlastSolver::new(exact_graph);
    exact.solve(&accel, false);
    let mut pcg = BlastSolver::new(graph.clone());
    let host = BlastSettings {
        solver: SolverKind::HostCgls,
        max_iterations: 32,
        ..graph.settings
    };
    let mut cg =
        BlastSolver::new(BlastGraph::new(graph.nodes.clone(), graph.bonds.clone(), host).unwrap());
    for _ in 0..30 {
        pcg.solve(&accel, false);
        cg.solve(&accel, false);
    }
    let (ep, ec) = (
        difference(&impulses(&pcg), &impulses(&exact)),
        difference(&impulses(&cg), &impulses(&exact)),
    );
    println!("after 30 frames: native {ep:.2e}, host {ec:.2e} from the converged forces");
    assert!(
        ep < 1e-2 && ep < 0.25 * ec,
        "native {ep:.2e}, host {ec:.2e}"
    );
}

#[test]
fn zero_load_retires_with_zero_forces() {
    let nodes = vec![
        node(0.0, 0.0, 0.0, 0.0),
        node(0.0, 1.0, 0.0, 2.0),
        node(0.0, 2.0, 0.0, 2.0),
    ];
    let bonds = vec![bond(0, 1, &nodes, 1.0), bond(1, 2, &nodes, 1.0)];
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, native(16, 1e-3)).unwrap());
    solver.x[0] = AngLin {
        ang: Vec3::ZERO,
        lin: Vec3::new(0.0, 5.0, 0.0),
    };
    let r = solver.solve(&uniform(3, Vec3::ZERO), false);
    assert!(r[0].converged && r[0].iterations == 0, "{r:?}");
    assert!((0..2).all(|j| solver.impulse(j).norm2() == 0.0));
    // A loaded anchor is never 'converged' at iteration 0 with zero forces (the relative
    // residual must hold too: StressComponentIteration.cuh's iteration-0 regression).
    let r = solver.solve(&uniform(3, Vec3::new(0.0, -G, 0.0)), false);
    assert!(
        r[0].iterations > 0 && solver.impulse(0).lin.y < 0.0,
        "{r:?}"
    );
}

#[test]
fn the_settled_certificate_skips_only_a_verified_island() {
    // Converge; the next identical frame converges at iteration 0 on the rebuilt residual
    // (the certificate); only the frame after that is skipped, and its forces are kept.
    let (graph, g) = house(native(200, 1e-3));
    let accel = uniform(graph.node_count(), g);
    let mut solver = BlastSolver::new(graph);
    let mut reports = solver.solve(&accel, false);
    let mut guard = 0;
    while !(reports[0].converged && reports[0].iterations == 0) {
        assert!(
            !reports[0].skipped,
            "skipped before it was verified: {reports:?}"
        );
        reports = solver.solve(&accel, false);
        guard += 1;
        assert!(guard < 50, "{reports:?}");
    }
    let before = impulses(&solver);
    let skipped = solver.solve(&accel, false);
    assert!(
        skipped[0].skipped && skipped[0].iterations == 0,
        "{skipped:?}"
    );
    assert_eq!(before, impulses(&solver));
    // A changed load is solved again.
    let heavier = uniform(solver.graph.node_count(), g * 1.5);
    assert!(!solver.solve(&heavier, false)[0].skipped);
}
