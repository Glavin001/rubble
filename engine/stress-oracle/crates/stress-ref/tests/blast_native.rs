//! The resident (native) solve of PhysX's GPU destruction stage (`SolverKind::Native` in
//! `stress_ref::blast_bond`) against physical invariants and against the host path's CGLS:
//! closed-form forces, node balance, free bodies, convergence to the same least-compliance
//! answer in far fewer iterations, the zero-load retirement, and the settled certificate.

use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

const G: f64 = 9.81;

#[test]
fn the_force_tolerance_stops_early_within_its_force_error() {
    // vibe-land's VIBE_NATIVE_STRESS_FORCE_TOLERANCE=1e-3 (PhysX forceTolerance): in f32 the
    // house's residual test has a floor that keeps a capped solve running every frame; the
    // force test stops it, and the forces stay close to the converged ones.
    let (graph, g) = house(BlastSettings {
        force_tolerance: 1e-3,
        ..native(16, 1e-3)
    });
    let accel = uniform(graph.node_count(), g);
    let (exact_graph, _) = house(native(20_000, 1e-9));
    let mut exact = BlastSolver::new(exact_graph);
    exact.solve(&accel, false);
    let mut s = BlastSolver::new(graph);
    s.device_precision = true;
    let mut last = Vec::new();
    for _ in 0..30 {
        last = s.solve(&accel, false);
    }
    let e = difference(&impulses(&s), &impulses(&exact));
    println!("force tolerance: {last:?}; {e:.2e} from the converged forces");
    assert!(last[0].converged && last[0].iterations <= 2, "{last:?}");
    assert!(e < 1e-2, "{e:.2e}");
}

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

/// A section whose radii of gyration are all `r`, on an axis in the bond's plane.
fn round_section(b: &BlastBond, r: f64) -> BondSection {
    let n = b.normal;
    let axis = n.cross(if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y }).normalized();
    BondSection {
        axis,
        gyration0: r,
        gyration1: r,
        polar_gyration: r,
        ..Default::default()
    }
}

/// A two-storey frame of 2 x 2 columns on supports, bonded to its neighbours at their midpoints.
fn frame(settings: BlastSettings) -> BlastGraph {
    let mut nodes = vec![];
    for z in 0..3 {
        for x in 0..2 {
            for y in 0..2 {
                let mass = if z == 0 { 0.0 } else { 1.0 + x as f64 };
                nodes.push(node(x as f64, y as f64, z as f64, mass));
            }
        }
    }
    let mut bonds = vec![];
    for a in 0..nodes.len() {
        for b in a + 1..nodes.len() {
            let near = (nodes[a].position - nodes[b].position).norm() < 1.01;
            if near && (nodes[a].mass > 0.0 || nodes[b].mass > 0.0) {
                bonds.push(bond(a, b, &nodes, 0.1));
            }
        }
    }
    BlastGraph::new(nodes, bonds, settings).unwrap()
}

#[test]
fn rotational_stiffness_of_unit_radius_is_the_uniform_solve() {
    // A = sum (r/L) e e' over an orthonormal frame is I when every radius is L: the same
    // operator as without rotational stiffness (the offsets move to the centroids, here the
    // midpoints), so the same converged forces.
    let plain = frame(native(20_000, 1e-10));
    let mut rotated = plain.clone();
    let sections: Vec<BondSection> =
        rotated.bonds.iter().map(|b| round_section(b, plain.length_scale)).collect();
    rotated.set_rotational_stiffness(Some(&sections)).unwrap();
    let load = AngLin {
        ang: Vec3::new(0.3, 0.0, 0.1),
        lin: Vec3::new(1.0, 0.5, -G),
    };
    let accel = vec![load; plain.node_count()];
    let (mut a, mut b) = (BlastSolver::new(plain), BlastSolver::new(rotated));
    assert!(a.solve(&accel, false)[0].converged && b.solve(&accel, false)[0].converged);
    let e = difference(&impulses(&b), &impulses(&a));
    assert!(e < 1e-7, "{e:.2e}");
}

#[test]
fn parallel_bonds_share_a_moment_by_their_rotational_stiffness() {
    // Two bonds from a support to one chunk at the same centroid, one with twice the radii of
    // gyration: four times the rotational stiffness (W = (r/L)^2), so four times the moment.
    let nodes = vec![node(0.0, 0.0, 0.0, 0.0), node(1.0, 0.0, 0.0, 2.0)];
    let bonds = vec![bond(0, 1, &nodes, 0.1), bond(0, 1, &nodes, 0.1)];
    let mut graph = BlastGraph::new(nodes, bonds, native(20_000, 1e-10)).unwrap();
    let sections = vec![round_section(&graph.bonds[0], 0.1), round_section(&graph.bonds[1], 0.2)];
    graph.set_rotational_stiffness(Some(&sections)).unwrap();
    let load = AngLin {
        ang: Vec3::new(0.7, -0.4, 0.2),
        lin: Vec3::new(0.0, 0.0, -G),
    };
    let accel = vec![AngLin::ZERO, load];
    let mut s = BlastSolver::new(graph);
    assert!(s.solve(&accel, false)[0].converged);
    let (m0, m1) = (s.impulse(0).ang, s.impulse(1).ang);
    for k in 0..3 {
        assert!((m1[k] - 4.0 * m0[k]).abs() <= 1e-6 * m1[k].abs().max(1e-9), "{m0:?} {m1:?}");
    }
    // The linear forces still balance the weight, equally (the same compliance).
    let (f0, f1) = (s.impulse(0).lin, s.impulse(1).lin);
    assert!((f0 - f1).norm() <= 1e-6 * f0.norm(), "{f0:?} {f1:?}");
}

#[test]
fn krylov_carry_reaches_the_answer_restarting_stalls_short_of() {
    // A capped, warm-started solve at rest restarts PCG every frame, losing the Krylov space the
    // slow modes need; the carry continues it (BLAST_STRESS_CARRY_KRYLOV, on by default with the
    // rotational stiffness). Every step is still an exact line search on the same energy: the
    // carried frames approach the converged forces, and the carry changes no converged answer.
    let settings = |carry: bool| BlastSettings {
        carry_krylov: Some(carry),
        ..native(4, 1e-6)
    };
    let stiff = |s: BlastSettings| {
        let mut g = frame(s);
        g.set_rotational_stiffness(None).unwrap();
        g
    };
    let load = AngLin {
        ang: Vec3::ZERO,
        lin: Vec3::new(2.0, 0.0, -G),
    };
    let accel = vec![load; stiff(settings(true)).node_count()];
    let mut exact = BlastSolver::new(stiff(native(20_000, 1e-10)));
    assert!(exact.solve(&accel, false)[0].converged);
    let error = |carry: bool, frames: usize| {
        let mut s = BlastSolver::new(stiff(settings(carry)));
        let mut converged_at = None;
        for f in 0..frames {
            if s.solve(&accel, false)[0].converged && converged_at.is_none() {
                converged_at = Some(f);
            }
        }
        (difference(&impulses(&s), &impulses(&exact)), converged_at)
    };
    let (carried, carried_at) = error(true, 40);
    let (restarted, restarted_at) = error(false, 40);
    println!("carry: {carried:.2e} (converged at {carried_at:?}); restarted: {restarted:.2e} ({restarted_at:?})");
    assert!(carried_at.is_some(), "the carried solve converges");
    assert!(carried < 1e-4, "{carried:.2e}");
    assert!(carried_at < restarted_at.or(Some(usize::MAX)) || carried < restarted, "{carried_at:?} {restarted_at:?}");
}
