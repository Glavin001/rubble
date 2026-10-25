//! Blast's bond-impulse solver (`stress_ref::blast_bond`) against physical invariants:
//! statically determinate structures have closed-form bond forces, every converged solve
//! balances each node, uniform acceleration of a free body loads no bond, and islands solve
//! independently.

use stress_ref::blast_bond::*;
use stress_ref::math::Vec3;

const G: f64 = 9.81;

fn converged() -> BlastSettings {
    BlastSettings {
        max_iterations: 100_000,
        tolerance: 1e-10,
        ..Default::default()
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

fn gravity(nodes: &[BlastNode]) -> Vec<AngLin> {
    nodes
        .iter()
        .map(|_| AngLin {
            ang: Vec3::ZERO,
            lin: Vec3::new(0.0, -G, 0.0),
        })
        .collect()
}

/// A column of `n` unit chunks standing on a support.
fn column(n: usize) -> (Vec<BlastNode>, Vec<BlastBond>) {
    let mut nodes = vec![node(0.0, 0.0, 0.0, 0.0)];
    for i in 1..=n {
        nodes.push(node(0.0, i as f64, 0.0, 10.0));
    }
    let bonds = (0..n).map(|i| bond(i, i + 1, &nodes, 1.0)).collect();
    (nodes, bonds)
}

/// Force balance at every dynamic node: the bonds' forces and moments cancel the node's load
/// (`M a` and `M L^2 alpha` under equalised masses). Returns the largest imbalance relative to
/// the largest load.
fn imbalance(solver: &BlastSolver, accel: &[AngLin]) -> f64 {
    let g = &solver.graph;
    let load_scale = match g.settings.mass_weighting {
        MassWeighting::Equalized => vec![g.mass_scale; g.node_count()],
        MassWeighting::Native => g.nodes.iter().map(|n| n.mass).collect(),
    };
    let mut worst = 0.0f64;
    let mut largest = 0.0f64;
    for i in 0..g.node_count() {
        if g.weight[i].is_static() || solver.islands.node[i] == NO_ISLAND {
            continue;
        }
        let mut force = accel[i].lin * load_scale[i];
        let mut moment = Vec3::ZERO;
        for j in 0..g.bond_count() {
            if !(solver.health[j] > 0.0) {
                continue;
            }
            let jb = solver.impulse(j);
            let b = g.bonds[j];
            if b.nodes[0] == i {
                force += jb.lin;
                moment += jb.ang - (g.offset0[j] * g.length_scale).cross(jb.lin);
            } else if b.nodes[1] == i {
                force -= jb.lin;
                moment -= jb.ang - (g.offset1[j] * g.length_scale).cross(jb.lin);
            }
        }
        largest = largest.max((accel[i].lin * load_scale[i]).norm());
        worst = worst.max(force.norm()).max(moment.norm() / g.length_scale);
    }
    worst / largest.max(1e-300)
}

#[test]
fn a_column_carries_the_weight_above_each_bond() {
    let (nodes, bonds) = column(5);
    let mut solver = BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, converged()).unwrap());
    let accel = gravity(&nodes);
    let reports = solver.solve(&accel, false);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].converged, "{reports:?}");
    for j in 0..5 {
        let above = (5 - j) as f64 * 10.0;
        let jb = solver.impulse(j);
        // Node 0 is below: the bond pushes it down and node 1 up, so jb.lin is -weight in y.
        assert!(
            (jb.lin.y + above * G).abs() < 1e-6 * above * G,
            "bond {j}: {jb:?}"
        );
        assert!(
            jb.lin.x.abs() + jb.lin.z.abs() < 1e-9 && jb.ang.norm() < 1e-6,
            "bond {j}: {jb:?}"
        );
        let s = solver.stress(j);
        assert!(
            (s.compression - above * G).abs() < 1e-6 * above * G && s.tension == 0.0,
            "bond {j}: {s:?}"
        );
    }
    assert!(imbalance(&solver, &accel) < 1e-9);
}

#[test]
fn a_cantilever_carries_shear_and_moment() {
    // A horizontal chain off a wall: statically determinate, so bond j carries the weight of
    // the chunks beyond it and their moment about the bond's reference point.
    let n = 4;
    let mut nodes = vec![node(0.0, 0.0, 0.0, 0.0)];
    for i in 1..=n {
        nodes.push(node(i as f64, 0.0, 0.0, 5.0));
    }
    let bonds: Vec<_> = (0..n).map(|i| bond(i, i + 1, &nodes, 0.25)).collect();
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds.clone(), converged()).unwrap());
    let accel = gravity(&nodes);
    assert!(solver.solve(&accel, false)[0].converged);
    for j in 0..n {
        let beyond: Vec<usize> = (j + 1..=n).collect();
        let weight = beyond.len() as f64 * 5.0 * G;
        // Reference point: the centroid against the support, the midpoint between two chunks.
        let point = if j == 0 {
            bonds[0].centroid
        } else {
            (nodes[j].position + nodes[j + 1].position) * 0.5
        };
        let moment: f64 = beyond
            .iter()
            .map(|&i| (nodes[i].position.x - point.x) * 5.0 * G)
            .sum();
        let jb = solver.impulse(j);
        assert!(
            (jb.lin.y.abs() - weight).abs() < 1e-6 * weight,
            "bond {j}: {jb:?} vs {weight}"
        );
        assert!(
            (jb.ang.z.abs() - moment).abs() < 1e-6 * moment.max(1.0),
            "bond {j}: {jb:?} vs {moment}"
        );
        // Bending dominates a slender cantilever: the root bond is both in tension and
        // compression on its two fibres, and its shear is the carried weight over the area.
        let s = solver.stress(j);
        assert!(
            (s.shear - weight / 0.25).abs() < 1e-6 * weight / 0.25,
            "{s:?}"
        );
        assert!(s.tension > 0.0 && s.compression > 0.0, "{s:?}");
    }
    assert!(imbalance(&solver, &accel) < 1e-9);
}

#[test]
fn a_redundant_frame_balances_every_node() {
    // A portal: two columns on two supports joined by a beam, plus a diagonal. Statically
    // indeterminate, so only balance (and least compliance) define the answer.
    let nodes = vec![
        node(0.0, 0.0, 0.0, 0.0),
        node(4.0, 0.0, 0.0, 0.0),
        node(0.0, 1.0, 0.0, 8.0),
        node(4.0, 1.0, 0.0, 12.0),
        node(0.0, 2.0, 0.0, 8.0),
        node(2.0, 2.0, 0.0, 6.0),
        node(4.0, 2.0, 0.0, 9.0),
    ];
    let pairs = [(0, 2), (2, 4), (1, 3), (3, 6), (4, 5), (5, 6), (2, 5)];
    let bonds: Vec<_> = pairs
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| BlastBond {
            modulus: 10e9 + k as f64 * 5e9,
            ..bond(a, b, &nodes, 0.2 + 0.1 * k as f64)
        })
        .collect();
    for weighting in [MassWeighting::Equalized, MassWeighting::Native] {
        let settings = BlastSettings {
            mass_weighting: weighting,
            ..converged()
        };
        let mut solver =
            BlastSolver::new(BlastGraph::new(nodes.clone(), bonds.clone(), settings).unwrap());
        let mut accel = gravity(&nodes);
        accel[5].lin.x += 3.0; // a side load on the beam
        let reports = solver.solve(&accel, false);
        assert!(reports[0].converged, "{weighting:?}: {reports:?}");
        let e = imbalance(&solver, &accel);
        assert!(e < 1e-8, "{weighting:?}: imbalance {e}");
    }
}

#[test]
fn least_compliance_is_the_minimum_over_balanced_force_sets() {
    // Two parallel bonds between a support and one chunk share its weight in proportion to
    // their compliance weights squared: the minimiser of sum |J_j|^2 / s_j^2 under J_1 + J_2 = W.
    let nodes = vec![node(0.0, 0.0, 0.0, 0.0), node(0.0, 1.0, 0.0, 10.0)];
    let mut b0 = bond(0, 1, &nodes, 1.0);
    let mut b1 = bond(0, 1, &nodes, 4.0);
    b0.centroid = Vec3::new(0.0, 0.5, 0.0);
    b1.centroid = Vec3::new(0.0, 0.5, 0.0);
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), vec![b0, b1], converged()).unwrap());
    assert!(solver.solve(&gravity(&nodes), false)[0].converged);
    let (s0, s1) = (solver.graph.col_scale[0], solver.graph.col_scale[1]);
    let (j0, j1) = (solver.impulse(0).lin.y, solver.impulse(1).lin.y);
    let w = 10.0 * G;
    let share0 = s0 * s0 / (s0 * s0 + s1 * s1);
    assert!(
        (j0 + share0 * w).abs() < 1e-6 * w && (j1 + (1.0 - share0) * w).abs() < 1e-6 * w,
        "{j0} {j1} {s0} {s1}"
    );
    // The area-4 bond is twice as compliant-weighted (sqrt of the area ratio): 4/5 of the load.
    assert!((share0 - 0.2).abs() < 1e-12);
}

#[test]
fn zero_load_converges_at_once_with_no_force() {
    let (nodes, bonds) = column(3);
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, BlastSettings::default()).unwrap());
    let reports = solver.solve(&vec![AngLin::ZERO; nodes.len()], false);
    assert!(
        reports[0].converged && reports[0].iterations == 0,
        "{reports:?}"
    );
    assert!((0..3).all(|j| solver.impulse(j) == AngLin::ZERO));
}

#[test]
fn uniform_acceleration_of_a_free_body_loads_no_bond() {
    // No support: the pieces fall together, the rigid mode carries the whole load.
    let nodes = vec![
        node(0.0, 0.0, 0.0, 3.0),
        node(1.0, 0.0, 0.0, 7.0),
        node(1.0, 1.0, 0.0, 2.0),
    ];
    let bonds = vec![
        bond(0, 1, &nodes, 1.0),
        bond(1, 2, &nodes, 1.0),
        bond(0, 2, &nodes, 1.0),
    ];
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, BlastSettings::default()).unwrap());
    let reports = solver.solve(&gravity(&nodes), false);
    assert!(
        reports[0].converged && reports[0].iterations == 0,
        "{reports:?}"
    );
    assert!((0..3).all(|j| solver.impulse(j).norm2() == 0.0));
}

#[test]
fn islands_are_cut_at_supports_and_solve_on_their_own() {
    // Two towers on one shared ground node: two islands. A capped solve stops each on its own
    // tolerance, and the quiet tower converges at iteration 0.
    let mut nodes = vec![node(0.0, 0.0, 0.0, 0.0)];
    for i in 1..=3 {
        nodes.push(node(0.0, i as f64, 0.0, 10.0));
    }
    for i in 1..=3 {
        nodes.push(node(5.0, i as f64, 0.0, 10.0));
    }
    let mut bonds = vec![
        bond(0, 1, &nodes, 1.0),
        bond(1, 2, &nodes, 1.0),
        bond(2, 3, &nodes, 1.0),
    ];
    bonds.push(BlastBond {
        centroid: Vec3::new(5.0, 0.5, 0.0),
        ..bond(0, 4, &nodes, 1.0)
    });
    bonds.push(bond(4, 5, &nodes, 1.0));
    bonds.push(bond(5, 6, &nodes, 1.0));
    let mut solver =
        BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, BlastSettings::default()).unwrap());
    assert_eq!(solver.islands.count, 2);
    assert_eq!(solver.islands.node[0], NO_ISLAND);
    let mut accel = gravity(&nodes);
    for a in &mut accel[4..] {
        *a = AngLin::ZERO;
    }
    let reports = solver.solve(&accel, false);
    assert!(
        reports[1].converged && reports[1].iterations == 0,
        "{reports:?}"
    );
    assert!(reports[0].iterations > 0);
    assert!((3..6).all(|j| solver.impulse(j) == AngLin::ZERO));

    // Breaking the middle of the first tower splits it: the top chunk becomes a free island.
    solver.set_health(1, 0.0);
    assert_eq!(solver.islands.count, 3);
}

#[test]
fn warm_start_carries_a_capped_solve_toward_the_answer() {
    // 2 iterations a frame: each frame continues from the last, and the residual falls.
    let (nodes, bonds) = column(12);
    let settings = BlastSettings {
        max_iterations: 2,
        ..Default::default()
    };
    let mut solver = BlastSolver::new(BlastGraph::new(nodes.clone(), bonds, settings).unwrap());
    let accel = gravity(&nodes);
    // The direction restarts every frame (Blast's warmth 1), so a capped frame is close to
    // steepest descent: the residual falls every frame, slowly (Blast needs tens to hundreds of
    // capped frames on a tall stack).
    let mut residuals = Vec::new();
    for _ in 0..400 {
        let r = solver.solve(&accel, false);
        residuals.push(r[0].residual2);
        if r[0].converged {
            break;
        }
    }
    assert!(residuals.windows(2).all(|w| w[1] <= w[0]), "{residuals:?}");
    assert!(
        residuals.last().unwrap() < &(residuals[0] * 1e-2),
        "{residuals:?}"
    );
    let base = solver.impulse(0).lin.y;
    assert!(
        (base + 12.0 * 10.0 * G).abs() < 0.05 * 12.0 * 10.0 * G,
        "{base}"
    );
}

#[test]
fn a_settled_island_is_skipped_and_keeps_its_forces() {
    let (nodes, bonds) = column(4);
    let mut solver = BlastSolver::new(
        BlastGraph::new(
            nodes.clone(),
            bonds,
            BlastSettings {
                max_iterations: 64,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let accel = gravity(&nodes);
    let mut reports = solver.solve(&accel, true);
    while !reports[0].converged {
        reports = solver.solve(&accel, true);
    }
    let before: Vec<_> = (0..4).map(|j| solver.impulse(j)).collect();
    let again = solver.solve(&accel, true);
    assert!(again[0].skipped && again[0].iterations == 0, "{again:?}");
    assert_eq!(
        before,
        (0..4).map(|j| solver.impulse(j)).collect::<Vec<_>>()
    );
}

#[test]
fn damage_follows_blasts_excess_law() {
    let m = BlastMaterial::CONCRETE;
    let at = |c: f64, t: f64, s: f64| BondStress {
        compression: c,
        tension: t,
        shear: s,
        ..Default::default()
    };
    // Below every elastic limit: nothing.
    assert!(!bond_damage(at(1e6, 1e5, 1e5), 1.0, 1.0, &m, 1.0 / 60.0, 2.0).command);
    // Fatal tension: the whole remaining area at once.
    let d = bond_damage(at(0.0, 3e6, 0.0), 0.7, 1.0, &m, 1.0 / 60.0, 2.0);
    assert!(d.command && d.damage == 0.7 && (d.multiplier - 1.0).abs() < 1e-12);
    // Halfway into tension's band plus a quarter into shear's: 0.75, at 2/s over 1/60 s.
    let d = bond_damage(at(0.0, 2.1e6, 2.2e6), 1.0, 1.0, &m, 1.0 / 60.0, 2.0);
    assert!(
        (d.multiplier - 0.75).abs() < 1e-12 && (d.damage - 0.75 * 2.0 / 60.0).abs() < 1e-12,
        "{d:?}"
    );
    // Compression and tension take the larger excess, they do not add.
    let d = bond_damage(at(21e6, 2.1e6, 0.0), 1.0, 1.0, &m, 0.0, 2.0);
    assert!((d.multiplier - 0.5).abs() < 1e-12, "{d:?}");
}

#[test]
fn section_bending_and_torsion_use_the_patch() {
    // A pure moment M about an axis in the bond plane on a 0.04 m^2 patch: bend = M/A, and the
    // section gain 6/sqrt(A) = 30 is capped at 3; torsion likewise at 4.81/sqrt(A), capped.
    let n = Vec3::Y;
    let s = bond_stress(
        AngLin {
            ang: Vec3::new(100.0, 0.0, 0.0),
            lin: Vec3::ZERO,
        },
        n,
        0.04,
        1.0,
        3.0,
        true,
    );
    assert!(
        (s.bend - 100.0 / 0.04 * 3.0).abs() < 1e-9 && s.normal == 0.0,
        "{s:?}"
    );
    assert!(s.tension == s.bend && s.compression == s.bend);
    let s = bond_stress(
        AngLin {
            ang: Vec3::new(0.0, 50.0, 0.0),
            lin: Vec3::ZERO,
        },
        n,
        4.0,
        1.0,
        3.0,
        true,
    );
    assert!((s.shear - 50.0 / 4.0 * 4.81 / 2.0).abs() < 1e-9, "{s:?}");
    // Legacy fold: bend goes into the axial stress with its sign, over the node distance.
    let s = bond_stress(
        AngLin {
            ang: Vec3::new(10.0, 0.0, 0.0),
            lin: Vec3::new(0.0, -20.0, 0.0),
        },
        n,
        1.0,
        2.0,
        0.0,
        true,
    );
    assert!(
        (s.normal + 20.0 + 10.0).abs() < 1e-12 && (s.compression - 30.0).abs() < 1e-12,
        "{s:?}"
    );
}
