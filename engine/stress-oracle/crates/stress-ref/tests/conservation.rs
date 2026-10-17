//! Testable properties: momentum conserved through fracture, fragments inherit
//! `v + w x r`, no energy gain with dissipation accounted for.

mod common;

use stress_ref::builders::*;
use stress_ref::material::Material;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::solver::ReferenceSolver;
use stress_ref::world::World;

/// A free (unsupported) concrete wall in zero gravity, hit by a fast steel sphere.
fn free_wall_impact(speed: f64, fracture: bool) -> Scene {
    let mut s = new_scene("free_wall", "free wall hit by a sphere in zero gravity");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let chunks = grid(Vec3::new(-0.5, 0.0, -0.5), [10, 2, 10], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    let mut b = body("wall", chunks, bonds);
    b.angular_velocity = [0.0, 0.0, 0.5];
    s.bodies.push(b);
    s.impactors.push(ImpactorDesc {
        name: "ball".into(),
        shape: ImpactorShape::Sphere { radius: 0.1 },
        mass: 50.0,
        position: [0.05, -0.101, 0.05],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s.sim.duration = 0.03;
    s.sim.frame_dt = 1e-3;
    s.sim.fracture = fracture;
    s.sim.gravity_prestress = false;
    s
}

/// Total linear and angular momentum (about the origin) of chunks and impactors.
fn momentum(w: &World) -> (Vec3, Vec3) {
    let mut p = Vec3::ZERO;
    let mut l = Vec3::ZERO;
    let sv = &w.solver;
    for (s, st) in sv.structures.iter().enumerate() {
        for (c, ch) in st.chunks.iter().enumerate() {
            if !sv.chunks[s][c].active {
                continue;
            }
            let (v, om) = sv.chunk_velocity(s, c);
            let x = sv.chunk_position(s, c);
            let r = sv.clusters[sv.chunks[s][c].cluster].rotation();
            p += v * ch.mass;
            l += x.cross(v) * ch.mass + (r * ch.inertia * r.transpose()) * om;
        }
    }
    for imp in &w.impactors {
        let r = imp.pose.rotation.to_mat3();
        p += imp.velocity * imp.mass;
        l += imp.pose.position.cross(imp.velocity) * imp.mass + (r * imp.inertia * r.transpose()) * imp.angular_velocity;
    }
    (p, l)
}

/// Linear momentum is conserved to round-off (all forces act in equal and opposite
/// pairs and splits preserve every chunk's velocity): `round_off_of_momentum` bounds
/// what rounding can change.
#[test]
fn linear_momentum_is_conserved_through_impact_and_fracture() {
    let scene = free_wall_impact(40.0, true);
    let mut w = World::new(&scene);
    let (p0, _) = momentum(&w);
    let mut worst: f64 = 0.0;
    for _ in 0..30 {
        w.step_frame();
        let (p, _) = momentum(&w);
        let bound = round_off_of_momentum(&w, 40.0, None);
        worst = worst.max((p - p0).norm() / bound);
        assert!((p - p0).norm() <= bound, "linear momentum drift {:?} vs {:?}: {:.2e} beyond the round-off bound {bound:.2e}", p, p0, (p - p0).norm());
    }
    println!("linear momentum: worst drift {worst:.2e} of the round-off bound");
    assert!(w.solver.broken_bond_count() > 0, "the impact should fracture the wall");
    assert!(w.solver.clusters.len() > 1, "the wall should split");
}

/// The moment the small-strain bonds fail to balance, `sum (u_a - u_b) x f` (world):
/// their forces act at the undeformed lever arms (the linear kinematics are the gradient
/// of the bond energy and invariant under infinitesimal rotations of the undeformed
/// configuration only), so internal forces turn the system by this much. The exact
/// angular-momentum target is `L - (this, integrated) = L0`.
/// The hidden state the substep's bond forces act on: per chunk its displacement and its
/// cluster's rotation, and which bonds are alive, at the start of the substep.
type Start = (Vec<Vec<(Vec3, stress_ref::math::Mat3)>>, Vec<Vec<bool>>);

fn substep_start(w: &World) -> Start {
    let sv = &w.solver;
    let chunks = sv.chunks.iter().map(|cs| cs.iter().map(|c| (c.u, sv.clusters[c.cluster].rotation())).collect()).collect();
    let alive = sv.bonds.iter().map(|bs| bs.iter().map(|b| b.alive).collect()).collect();
    (chunks, alive)
}

/// The moment the small-strain bonds fail to balance over a substep, `sum (u_a - u_b) x f`
/// (world), with the forces the substep applied and the displacements they acted at
/// (`start`): their forces act at the undeformed lever arms (the linear kinematics are
/// the gradient of the bond energy and invariant under infinitesimal rotations of the
/// undeformed configuration only), so internal forces turn the system by this much.
/// The exact angular-momentum target is `L - (this, summed) = L0`.
fn bond_moment_imbalance(w: &World, start: &Start) -> Vec3 {
    let sv = &w.solver;
    let mut t = Vec3::ZERO;
    for (s, bonds) in sv.bonds.iter().enumerate() {
        for (bi, b) in bonds.iter().enumerate() {
            if !start.1[s][bi] {
                continue;
            }
            let g = &b.geometry;
            let ((ua, ra), (ub, _)) = (start.0[s][g.a], start.0[s][g.b]);
            let (fa, ma, fb, mb) = g.chunk_loads(&b.force);
            let (xa, xb) = (sv.structures[s].chunks[g.a].center + ua, sv.structures[s].chunks[g.b].center + ub);
            t += ra * (xa.cross(fa) + ma + xb.cross(fb) + mb);
        }
    }
    t
}

/// Angular momentum (no external torque) changes by exactly the small-strain bonds'
/// moment imbalance (`bond_moment_imbalance`, summed substep by substep with the forces
/// and displacements of each), up to integration error that vanishes with the substep
/// (`assert_converges_to_zero`): nothing else in the model (contacts and their crushing,
/// the floating frame, the rigid update) may create or destroy angular momentum. The
/// 40 m/s impact without fracture: with fracture every substep breaks a different set of
/// bonds (the runs are chaotic), so the error of each is not one sequence in the substep;
/// splits are held exactly by `a_split_of_a_deformed_body_keeps_its_motion`. The
/// imbalance itself (about 5e-4 of `|L0|`) is the open model item: co-rotational bond
/// kinematics remove it (DECISIONS.md 36).
#[test]
fn angular_momentum_changes_only_by_the_bond_imbalance() {
    let runs: Vec<(f64, u64, f64)> = [0.5, 0.25, 0.125, 0.0625]
        .iter()
        .map(|&k| {
            let mut s = free_wall_impact(40.0, false);
            // The ledger is the small-strain bonds' imbalance (co-rotational bonds have none).
            s.sim.methods.corotational_bonds = false;
            s.sim.courant_safety = k;
            let dt = World::new(&s).substep_dt();
            s.sim.max_substep = Some(dt);
            s.sim.frame_dt = dt;
            let mut w = World::new(&s);
            let (_, l0) = momentum(&w);
            let (mut ledger, mut worst, mut imbalance) = (Vec3::ZERO, 0.0f64, 0.0f64);
            let total = (0.03 / dt).round() as u64;
            while w.solver.substeps < total {
                let start = substep_start(&w);
                w.step_frame();
                let t = bond_moment_imbalance(&w, &start);
                ledger += t * dt;
                let (_, l) = momentum(&w);
                worst = worst.max((l - l0 - ledger).norm() / l0.norm());
                imbalance = imbalance.max(ledger.norm() / l0.norm());
            }
            (worst, w.solver.substeps, imbalance)
        })
        .collect();
    println!("bond moment imbalance, integrated: {:?} of |L0|", runs.iter().map(|r| format!("{:.3e}", r.2)).collect::<Vec<_>>());
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("angular momentum less the bond imbalance", &errors, energy_floor(runs[3].1));
}

/// `corotational_bonds`: the bonds balance moments about the deformed configuration, so
/// angular momentum itself (no ledger) is conserved up to integration error that
/// vanishes with the substep, where the small-strain bonds turn the system by their
/// imbalance (5e-4 of `|L0|`).
#[test]
fn corotational_bonds_conserve_angular_momentum() {
    let drift = |corotational: bool| -> Vec<(f64, u64)> {
        [0.5, 0.25, 0.125, 0.0625]
            .iter()
            .map(|&k| {
                let mut s = free_wall_impact(40.0, false);
                s.sim.methods.corotational_bonds = corotational;
                s.sim.courant_safety = k;
                let dt = World::new(&s).substep_dt();
                s.sim.max_substep = Some(dt);
                s.sim.frame_dt = dt;
                let mut w = World::new(&s);
                let (_, l0) = momentum(&w);
                let mut worst = 0.0f64;
                let total = (0.03 / dt).round() as u64;
                while w.solver.substeps < total {
                    w.step_frame();
                    let (_, l) = momentum(&w);
                    worst = worst.max((l - l0).norm() / l0.norm());
                }
                (worst, w.solver.substeps)
            })
            .collect()
    };
    let (small, exact) = (drift(false), drift(true));
    println!("angular momentum drift / |L0|: small strain {:?}, co-rotational {:?}", small.iter().map(|r| format!("{:.3e}", r.0)).collect::<Vec<_>>(), exact.iter().map(|r| format!("{:.3e}", r.0)).collect::<Vec<_>>());
    let errors: Vec<f64> = exact.iter().map(|r| r.0).collect();
    assert_converges_to_zero("angular momentum with co-rotational bonds", &errors, energy_floor(exact[3].1));
}

/// What rounding can change in the total linear momentum after `w.solver.substeps`
/// substeps, with no body faster than `speed`, and (`arm = Some(reach)`) in the angular
/// momentum, with no body farther than `reach` from the origin (its terms `m x v` and
/// `I w`, `|I w| <= m r |v|`, round with the velocity at `reach` times `ulp(speed)` and
/// with the position at `speed` times `ulp(reach)`): every substep each body's (chunk's,
/// impactor's) velocity is updated along a path of at most eight roundings at the
/// scale of `speed` (load sum, rotation into the body frame, `F dt / m`, the add,
/// the floating frame's transfer), `4 ulp(speed)` per unit mass (DECISIONS.md 6);
/// measuring the momentum (`N` terms of `m v`, each of a few products) rounds by at
/// most `2 (N + 8) eps sum m |v|`, twice (start and now).
fn round_off_of_momentum(w: &World, speed: f64, arm: Option<f64>) -> f64 {
    let ulp = |x: f64| f64::from_bits(x.abs().to_bits() + 1) - x.abs();
    let sv = &w.solver;
    let mut mass = 0.0;
    let mut terms = 0usize;
    for st in &sv.structures {
        for ch in &st.chunks {
            mass += ch.mass;
            terms += 1;
        }
    }
    for imp in &w.impactors {
        mass += imp.mass;
        terms += 1;
    }
    let measure = 2.0 * 2.0 * (terms + 8) as f64 * f64::EPSILON * mass * speed;
    match arm {
        None => sv.substeps as f64 * mass * 4.0 * ulp(speed) + measure,
        Some(reach) => sv.substeps as f64 * mass * 4.0 * (ulp(speed) * reach + ulp(reach) * speed) + measure * reach,
    }
}

#[test]
fn fragments_inherit_parent_rigid_velocity_field() {
    // A spinning, translating free beam whose middle bond is cut: with no hidden
    // deformation, each half must move with v + w x r of the parent.
    let mut s = new_scene("spin", "spinning beam split");
    s.gravity = [0.0; 3];
    s.materials.insert("m".into(), Material::analytic_test());
    let chunks = grid(Vec3::new(-0.4, -0.05, -0.05), [8, 1, 1], Vec3::splat(0.1), "m");
    let bonds = auto_bonds(&chunks, |_, _| "m".into());
    let mut b = body("beam", chunks, bonds);
    b.linear_velocity = [1.0, 2.0, -0.5];
    b.angular_velocity = [0.3, -0.2, 5.0];
    s.bodies.push(b);
    let mut solver = ReferenceSolver::new(&s);
    // Some hidden deformation velocity too, so the split must redistribute it.
    for (i, c) in solver.chunks[0].iter_mut().enumerate() {
        c.v = Vec3::new(0.01 * i as f64, -0.02, 0.005 * (i % 3) as f64);
        c.w = Vec3::new(0.0, 0.1 * i as f64, 0.0);
    }
    // Momentum, and the sums of the magnitudes of its terms (the scale of its rounding).
    let momentum = |sv: &ReferenceSolver| {
        let (mut p, mut l, mut sp, mut sl) = (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0);
        for (c, ch) in sv.structures[0].chunks.iter().enumerate() {
            let (v, w) = sv.chunk_velocity(0, c);
            let r = sv.clusters[sv.chunks[0][c].cluster].rotation();
            let x = sv.chunk_position(0, c);
            p += v * ch.mass;
            l += x.cross(v) * ch.mass + r * ch.inertia * r.transpose() * w;
            sp += ch.mass * v.norm();
            sl += ch.mass * x.norm() * v.norm() + (r * ch.inertia * r.transpose() * w).norm() * 3.0;
        }
        (p, l, sp, sl)
    };
    let (p0, l0, sp, sl) = momentum(&solver);
    let mid = solver.bonds[0].iter().position(|b| b.geometry.a == 3 && b.geometry.b == 4).unwrap();
    solver.bonds[0][mid].joint.damage = 1.0;
    solver.split_cluster(0);
    assert_eq!(solver.clusters.len(), 2);
    let (p1, l1, _, _) = momentum(&solver);
    // Rounding: measuring sums `N` terms each formed by at most 10 roundings
    // (`v + w x r + R v_hidden`, the lever arm, the products), `(N + 10) eps` of the sum
    // of their magnitudes (Higham's gamma); the split forms each child's momentum and
    // re-expresses every chunk's velocity in it by the same count. Three such passes.
    let gamma = (solver.structures[0].chunks.len() + 10) as f64 * f64::EPSILON;
    let (bp, bl) = (3.0 * gamma * sp, 3.0 * gamma * sl);
    println!("split: linear {:.2e} of {bp:.2e}, angular {:.2e} of {bl:.2e}", (p1 - p0).norm(), (l1 - l0).norm());
    assert!((p1 - p0).norm() <= bp && (l1 - l0).norm() <= bl, "split changed momentum beyond rounding: {:.2e} > {bp:.2e} or {:.2e} > {bl:.2e}", (p1 - p0).norm(), (l1 - l0).norm());

    // Without hidden velocities, each half moves with the parent's rigid field.
    let mut solver = ReferenceSolver::new(&s);
    let parent = solver.clusters[0].clone();
    solver.bonds[0][mid].joint.damage = 1.0;
    solver.split_cluster(0);
    let com_p = parent.com_world();
    // Rounding: each child's rigid velocity is its chunks' momentum over its mass, each
    // chunk's velocity `v + w x r` (at most 10 roundings each, `N` terms summed), so it
    // carries `(N + 10) eps` of `|v| + |w| r_max`; its angular velocity, solved from the
    // angular momentum through the inertia, the same count again.
    let r_max = solver.structures[0].chunks.iter().map(|c| (c.center - parent.com).norm()).fold(0.0, f64::max);
    let scale = parent.velocity.norm() + parent.angular_velocity.norm() * r_max;
    let gamma = (solver.structures[0].chunks.len() + 10) as f64 * f64::EPSILON;
    for child in &solver.clusters {
        let expected = parent.velocity + parent.angular_velocity.cross(child.com_world() - com_p);
        println!("child: velocity {:.2e}, angular velocity {:.2e} (bounds {:.2e}, {:.2e})", (child.velocity - expected).norm(), (child.angular_velocity - parent.angular_velocity).norm(), 2.0 * gamma * scale, 2.0 * gamma * scale / r_max);
        assert!((child.velocity - expected).norm() <= 2.0 * gamma * scale, "{:?} vs {:?}", child.velocity, expected);
        assert!((child.angular_velocity - parent.angular_velocity).norm() <= 2.0 * gamma * scale / r_max);
    }
}

/// An error that the integration makes and that vanishes with the substep: `errors` at
/// substeps halving each time (at least four), each above the rounding `floor` (else no
/// trend can be seen and the check is passed). Their magnitudes must fall with every
/// halving, and the limit the sequence extrapolates to must be zero to within the
/// extrapolation's own uncertainty: Aitken's limit from the last three, against how far
/// it moved from the estimates of the (up to three) runs of three before. A residual
/// converging to something else (an energy the ledger misses) has the limits agree away
/// from zero.
/// No tolerance is chosen: the data's own convergence sets the resolution.
fn assert_converges_to_zero(label: &str, errors: &[f64], floor: f64) {
    let list = errors.iter().map(|e| format!("{e:.3e}")).collect::<Vec<_>>().join(", ");
    println!("{label}: errors [{list}] (round-off floor {floor:.1e})");
    assert!(errors.len() >= 4, "{label}: four substeps are needed to judge the limit");
    if errors.iter().all(|e| e.abs() <= floor) {
        return;
    }
    for w in errors.windows(2) {
        assert!(w[1].abs() < w[0].abs() || w[1].abs() <= floor, "{label}: the error does not shrink with the substep: [{list}]");
    }
    let aitken = |e: &[f64]| {
        let (d1, d2) = (e[1] - e[0], e[2] - e[1]);
        if d2 != d1 { e[2] - d2 * d2 / (d2 - d1) } else { e[2] }
    };
    // Aitken's limit from each run of three, the last against how far it moved over
    // the estimates before it (up to the last three).
    let limits: Vec<f64> = errors.windows(3).map(aitken).collect();
    let limit = *limits.last().unwrap();
    let moved = limits.iter().rev().skip(1).take(3).map(|l| (l - limit).abs()).fold(0.0, f64::max);
    println!("{label}: extrapolated limit {limit:.3e} (moved {moved:.3e} over the previous estimates)");
    assert!(limit.abs() <= moved + floor, "{label}: the error converges to {limit:.3e} (estimates moved {moved:.3e}), not zero: [{list}]");
}

/// What rounding alone does to an energy balance over `substeps` substeps: each
/// substep's sums of energies and works round by a few `eps` of the energy (eight
/// half-ulp roundings at its scale, DECISIONS.md 6), accumulating at worst linearly.
fn energy_floor(substeps: u64) -> f64 {
    substeps as f64 * 4.0 * f64::EPSILON
}

/// Over a run of `scene`: the largest imbalance `|E + D - E0| / E0` (mechanical energy
/// plus everything dissipated, against the initial energy), the substeps taken, and
/// whether the booked contact dissipation ever fell (frame to frame).
fn energy_balance(scene: &Scene) -> (f64, u64, f64) {
    let mut w = World::new(scene);
    let e0 = w.mechanical_energy();
    let (mut worst, mut contact, mut fell) = (0.0f64, 0.0, 0.0f64);
    for _ in 0..(scene.sim.duration / scene.sim.frame_dt).round() as usize {
        w.step_frame();
        let e = w.mechanical_energy() + w.dissipated_energy();
        if (e - e0).abs() > worst.abs() {
            worst = (e - e0) / e0;
        }
        fell = fell.max(contact - w.contact.dissipated);
        contact = w.contact.dissipated;
    }
    (worst, w.solver.substeps, fell)
}

/// Without fracture the mechanical energy plus what the contacts dissipate stays at the
/// initial energy, up to the explicit integration's error, which vanishes with the
/// substep (measured first order: 0.5 per halving with both contact laws).
#[test]
fn elastic_impact_conserves_energy_up_to_contact_dissipation() {
    let runs: Vec<(f64, u64, f64)> = [0.5, 0.25, 0.125, 0.0625]
        .iter()
        .map(|&k| {
            let mut s = free_wall_impact(5.0, false);
            s.sim.courant_safety = k;
            energy_balance(&s)
        })
        .collect();
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("elastic impact", &errors, energy_floor(runs[3].1));
}

/// Mechanical energy plus everything dissipated (fracture energy, friction, dashpots,
/// contact damping, softening overshoot, energy left in bonds a split separates) stays
/// at the initial energy: the residual is integration error and vanishes with the
/// substep. The contacts never give energy back: their booked dissipation never falls
/// (with the penalty contact it is a sum of non-negative terms, which floating point
/// keeps exactly monotone; with the elastic layer the work it actually took out of the
/// motion, net of what it stores).
///
/// The balance needs a contact law whose dissipation is what it takes out of the motion
/// (`layer_contact` books it from the work): the sample-point penalty contact books its
/// own dashpot quadrature, which is not the work its non-gradient force does (0.33 e0
/// booked against 0.34 taken at the finest substep here, a gap that does not shrink).
///
/// Status (DECISIONS.md): fails with `layer_contact`. The residual converges at first
/// order to about -1.3e-3 e0, not zero: an energy loss at a fixed rate while contacts
/// push loose, cracked chunks (none without contact, none without fracture). It is the
/// same small-strain floating-frame coupling that keeps angular momentum from being
/// conserved (`angular_momentum_changes_only_by_the_bond_imbalance`).
#[test]
fn fracture_never_gains_energy_and_dissipation_is_accounted() {
    let layer = Methods::default().layer_contact;
    let safeties: &[f64] = if layer { &[0.5, 0.25, 0.125, 0.0625, 0.03125] } else { &[0.5] };
    let runs: Vec<(f64, u64, f64)> = safeties
        .iter()
        .map(|&k| {
            let mut s = free_wall_impact(40.0, true);
            s.sim.courant_safety = k;
            energy_balance(&s)
        })
        .collect();
    for r in &runs {
        assert!(r.2 <= 0.0, "the contacts gave back energy: their booked dissipation fell by {:.3e} J", r.2);
    }
    if !layer {
        return;
    }
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("fracturing impact", &errors, energy_floor(runs[4].1));
}

/// A frictionless, perfectly elastic collision of two free cubes (face on face
/// overhanging an edge, tilted onto an edge, edge on edge, spinning) conserves energy:
/// the contact force is the gradient of the layer's energy, so what is left is the
/// integration error, which shrinks with the substep. A contact law that is not a
/// gradient (a normal tilted by an overhang, a stiffness changing with orientation
/// without its torque) gains or loses energy whatever the substep.
#[test]
fn an_elastic_collision_of_two_cubes_conserves_energy() {
    if !Methods::default().layer_contact {
        return; // the sample-point penalty contact is not a gradient (IMPROVEMENTS.md)
    }
    for (label, offset, tilt) in [("face", [0.03, 0.02], 0.0), ("tilted", [0.03, 0.02], 0.3), ("edge", [0.09, 0.0], 0.7)] {
        let error = |safety: f64| {
            let mut s = new_scene("two_cubes", "two free cubes collide");
            s.gravity = [0.0; 3];
            s.materials.insert("concrete".into(), oracle_concrete(None));
            s.bodies.push(body("a", vec![chunk(Vec3::ZERO, Vec3::splat(0.05), "concrete")], vec![]));
            let mut b = body("b", vec![chunk(Vec3::ZERO, Vec3::splat(0.05), "concrete")], vec![]);
            let q = stress_ref::math::Quat::from_axis_angle(Vec3::new(1.0, 0.5, 0.2).normalized(), tilt);
            // Just clear of a's top face.
            let low = (0..8)
                .map(|i| q.rotate(Vec3::new(if i & 1 == 0 { -0.05 } else { 0.05 }, if i & 2 == 0 { -0.05 } else { 0.05 }, if i & 4 == 0 { -0.05 } else { 0.05 })).z)
                .fold(f64::INFINITY, f64::min);
            b.position = [offset[0], offset[1], 0.05 - low + 2e-4];
            b.orientation = q.to_wxyz();
            b.linear_velocity = [0.3, -0.2, -2.0];
            b.angular_velocity = [3.0, -2.0, 1.0];
            s.bodies.push(b);
            s.sim.duration = 0.01;
            s.sim.frame_dt = 1e-3;
            s.sim.fracture = false;
            s.sim.gravity_prestress = false;
            s.sim.contact_restitution = 1.0;
            s.sim.contact_friction = Some(0.0);
            s.sim.courant_safety = safety;
            let mut w = World::new(&s);
            let e0 = w.mechanical_energy();
            for _ in 0..10 {
                w.step_frame();
            }
            assert!(w.contact.kinds[3].work != 0.0, "{label}: the cubes never touched");
            // What the contact took out of the motion is booked (zero but for crushing,
            // with `contact_crushing`: an edge concentrates the pressure past `f_c`).
            ((w.mechanical_energy() + w.contact.dissipated - e0) / e0, w.solver.substeps)
        };
        let runs: Vec<(f64, u64)> = [0.5, 0.25, 0.125, 0.0625, 0.03125].iter().map(|&k| error(k)).collect();
        let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
        assert_converges_to_zero(label, &errors, energy_floor(runs[4].1));
    }
}

/// The joint law alone balances energy: along any displacement path, the work done on
/// the bond (trapezoid rule) equals the energy it stores at the end plus everything it
/// dissipated, up to the quadrature's error, which vanishes with the step; and no
/// update ever releases negative energy (the second law, exactly). Random smooth paths
/// through tension, shear, compression, bending and torsion (40 fixed seeds) cross
/// damage, crushing, the cracked joint's contact patch and its friction. The law's own
/// accounts once booked up to twice what the joint actually released while its cracked
/// share still held the energy (DECISIONS.md 33).
#[test]
fn the_joint_law_balances_energy_on_any_path() {
    use stress_ref::bond::Local6;
    let mut s = new_scene("bond", "two chunks");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let chunks = grid(Vec3::new(-0.1, -0.05, -0.05), [2, 1, 1], Vec3::splat(0.1), "concrete");
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("b", chunks, bonds));
    let solver = ReferenceSolver::new(&s);
    let bond = &solver.bonds[0][0];
    let model = bond.model();
    let w = bond.geometry.width;
    // Displacements a few times the tensile strain at strength: every mechanism engages.
    let lin = 3e-5;
    let ang = lin / w[0].max(w[1]);
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for case in 0..40 {
        let mut a = [[0.0f64; 3]; 6];
        for c in a.iter_mut() {
            *c = [rnd() * 2.0, rnd() * 20.0 + 3.0, rnd() * 6.0];
        }
        let path = |t: f64| {
            let g = |c: usize| a[c][0] * (a[c][1] * t + a[c][2]).sin() * (1.0 + 3.0 * t) - a[c][0] * a[c][2].sin();
            Local6 { lin: Vec3::new(g(0), g(1), g(2)) * lin, ang: Vec3::new(g(3), g(4), g(5)) * ang }
        };
        let mut errors = Vec::new();
        let mut steps = 0;
        for n in [1000usize, 2000, 4000, 8000, 16000, 32000, 64000] {
            let dt = 1e-3 / n as f64;
            let mut st = bond.joint.clone();
            let (mut work, mut released, mut stored) = (0.0, 0.0, 0.0);
            let (mut f_prev, mut d_prev) = (Local6::ZERO, Local6::ZERO);
            for i in 1..=n {
                let d = path(i as f64 / n as f64);
                let r = model.evaluate(&st, &d, dt, true);
                assert!(r.dissipated + r.overshoot >= 0.0, "case {case}: an update released {:e} J (step {i} of {n})", r.dissipated + r.overshoot);
                let dd = Local6 { lin: d.lin - d_prev.lin, ang: d.ang - d_prev.ang };
                work += 0.5 * (f_prev.dot(&dd) + r.force.dot(&dd));
                released += r.dissipated + r.overshoot;
                stored = r.stored;
                f_prev = r.force;
                d_prev = d;
                st = r.state;
            }
            errors.push((work - stored - released) / work.abs());
            steps = n;
        }
        assert_converges_to_zero(&format!("joint path {case}"), &errors, energy_floor(steps as u64));
    }
}

/// A split keeps every chunk where it is and moving as it was, whatever hidden
/// deformation the parent carries (displacements, rotations, their rates): momentum,
/// angular momentum and kinetic energy are unchanged to rounding (Higham's
/// `gamma_{N+10}` of the sums of the terms' magnitudes, three passes, as above). The
/// established split re-expressed velocities with lever arms that leave out the hidden
/// displacement (5e-2 J created here); `exact_frame_transfer` keeps them exactly
/// (DECISIONS.md 35).
#[test]
fn a_split_of_a_deformed_body_keeps_its_motion() {
    let mut s = new_scene("spin", "spinning, deformed beam split");
    s.gravity = [0.0; 3];
    s.materials.insert("m".into(), Material::analytic_test());
    let chunks = grid(Vec3::new(-0.4, -0.05, -0.05), [8, 1, 1], Vec3::splat(0.1), "m");
    let bonds = auto_bonds(&chunks, |_, _| "m".into());
    let mut b = body("beam", chunks, bonds);
    b.linear_velocity = [1.0, 2.0, -0.5];
    b.angular_velocity = [0.3, -0.2, 5.0];
    s.bodies.push(b);
    let mut solver = ReferenceSolver::new(&s);
    for (i, c) in solver.chunks[0].iter_mut().enumerate() {
        c.u = Vec3::new(0.001 * i as f64, -0.002, 0.0005 * (i % 3) as f64);
        c.th = Vec3::new(0.01, 0.02 * i as f64, -0.01);
        c.v = Vec3::new(0.01 * i as f64, -0.02, 0.005 * (i % 3) as f64);
        c.w = Vec3::new(0.0, 0.1 * i as f64, 0.0);
    }
    // Momentum, angular momentum, kinetic energy and the sums of their terms' magnitudes.
    let state = |sv: &ReferenceSolver| {
        let (mut p, mut l, mut ke, mut sp, mut sl) = (Vec3::ZERO, Vec3::ZERO, 0.0, 0.0, 0.0);
        for (c, ch) in sv.structures[0].chunks.iter().enumerate() {
            let (v, w) = sv.chunk_velocity(0, c);
            let r = sv.clusters[sv.chunks[0][c].cluster].rotation();
            let x = sv.chunk_position(0, c);
            let iw = r * ch.inertia * r.transpose() * w;
            p += v * ch.mass;
            l += x.cross(v) * ch.mass + iw;
            ke += 0.5 * ch.mass * v.norm2() + 0.5 * w.dot(iw);
            sp += ch.mass * v.norm();
            sl += ch.mass * x.norm() * v.norm() + iw.norm() * 3.0;
        }
        (p, l, ke, sp, sl)
    };
    let (p0, l0, k0, sp, sl) = state(&solver);
    let mid = solver.bonds[0].iter().position(|b| b.geometry.a == 3 && b.geometry.b == 4).unwrap();
    solver.bonds[0][mid].joint.damage = 1.0;
    solver.split_cluster(0);
    assert_eq!(solver.clusters.len(), 2);
    let (p1, l1, k1, _, _) = state(&solver);
    let gamma = (solver.structures[0].chunks.len() + 10) as f64 * f64::EPSILON;
    println!("split of a deformed beam: momentum {:.2e}, angular momentum {:.2e}, kinetic energy {:.2e}", (p1 - p0).norm(), (l1 - l0).norm(), k1 - k0);
    assert!((p1 - p0).norm() <= 3.0 * gamma * sp, "momentum changed by {:.2e}", (p1 - p0).norm());
    assert!((l1 - l0).norm() <= 3.0 * gamma * sl, "angular momentum changed by {:.2e}", (l1 - l0).norm());
    // Kinetic energy: each term `m v^2 / 2` rounds like the momentum's, times a speed.
    let speed = sp / solver.structures[0].chunks.iter().map(|c| c.mass).sum::<f64>();
    assert!((k1 - k0).abs() <= 3.0 * gamma * (sp * speed + sl), "kinetic energy changed by {:.2e}", k1 - k0);
}
