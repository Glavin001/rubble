//! `sim.methods.multirate`: explicit multirate integration of the bond network. Its
//! bonds kick both chunks with balanced impulses, so it must conserve what the
//! single-rate integrator conserves: linear momentum to round-off, angular momentum
//! (co-rotational bonds) and energy up to integration error that vanishes with the
//! substep; and where every chunk shares one rate it must be the single-rate
//! integrator, bit for bit.

use stress_ref::builders::*;
use stress_ref::material::Material;
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::world::World;

/// The sliver wall's light column (along x).
const SLIVER: usize = 4;

/// A free concrete wall of 0.1 m cubes in zero gravity, ten wide, two thick and ten
/// high, spinning slowly, hit by a steel sphere at `speed` on the centre of the cube
/// beside the sliver column. The fifth column's cubes are slivers: concrete at 1/25 of
/// its density, bonded with concrete, so their own stable step is a fifth of the
/// others'. (Cubes keep each chunk's inertia isotropic: a geometric sliver's hidden
/// rotation turns its anisotropic inertia, which neither integrator follows, and costs
/// single rate 1e-5 of the angular momentum.)
fn sliver_wall(speed: f64, fracture: bool) -> Scene {
    let mut s = new_scene("sliver_wall", "free wall with a light column hit by a sphere in zero gravity");
    s.gravity = [0.0; 3];
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("light".into(), Material { density: oracle_concrete(None).density / 25.0, ..oracle_concrete(None) });
    s.materials.insert("steel".into(), elastic_steel());
    let mut chunks = grid(Vec3::new(-0.5, 0.0, -0.5), [10, 2, 10], Vec3::splat(0.1), "concrete");
    for (k, c) in chunks.iter_mut().enumerate() {
        if k % 10 == SLIVER {
            c.material = "light".into();
        }
    }
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
    s.sim.methods.multirate = true;
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

/// The bond evaluations a single-rate run of the same substeps takes, from the cluster
/// bond counts at each substep: here, the bonds alive times the substeps is an upper
/// bound, and multirate must stay well under it.
fn single_rate_evaluations(w: &World) -> f64 {
    w.solver.bonds.iter().map(|b| b.len()).sum::<usize>() as f64 * w.solver.substeps as f64
}

/// The rates the sliver wall's bonds take: the sliver column and its bonds to its
/// neighbours run at the substep, the rest of the wall at a multiple of it.
#[test]
fn slivers_run_at_the_substep_and_the_rest_at_its_multiples() {
    let scene = sliver_wall(10.0, false);
    let w = World::new(&scene);
    let dt = w.substep_dt();
    let contact = w.chunk_contact_dts();
    let rates = w.solver.bond_rates(0, dt, 1 << 20, &|s, c| contact[s][c]);
    let sliver = |c: usize| c % 10 == SLIVER;
    let mut coarse = u32::MAX;
    for &(bi, r) in &rates {
        let g = &w.solver.bonds[0][bi].geometry;
        if sliver(g.a) || sliver(g.b) {
            assert_eq!(r, 1, "a sliver bond runs at the substep");
        } else {
            coarse = coarse.min(r);
        }
    }
    println!("substep {dt:.3e} s; rates {:?}", rates.iter().map(|r| r.1).collect::<std::collections::BTreeSet<_>>());
    assert!(coarse >= 2, "the cubes' bonds should run at a multiple of the substep (lowest rate {coarse})");
}

/// Where every bond takes rate 1 the multirate integrator is the single-rate one, bit
/// for bit (a uniform wall, struck hard enough to break).
#[test]
fn one_rate_everywhere_is_single_rate_exactly() {
    let mut s = sliver_wall(40.0, true);
    s.bodies[0].chunks = grid(Vec3::new(-0.5, 0.0, -0.5), [10, 2, 10], Vec3::splat(0.1), "concrete");
    s.bodies[0].bonds = auto_bonds(&s.bodies[0].chunks, |_, _| "concrete".into());
    s.sim.duration = 0.01;
    let on = World::new(&s).run();
    s.sim.methods.multirate = false;
    let off = World::new(&s).run();
    assert!(on.values["broken_bonds"] > 0.0, "the uniform wall should break");
    for (k, v) in &off.values {
        if k != "wall_seconds" {
            assert_eq!(on.values[k].to_bits(), v.to_bits(), "{k}: {} with multirate, {v} without", on.values[k]);
        }
    }
    assert_eq!(on.bodies.len(), off.bodies.len());
    for (name, chunks) in &off.bodies {
        for (a, b) in on.bodies[name].iter().zip(chunks) {
            assert_eq!(a.displacement, b.displacement);
            assert_eq!(a.velocity, b.velocity);
        }
    }
}

/// Linear momentum is conserved to round-off through impact and fracture, as in single
/// rate: every bond impulse acts on both chunks at once, equal and opposite.
#[test]
fn multirate_conserves_linear_momentum_through_impact_and_fracture() {
    let scene = sliver_wall(40.0, true);
    let mut w = World::new(&scene);
    let (p0, _) = momentum(&w);
    let mass: f64 = w.solver.structures.iter().flat_map(|s| s.chunks.iter()).map(|c| c.mass).sum::<f64>() + 50.0;
    let ulp = |x: f64| f64::from_bits(x.abs().to_bits() + 1) - x.abs();
    let mut worst: f64 = 0.0;
    for _ in 0..30 {
        w.step_frame();
        let (p, _) = momentum(&w);
        // `conservation.rs`'s round-off bound: four ulps of the speed per substep and
        // unit mass, and the measurement's rounding.
        let terms = w.solver.structures.iter().map(|s| s.chunks.len()).sum::<usize>() + 1;
        let bound = w.solver.substeps as f64 * mass * 4.0 * ulp(40.0) + 4.0 * (terms + 8) as f64 * f64::EPSILON * mass * 40.0;
        worst = worst.max((p - p0).norm() / bound);
        assert!((p - p0).norm() <= bound, "linear momentum drift {:.2e} beyond the round-off bound {bound:.2e}", (p - p0).norm());
    }
    println!("linear momentum: worst drift {worst:.2e} of the round-off bound; bond evaluations {} of {:.0}", w.solver.bond_evaluations, single_rate_evaluations(&w));
    assert!(w.solver.broken_bond_count() > 0, "the impact should fracture the wall");
    assert!(w.solver.clusters.len() > 1, "the wall should split");
    assert!((w.solver.bond_evaluations as f64) < 0.75 * single_rate_evaluations(&w), "multirate should skip most of the cubes' bond evaluations");
}

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
        if d2 != d1 {
            e[2] - d2 * d2 / (d2 - d1)
        } else {
            e[2]
        }
    };
    let limits: Vec<f64> = errors.windows(3).map(aitken).collect();
    let limit = *limits.last().unwrap();
    let moved = limits.iter().rev().skip(1).take(3).map(|l| (l - limit).abs()).fold(0.0, f64::max);
    println!("{label}: extrapolated limit {limit:.3e} (moved {moved:.3e} over the previous estimates)");
    assert!(limit.abs() <= moved + floor, "{label}: the error converges to {limit:.3e} (estimates moved {moved:.3e}), not zero: [{list}]");
}

fn energy_floor(substeps: u64) -> f64 {
    substeps as f64 * 4.0 * f64::EPSILON
}

/// Angular momentum (no external torque) changes as in single rate: with co-rotational
/// bonds, the largest difference between the two runs' angular momenta over the impact
/// vanishes with the substep (every safety keeps the slivers at the substep and the
/// cubes at a multiple of it). Single rate itself does not conserve it exactly here:
/// it drifts by 1.5e-5 of `|L0|` whatever the substep (1.7e-5 without spin), where the
/// uniform wall's drift vanishes (`conservation.rs`); multirate shares that drift.
#[test]
fn multirate_changes_angular_momentum_as_single_rate_does() {
    let runs: Vec<(f64, u64, f64, f64)> = [0.5, 0.25, 0.125, 0.0625]
        .iter()
        .map(|&k| {
            let mut s = sliver_wall(40.0, false);
            s.sim.methods.corotational_bonds = true;
            s.sim.courant_safety = k;
            let mut multi = World::new(&s);
            s.sim.methods.multirate = false;
            let mut single = World::new(&s);
            let (_, l0) = momentum(&multi);
            let (mut worst, mut drift) = (0.0f64, 0.0f64);
            for _ in 0..30 {
                multi.step_frame();
                single.step_frame();
                let ((_, lm), (_, ls)) = (momentum(&multi), momentum(&single));
                worst = worst.max((lm - ls).norm() / l0.norm());
                drift = drift.max((ls - l0).norm() / l0.norm());
            }
            (worst, multi.solver.substeps, multi.solver.bond_evaluations as f64 / single.solver.bond_evaluations as f64, drift)
        })
        .collect();
    println!("bond evaluations over single rate: {:?}; single rate's own drift {:?}", runs.iter().map(|r| format!("{:.3}", r.2)).collect::<Vec<_>>(), runs.iter().map(|r| format!("{:.3e}", r.3)).collect::<Vec<_>>());
    for r in &runs {
        assert!(r.2 < 0.75, "multirate should be active at every safety ({:.3} of single rate)", r.2);
    }
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("angular momentum, multirate against single rate", &errors, energy_floor(runs[3].1));
}

/// Over a run of `scene`: the largest imbalance `|E + D - E0| / E0` (mechanical energy
/// plus everything dissipated, against the initial energy) and the substeps taken.
fn energy_balance(scene: &Scene) -> (f64, u64) {
    let mut w = World::new(scene);
    let e0 = w.mechanical_energy();
    let mut worst = 0.0f64;
    for _ in 0..(scene.sim.duration / scene.sim.frame_dt).round() as usize {
        w.step_frame();
        let e = w.mechanical_energy() + w.dissipated_energy();
        if (e - e0).abs() > worst.abs() {
            worst = (e - e0) / e0;
        }
    }
    (worst, w.solver.substeps)
}

/// Without fracture the mechanical energy plus what the contacts and dashpots dissipate
/// stays at the initial energy up to integration error that vanishes with the substep:
/// the elastic impact on the sliver, as `conservation.rs` checks it in single rate.
#[test]
fn multirate_conserves_energy_through_an_elastic_impact() {
    let runs: Vec<(f64, u64)> = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
        .iter()
        .map(|&k| {
            let mut s = sliver_wall(5.0, false);
            s.sim.courant_safety = k;
            energy_balance(&s)
        })
        .collect();
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("elastic impact, multirate", &errors, energy_floor(runs[runs.len() - 1].1));
}

/// The free sliver wall without damping or contact, set vibrating by a smooth hidden
/// velocity field, at stress safety `safety`, in frames of `ticks` substeps.
fn vibrating_wall(safety: f64, ticks: usize, duration: f64) -> World {
    let mut s = sliver_wall(0.0, false);
    s.impactors.clear();
    s.sim.features.damping = false;
    s.sim.methods.rayleigh_damping = false;
    s.sim.courant_safety = safety;
    let dt = World::new(&s).substep_dt();
    s.sim.max_substep = Some(dt);
    s.sim.frame_dt = dt * ticks as f64;
    s.sim.duration = duration;
    let mut w = World::new(&s);
    for (c, st) in w.solver.chunks[0].iter_mut().enumerate() {
        let p = s.bodies[0].chunks[c].center;
        st.v = Vec3::new((7.0 * p[2]).sin(), (5.0 * p[0]).cos() * 0.5, (3.0 * p[0] + 2.0 * p[2]).sin()) * 0.05;
    }
    w
}

/// The energy error `(E + D - E0) / E0` of `w` at the end of every frame up to its duration.
fn energy_errors(w: &mut World) -> Vec<f64> {
    let e0 = w.mechanical_energy();
    let frames = (w.scene.sim.duration / w.scene.sim.frame_dt).round() as usize;
    (0..frames)
        .map(|_| {
            w.step_frame();
            (w.mechanical_energy() + w.dissipated_energy() - e0) / e0
        })
        .collect()
}

/// A free, undamped, vibrating sliver wall keeps its energy: the error (measured at
/// frame ends, where every bond's interval closes) shrinks with the substep at fixed
/// rates. Leapfrog's velocities lag the positions by half an interval, so the reading
/// is first order, with the coarse bonds' interval in place of the substep. Frames are
/// 64 substeps at every safety, so every run closes the coarse intervals alike.
#[test]
fn multirate_energy_converges_in_free_vibration() {
    let runs: Vec<(f64, u64)> = [0.4, 0.2, 0.1, 0.05]
        .iter()
        .map(|&k| {
            let mut w = vibrating_wall(k, 64, 0.01);
            let worst = energy_errors(&mut w).iter().fold(0.0f64, |a, e| a.max(e.abs()));
            println!("safety {k}: worst {worst:.3e}, {} substeps, bond evaluations {:.3} of single rate", w.solver.substeps, w.solver.bond_evaluations as f64 / single_rate_evaluations(&w));
            (worst, w.solver.substeps)
        })
        .collect();
    let errors: Vec<f64> = runs.iter().map(|r| r.0).collect();
    assert_converges_to_zero("free vibration, multirate", &errors, energy_floor(runs[runs.len() - 1].1));
}

/// At safety 0.9 (A1's; the default is 0.5) the coarse bonds' interval is a multiple of a
/// step already near the slivers' stability limit, where impulse multiple-time-stepping can
/// resonate with the fast modes (Biesiadecki and Skeel 1993). Over 1.6e5 substeps of free
/// vibration without damping the energy error stays bounded: the last tenth of the run
/// is no worse than the first.
#[test]
fn multirate_energy_does_not_drift_at_a_high_safety() {
    let mut w = vibrating_wall(0.9, 1000, 0.25);
    let errors = energy_errors(&mut w);
    let tenth = errors.len() / 10;
    let worst = |e: &[f64]| e.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let (first, last) = (worst(&errors[..tenth]), worst(&errors[errors.len() - tenth..]));
    println!("energy error, first tenth {first:.3e}, last tenth {last:.3e} ({} substeps; bond evaluations {:.3} of single rate)", w.solver.substeps, w.solver.bond_evaluations as f64 / single_rate_evaluations(&w));
    assert!(first < 1e-2, "the energy error should be small: {first:.3e}");
    assert!(last <= 1.5 * first, "the energy error grows: {first:.3e} in the first tenth, {last:.3e} in the last");
}
