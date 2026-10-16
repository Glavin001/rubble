//! Failure capabilities of the spec: tension cutoff, Mohr-Coulomb shear, crushing,
//! buckling cap, dynamic increase factor, fracture-energy softening (resolution
//! independence), static fatigue, Weibull strengths, same-step cascade, rebar,
//! steel vs brittle energy absorption.

mod common;

use stress_ref::bond::{BondGeometry, BondStiffness, Local6};
use stress_ref::builders::*;
use stress_ref::joint::{JointModel, JointState, JointStrength};
use stress_ref::material::{gamma, Material, StaticFatigue};
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::solver::{ChunkLoads, ReferenceSolver, SolverEvent, CREAK_STRENGTH};
use stress_ref::statics::StaticOptions;
use stress_ref::world::World;

const AREA: f64 = 0.01;

fn cube_bond() -> BondGeometry {
    BondGeometry::new(0, 1, Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0), Vec3::X, Vec3::Y, AREA, [0.1, 0.1])
}

/// Drive a bond along `base + s * dir` (local generalised displacements) and return
/// the actual force at the first evaluation that damages it.
fn force_at_onset(m: &Material, buckling: Option<f64>, base: Local6, dir: Local6) -> Local6 {
    let g = cube_bond();
    let k = BondStiffness::new(&g, m, 1.0);
    let strength = JointStrength::new(m, &g, buckling, 1.0, &Features::default());
    let model = JointModel { geometry: &g, stiffness: &k, strength: &strength, rebar: None, weibull: 1.0 };
    let mut state = JointState::new();
    let mut last = Local6::ZERO;
    for i in 0..2_000_000 {
        let d = base.add(&dir.scale(i as f64 * 1e-9));
        let r = model.evaluate(&state, &d, 0.0, true);
        if r.state.damage > 0.0 || r.state.crush > 0.0 {
            return last;
        }
        last = r.force;
        state = r.state;
    }
    panic!("bond never failed");
}

#[test]
fn mohr_coulomb_shear_strength_rises_with_compression() {
    let m = Material::analytic_test();
    let kn = m.youngs_modulus * AREA / 0.1;
    let mut previous = 0.0;
    for sigma in [0.0, 1e6, 2e6, 4e6] {
        let base = Local6 { lin: Vec3::new(0.0, 0.0, -sigma * AREA / kn), ang: Vec3::ZERO };
        let f = force_at_onset(&m, None, base, Local6 { lin: Vec3::new(1.0, 0.0, 0.0), ang: Vec3::ZERO });
        let expected = (m.cohesion + m.friction * sigma) * AREA;
        assert!(common::rel(f.lin.x, expected) < 1e-3, "sigma {sigma}: {} vs {expected}", f.lin.x);
        assert!(f.lin.x > previous);
        previous = f.lin.x;
    }
}

#[test]
fn tension_cutoff_and_crushing() {
    let m = Material::analytic_test();
    let t = force_at_onset(&m, None, Local6::ZERO, Local6 { lin: Vec3::Z, ang: Vec3::ZERO });
    assert!(common::rel(t.lin.z, m.tensile_strength * AREA) < 1e-3);
    let c = force_at_onset(&m, None, Local6::ZERO, Local6 { lin: -Vec3::Z, ang: Vec3::ZERO });
    assert!(common::rel(-c.lin.z, m.compressive_strength * AREA) < 1e-3);
    // Bending: the extreme fibre reaches f_t at M = f_t S.
    let b = force_at_onset(&m, None, Local6::ZERO, Local6 { lin: Vec3::ZERO, ang: Vec3::X * 10.0 });
    assert!(common::rel(b.ang.x, m.tensile_strength * 0.1f64.powi(3) / 6.0) < 1e-3);
}

#[test]
fn buckling_caps_compression_of_slender_members_at_the_euler_load() {
    let m = Material::analytic_test();
    let lb = 2.0;
    let p_cr = std::f64::consts::PI.powi(2) * m.youngs_modulus * (0.1f64.powi(4) / 12.0) / (lb * lb);
    assert!(p_cr < m.compressive_strength * AREA);
    let c = force_at_onset(&m, Some(lb), Local6::ZERO, Local6 { lin: -Vec3::Z, ang: Vec3::ZERO });
    assert!(common::rel(-c.lin.z, p_cr) < 1e-3, "{} vs {p_cr}", -c.lin.z);
    // Beyond the cap the member holds about the Euler load (plateau), it does not snap.
    let g = cube_bond();
    let k = BondStiffness::new(&g, &m, 1.0);
    let strength = JointStrength::new(&m, &g, Some(lb), 1.0, &Features::default());
    let model = JointModel { geometry: &g, stiffness: &k, strength: &strength, rebar: None, weibull: 1.0 };
    let mut st = JointState::new();
    let d0 = p_cr / k.kn;
    let mut f = 0.0;
    for i in 1..=300 {
        let r = model.evaluate(&st, &Local6 { lin: Vec3::new(0.0, 0.0, -d0 * (1.0 + 0.01 * i as f64)), ang: Vec3::ZERO }, 0.0, true);
        st = r.state;
        f = -r.force.lin.z;
    }
    assert!(common::rel(f, p_cr) < 1e-3 && st.crush < 1.0, "plateau force {f}");
}

/// The effective tensile stress at which damage starts equals the static strength times
/// the DIF of the strain rate at that moment, and is higher for faster loading.
#[test]
fn dynamic_increase_factor_raises_strength_with_loading_rate() {
    let m = Material { weibull_modulus: None, static_fatigue: None, damping_ratio: 0.0, ..Material::concrete() };
    let dif = m.dif.unwrap();
    let h = 0.05;
    let f_static = m.tensile_strength * 4.0 * h * h;
    // Returns (onset stress / static strength, strain rate at onset).
    let onset = |ramp: f64| {
        let mut s = bond_pull("dif", m.clone(), h, TimeFunction::Ramp { t0: 0.0, t1: ramp, value: 3.0 * f_static }, ramp + 3e-3);
        s.sim.frame_dt = (ramp / 200.0).min(1e-4);
        let mut w = World::new(&s);
        loop {
            let (before, rate) = (w.solver.bonds[0][0].measures, w.solver.bonds[0][0].joint.strain_rate);
            w.step_frame();
            if w.solver.bonds[0][0].joint.damage > 0.0 {
                return (before.tension / m.tensile_strength, rate);
            }
            assert!(w.solver.time < ramp + 3e-3, "never failed");
        }
    };
    let (slow, slow_rate) = onset(1.0);
    let (fast, fast_rate) = onset(2e-4);
    println!("slow {slow} at {slow_rate}/s, fast {fast} at {fast_rate}/s");
    assert!(fast_rate > 100.0 * slow_rate);
    assert!(slow >= 1.0 && common::rel(slow, dif.factor(slow_rate)) < 0.02, "slow {slow}");
    assert!(common::rel(fast, dif.factor(fast_rate)) < 0.05, "fast {fast} vs {}", dif.factor(fast_rate));
    assert!(fast > 1.1 * slow);
}

#[test]
fn fracture_energy_per_area_is_independent_of_chunk_size() {
    let m = Material::analytic_test();
    for h in [0.025, 0.05, 0.1] {
        let area = 4.0 * h * h;
        let s = bond_pull("size", m.clone(), h, TimeFunction::Ramp { t0: 0.0, t1: 0.5, value: 2.0 * m.tensile_strength * area }, 0.6);
        let obs = common::run(&s);
        let per_area = obs.values["bond_dissipation"] / area;
        assert!(common::rel(per_area, m.fracture_energy.tension) < 0.01, "h {h}: {per_area}");
    }
}

#[test]
fn steel_absorbs_its_large_fracture_energy_brick_shatters_cheaply() {
    let h = 0.05;
    let area = 4.0 * h * h;
    let mut absorbed = Vec::new();
    for m in [Material::steel(), Material::brick()] {
        let m = Material { damping_ratio: 0.0, ..m };
        let s = bond_pull("absorb", m.clone(), h, TimeFunction::Ramp { t0: 0.0, t1: 0.5, value: 1.5 * m.tensile_strength * area }, 0.6);
        let obs = common::run(&s);
        let e = obs.values["bond_dissipation"];
        assert!(common::rel(e, m.fracture_energy.tension * area) < 0.02, "{e} vs {}", m.fracture_energy.tension * area);
        absorbed.push(e);
    }
    assert!(absorbed[0] > 1000.0 * absorbed[1]);
}

/// Creak, crack, give way (static fatigue): a bond held at 97% of its strength creaks
/// when it has lost 1% of its strength and fails at the closed-form lifetime
/// `t_test (1 - s^(n-2)) / ((n + 1) s^n)`; at 60% its lifetime is hours.
#[test]
fn sustained_overload_creaks_cracks_then_gives_way() {
    let fatigue = StaticFatigue { exponent: 20.0, test_time: 10.0 };
    let m = Material { static_fatigue: Some(fatigue), dif: None, damping_ratio: 0.05, ..Material::analytic_test() };
    let h = 0.05;
    let area = 4.0 * h * h;
    let ratio = 0.97;
    let s = bond_pull("fatigue", m.clone(), h, TimeFunction::Constant { value: ratio * m.tensile_strength * area }, 1.0);
    let mut w = World::new(&s);
    w.run();
    let ev = &w.solver.events;
    let time = |pred: &dyn Fn(&SolverEvent) -> bool| ev.iter().find(|e| pred(e)).map(|e| match e {
        SolverEvent::Creaked { time, .. } | SolverEvent::Cracked { time, .. } | SolverEvent::Broken { time, .. } => *time,
        _ => f64::NAN,
    });
    let creak = time(&|e| matches!(e, SolverEvent::Creaked { .. })).expect("creak");
    let crack = time(&|e| matches!(e, SolverEvent::Cracked { .. })).expect("crack");
    let broken = time(&|e| matches!(e, SolverEvent::Broken { .. })).expect("give way");
    let n = fatigue.exponent;
    let creak_expected = (1.0 - CREAK_STRENGTH.powf(n - 2.0)) / fatigue.life_rate(ratio);
    assert!(common::rel(creak, creak_expected) < 0.01, "creak {creak} vs {creak_expected}");
    assert!(common::rel(crack, fatigue.lifetime(ratio)) < 0.01, "crack {crack} vs {}", fatigue.lifetime(ratio));
    assert!(creak < crack && crack <= broken);

    let s = bond_pull("fatigue_low", m.clone(), h, TimeFunction::Constant { value: 0.6 * m.tensile_strength * area }, 1.0);
    let mut w = World::new(&s);
    w.run();
    assert!(fatigue.lifetime(0.6) > 3600.0);
    assert!(w.solver.events.is_empty(), "no damage within a second at 60%");
}

#[test]
fn weibull_strengths_have_unit_mean_and_the_weibull_spread() {
    let modulus = 8.0;
    let mut s = new_scene("weibull", "");
    s.materials.insert("c".into(), Material { weibull_modulus: Some(modulus), ..Material::analytic_test() });
    let chunks = grid(Vec3::ZERO, [20, 20, 5], Vec3::splat(0.1), "c");
    let bonds = auto_bonds(&chunks, |_, _| "c".into());
    s.bodies.push(body("b", chunks, bonds));
    let factors = |seed: u64| -> Vec<f64> {
        let mut sc = s.clone();
        sc.sim.seed = seed;
        ReferenceSolver::new(&sc).bonds[0].iter().map(|b| b.weibull).collect()
    };
    let f = factors(1);
    let n = f.len() as f64;
    let mean = f.iter().sum::<f64>() / n;
    let cv = (f.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt() / mean;
    let cv_expected = (gamma(1.0 + 2.0 / modulus) / gamma(1.0 + 1.0 / modulus).powi(2) - 1.0).sqrt();
    assert!((mean - 1.0).abs() < 0.01, "mean {mean}");
    assert!(common::rel(cv, cv_expected) < 0.05, "cv {cv} vs {cv_expected}");
    assert_eq!(f, factors(1), "same seed, same strengths");
    assert_ne!(f, factors(2), "different seed, different strengths");
}

/// A bar hung from five supports with slightly different strengths. A load just above
/// the weakest support's share breaks it; the others are then overloaded and the
/// same-step cascade (re-solve, recheck) drops the bar within one call.
#[test]
fn same_step_cascade_resolves_until_stable() {
    let m = Material::analytic_test();
    let mut s = new_scene("bundle", "");
    s.gravity = [0.0; 3];
    s.materials.insert("m".into(), m.clone());
    let mut chunks = grid(Vec3::new(0.0, 0.0, 0.1), [5, 1, 1], Vec3::splat(0.1), "m");
    for c in &mut chunks {
        c.support = Support::Fixed;
    }
    chunks.push(chunk(Vec3::new(0.25, 0.05, 0.05), Vec3::new(0.25, 0.05, 0.05), "m"));
    let bonds = auto_bonds(&chunks, |_, _| "m".into());
    assert_eq!(bonds.len(), 9, "4 bonds between supports, 5 to the bar");
    s.bodies.push(body("b", chunks, bonds));
    let run = |cascade: bool| {
        let mut solver = ReferenceSolver::new(&s);
        let bar = 5;
        for (i, b) in solver.bonds[0].iter_mut().filter(|b| b.geometry.b == bar).enumerate() {
            b.weibull = 1.0 + 0.05 * i as f64;
        }
        let mut loads = ChunkLoads::new(&solver);
        let w = 5.0 * m.tensile_strength * AREA * 1.02;
        loads.force[0][bar] = Vec3::new(0.0, 0.0, -w);
        let opts = StaticOptions { cascade, max_cascade: if cascade { 200 } else { 1 }, ..Default::default() };
        let r = solver.solve_static_all(&loads, &opts, 0.0);
        (r, solver.broken_bond_count())
    };
    let (r, broken) = run(true);
    assert_eq!(broken, 5, "{r:?} broken {broken}");
    assert!(r.cascade_passes > 2);
    let (_, broken_once) = run(false);
    assert!(broken_once < 5, "without the cascade only the first break happens ({broken_once})");
}

#[test]
fn rebar_holds_cracked_concrete_together_until_it_ruptures() {
    let m = Material { damping_ratio: 0.05, ..Material::analytic_test() };
    let h = 0.05;
    let mut s = bond_pull("rebar", m.clone(), h, TimeFunction::Ramp { t0: 0.0, t1: 1.0, value: 60e3 }, 1.0);
    s.sim.frame_dt = 1e-4;
    s.materials.insert("steel".into(), Material::steel());
    s.bodies[0].bonds[0].rebar = Some(RebarSpec { area: 1e-4, material: "steel".into() });
    let mut w = World::new(&s);
    let mut held_while_cracked = false;
    for _ in 0..10_000 {
        w.step_frame();
        let b = &w.solver.bonds[0][0];
        if b.joint.damage >= 1.0 && !b.joint.rebar_broken {
            assert_eq!(w.solver.clusters.len(), 1, "the rebar keeps the cracked bond connected");
            held_while_cracked = true;
        }
    }
    assert!(held_while_cracked);
    assert!(w.solver.bonds[0][0].joint.rebar_broken, "the rebar finally ruptures");
    assert_eq!(w.solver.clusters.len(), 2);
}
