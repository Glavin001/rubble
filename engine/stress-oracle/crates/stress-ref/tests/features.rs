//! Model switches (`sim.features`): each switch removes exactly the mechanism it names.
//! These are the "without it the scene behaves like this" comparisons, made checkable.

mod common;

use stress_ref::bond::{BondGeometry, Local6};
use stress_ref::builders::*;
use stress_ref::joint::{JointModel, JointState};
use stress_ref::material::{Material, StaticFatigue};
use stress_ref::math::Vec3;
use stress_ref::scene::*;
use stress_ref::structure::{bond_physics, Structure};
use stress_ref::world::World;

#[test]
fn switches_round_trip_and_unknown_names_are_rejected() {
    let mut f = Features::default();
    for name in Features::NAMES {
        f.set(name, false).unwrap();
    }
    let json = serde_json::to_value(f).unwrap();
    for name in Features::NAMES {
        assert_eq!(json[name], false, "{name}");
    }
    assert!(f.set("warp_drive", true).is_err());
    // Scenes written before the switches existed get every feature on.
    let mut json = serde_json::to_value(bond_tension()).unwrap();
    json["sim"].as_object_mut().unwrap().remove("features");
    let scene: Scene = serde_json::from_value(json).unwrap();
    assert_eq!(scene.sim.features, Features::default());
}

#[test]
fn scene_overrides_reach_nested_fields() {
    let s = wall_impact(10.0, "w");
    let t = s.with_override("sim.stiffness_scale", "0.25").unwrap();
    assert_eq!(t.sim.stiffness_scale, 0.25);
    let t = s.with_override("impactors.0.velocity", "[0, 3, 0]").unwrap();
    assert_eq!(t.impactors[0].velocity, [0.0, 3.0, 0.0]);
    let t = s.with_override("sim.solve_mode", "adaptive").unwrap();
    assert_eq!(t.sim.solve_mode, SolveMode::Adaptive);
    let t = s.with_override("sim.features.softening", "false").unwrap();
    assert!(!t.sim.features.softening);
    assert!(s.with_override("impactors.7.velocity", "[0, 3, 0]").is_err());
    assert!(s.with_override("sim.stiffness_scale", "\"soft\"").is_err());
}

/// Without softening a bond breaks the instant it reaches its strength and dissipates
/// only the elastic energy it stored (a threshold model); with it, exactly `G_f A`.
#[test]
fn softening_off_breaks_at_strength_and_dissipates_only_the_stored_energy() {
    let on = bond_tension();
    let off = on.with_feature("softening", false).unwrap();
    let m = Material::analytic_test();
    let (h, area) = (0.05, 0.01);
    let r_on = common::run(&on);
    let r_off = common::run(&off);
    // Same strength: the crack starts at the same moment of the force ramp.
    let onset = |o: &stress_ref::observation::Observation| o.values["first_crack_time"];
    assert!(common::rel(onset(&r_off), onset(&r_on)) < 1e-3, "{} vs {}", onset(&r_off), onset(&r_on));
    let gf = m.fracture_energy.tension * area;
    let stored = m.tensile_strength.powi(2) * area * 2.0 * h / (2.0 * m.youngs_modulus);
    assert!(common::rel(r_on.values["bond_dissipation"], gf) < 0.02);
    assert!(common::rel(r_off.values["bond_dissipation"], stored) < 0.05, "{} vs {stored}", r_off.values["bond_dissipation"]);
}

/// Without rigid-motion loads a spinning bar feels no centripetal tension.
#[test]
fn rigid_motion_loads_off_misses_the_centripetal_tension() {
    let omega = 20.0;
    let mut s = new_scene("spin", "");
    s.gravity = [0.0; 3];
    s.materials.insert("m".into(), Material { damping_ratio: 0.2, ..Material::analytic_test() });
    let chunks = grid(Vec3::new(-0.5, -0.05, -0.05), [10, 1, 1], Vec3::splat(0.1), "m");
    let bonds = auto_bonds(&chunks, |_, _| "m".into());
    let mut b = body("bar", chunks, bonds);
    b.angular_velocity = [0.0, 0.0, omega];
    s.bodies.push(b);
    s.sim.duration = 0.3;
    s.sim.fracture = false;
    let tension = |scene: &Scene| {
        let mut w = World::new(scene);
        w.run();
        w.solver.bonds[0].iter().find(|b| b.geometry.a == 4 && b.geometry.b == 5).unwrap().force.lin.z
    };
    let m = 2000.0 * 0.001;
    let expected: f64 = (5..10).map(|i| m * omega * omega * ((i as f64 - 4.5) * 0.1)).sum();
    assert!(common::rel(tension(&s), expected) < 0.01);
    let off = tension(&s.with_feature("rigid_motion_loads", false).unwrap());
    assert!(off.abs() < 0.02 * expected, "tension {off} without frame loads");
}

/// Without rate effects a fast-loaded concrete bond fails at its static strength.
#[test]
fn rate_effects_off_fails_fast_loading_at_the_static_strength() {
    let m = Material { weibull_modulus: None, static_fatigue: None, damping_ratio: 0.0, ..Material::concrete() };
    let h = 0.05;
    let f_static = m.tensile_strength * 4.0 * h * h;
    let onset = |scene: &Scene| {
        let mut w = World::new(scene);
        loop {
            let before = w.solver.bonds[0][0].measures.tension;
            w.step_frame();
            if w.solver.bonds[0][0].joint.damage > 0.0 {
                return before / m.tensile_strength;
            }
            assert!(w.solver.time < 0.01, "never failed");
        }
    };
    let ramp = 2e-4;
    let mut s = bond_pull("fast", m.clone(), h, TimeFunction::Ramp { t0: 0.0, t1: ramp, value: 3.0 * f_static }, ramp + 3e-3);
    s.sim.frame_dt = 1e-6;
    let with = onset(&s);
    let without = onset(&s.with_feature("rate_effects", false).unwrap());
    println!("onset with DIF {with}, without {without}");
    assert!(with > 1.1);
    assert!((without - 1.0).abs() < 0.02, "{without}");
}

/// Without static fatigue a bond held at 97% of its strength never fails.
#[test]
fn static_fatigue_off_holds_a_sustained_load_forever() {
    let m = Material {
        static_fatigue: Some(StaticFatigue { exponent: 20.0, test_time: 10.0 }),
        dif: None,
        damping_ratio: 0.05,
        ..Material::analytic_test()
    };
    let load = TimeFunction::Constant { value: 0.97 * m.tensile_strength * 0.01 };
    let s = bond_pull("fatigue", m, 0.05, load, 1.0);
    let events = |scene: &Scene| {
        let mut w = World::new(scene);
        w.run();
        w.solver.events.len()
    };
    assert!(events(&s) > 0);
    assert_eq!(events(&s.with_feature("static_fatigue", false).unwrap()), 0);
}

#[test]
fn weibull_off_gives_every_bond_the_mean_strength() {
    let s = pressure_panel_weibull(8.0, "weibull_panel");
    let spread = |scene: &Scene| {
        let st = Structure::from_scene(scene, 0);
        let w: Vec<f64> = st.bonds.iter().map(|b| b.weibull).collect();
        w.iter().copied().fold(0.0, f64::max) - w.iter().copied().fold(f64::INFINITY, f64::min)
    };
    assert!(spread(&s) > 0.3);
    assert_eq!(spread(&s.with_feature("weibull", false).unwrap()), 0.0);
}

/// A fully cracked joint pressed closed carries the compression through its contact
/// patch; without crack contact it carries nothing.
#[test]
fn crack_contact_off_leaves_a_cracked_joint_carrying_nothing() {
    let s = bond_tension();
    let structure = Structure::from_scene(&s, 0);
    let g: &BondGeometry = &structure.bonds[0].geometry;
    let m = Material::analytic_test();
    let mut cracked = JointState::new();
    cracked.damage = 1.0;
    let closing = Local6 { lin: Vec3::new(0.0, 0.0, -1e-5), ang: Vec3::ZERO };
    let force = |features: &Features| {
        let (stiffness, strength, _, _) = bond_physics(g, &m, None, None, 1.0, 1.0, features, &Methods::default());
        let model = JointModel { geometry: g, stiffness: &stiffness, strength: &strength, rebar: None, weibull: 1.0 };
        model.evaluate(&cracked, &closing, 0.0, true).force.lin.z
    };
    let on = force(&Features::default());
    let off = force(&Features { crack_contact: false, ..Features::default() });
    assert!(on < 0.0, "contact carries compression: {on}");
    assert_eq!(off, 0.0);
}

#[test]
fn damping_buckling_rebar_and_crushing_switches_remove_their_terms() {
    let s = bond_tension();
    let structure = Structure::from_scene(&s, 0);
    let g = &structure.bonds[0].geometry;
    let m = Material { damping_ratio: 0.05, ..Material::analytic_test() };
    let rebar = (1e-4, Material::steel());
    let off = Features { damping: false, buckling: false, rebar: false, crushing: false, ..Features::default() };
    let (_, strength, rb, damping) = bond_physics(g, &m, Some(2.0), Some(&rebar), 1.0, 1.0, &Features::default(), &Methods::default());
    assert!(strength.buckling_load.is_some() && rb.is_some() && damping.lin.z > 0.0);
    let (_, strength, rb, damping) = bond_physics(g, &m, Some(2.0), Some(&rebar), 1.0, 1.0, &off, &Methods::default());
    assert!(strength.buckling_load.is_none() && rb.is_none() && damping.lin.z == 0.0);
    assert!(strength.compressive.is_infinite(), "no crushing");
}
