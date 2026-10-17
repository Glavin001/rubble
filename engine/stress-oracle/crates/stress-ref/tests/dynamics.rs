//! Loads and stress dynamics: gravity prestress, frame-rate independence, timestep and
//! chunk-size convergence, solve modes, debris landing, stiffness scaling.

mod common;

use stress_ref::builders::*;
use stress_ref::material::Material;
use stress_ref::math::Vec3;
use stress_ref::metrics::MetricValue;
use stress_ref::scene::*;
use stress_ref::world::World;

/// A supported structure prestressed under gravity starts in equilibrium: no shock,
/// no motion, no damage.
#[test]
fn gravity_prestress_starts_structures_at_rest() {
    let mut s = wall_impact(0.0, "rest");
    s.impactors.clear();
    s.probes.clear();
    s.metrics.clear();
    s.sim.duration = 0.1;
    let mut w = World::new(&s);
    w.run();
    let mut vmax: f64 = 0.0;
    for c in 0..w.solver.structures[0].chunks.len() {
        vmax = vmax.max(w.solver.chunk_velocity(0, c).0.norm());
    }
    assert!(vmax < 1e-6, "max chunk velocity {vmax}");
    assert!(w.solver.events.is_empty());
    // The supports carry the whole weight (their own chunks included).
    let weight: f64 = w.solver.structures[0].chunks.iter().map(|c| c.mass * 9.81).sum();
    let reaction: f64 = (0..w.solver.structures[0].chunks.len()).map(|c| w.solver.reaction_world(0, c).0.z).sum();
    assert!(common::rel(reaction, weight) < 1e-6, "{reaction} vs {weight}");
}

/// The same scene at 30, 60 and 120 frames per second gives the same outcome.
#[test]
fn same_outcome_at_different_frame_rates() {
    let mut results = Vec::new();
    for fps in [30.0, 60.0, 120.0] {
        let mut s = pressure_panel(300e3, "fps");
        s.sim.frame_dt = 1.0 / fps;
        s.sim.duration = 4.0 / 60.0;
        s.sim.sample_interval = Some(2.5e-4);
        let obs = common::run(&s);
        results.push((
            common::metric_value(&s, &obs, "breach"),
            common::metric_number(&s, &obs, "fragment_speed"),
            obs.values["broken_bonds"],
        ));
    }
    println!("{results:?}");
    for r in &results[1..] {
        assert_eq!(r.0, results[0].0);
        assert!(common::rel(r.1, results[0].1) < 0.10, "fragment speed {} vs {}", r.1, results[0].1);
        assert!(common::rel(r.2, results[0].2) < 0.10, "broken bonds {} vs {}", r.2, results[0].2);
    }
}

/// Halving the substep changes the wave speed by well under a percent and the
/// fracture metrics of the breached panel by less than 10% (benchmark 10, time).
#[test]
fn converges_as_the_timestep_shrinks() {
    let base = bar_wave(1.0, "bar");
    let obs = common::run(&base);
    let c1 = common::metric_number(&base, &obs, "wave_speed");
    let dt = World::new(&base).substep_dt();
    let mut fine = base.clone();
    fine.sim.max_substep = Some(dt / 2.0);
    let c2 = common::metric_number(&fine, &common::run(&fine), "wave_speed");
    assert!(common::rel(c2, c1) < 0.005, "{c2} vs {c1}");

    let panel = pressure_panel(300e3, "panel");
    let o1 = common::run(&panel);
    let dt = World::new(&panel).substep_dt();
    let mut fine = panel.clone();
    fine.sim.max_substep = Some(dt / 2.0);
    let o2 = common::run(&fine);
    assert_eq!(common::metric_value(&panel, &o1, "breach"), common::metric_value(&fine, &o2, "breach"));
    let (v1, v2) = (common::metric_number(&panel, &o1, "fragment_speed"), common::metric_number(&fine, &o2, "fragment_speed"));
    assert!(common::rel(v2, v1) < 0.10, "fragment speed {v2} vs {v1}");
}

/// Same wall, two chunk sizes: the elastic response and the fracture outcome agree
/// (benchmark 10, chunk size; showcase "same wall, two chunk sizes").
#[test]
fn same_panel_two_chunk_sizes_same_outcome() {
    for (peak, breach) in [(10e3, false), (300e3, true)] {
        let coarse = pressure_panel_sized(peak, 0.1, "coarse");
        let fine = pressure_panel_sized(peak, 0.05, "fine");
        let (oc, of) = (common::run(&coarse), common::run(&fine));
        assert_eq!(common::metric_value(&coarse, &oc, "breach"), MetricValue::Bool(breach));
        assert_eq!(common::metric_value(&fine, &of, "breach"), MetricValue::Bool(breach));
        let (dc, df) = (
            common::metric_number(&coarse, &oc, "peak_center_displacement"),
            common::metric_number(&fine, &of, "peak_center_displacement"),
        );
        println!("peak {peak}: centre displacement {dc} vs {df}");
        if !breach {
            assert!(common::rel(df, dc) < 0.10, "elastic response {df} vs {dc}");
        } else {
            // Reported, not gated: two layers through the thickness resolve hinge
            // crushing and delamination that one layer cannot (see README, benchmark 10).
            let (vc, vf) = (common::metric_number(&coarse, &oc, "fragment_speed"), common::metric_number(&fine, &of, "fragment_speed"));
            println!("fragment speed {vc} vs {vf} ({:.0}% change)", 100.0 * common::rel(vf, vc));
        }
    }
}

/// Quasi-static and adaptive solves agree with the explicit solve where they apply.
#[test]
fn solve_modes_agree() {
    // Static deflection: quasi-static == explicit prestress (both equilibrium).
    let explicit = cantilever(20, "c");
    let mut qs = explicit.clone();
    qs.sim.solve_mode = SolveMode::QuasiStatic;
    qs.sim.duration = 0.03;
    qs.loads[0] = LoadDesc::PointForce {
        body: "beam".into(),
        chunk: 20,
        direction: [0.0, 0.0, -1.0],
        magnitude: TimeFunction::Constant { value: 30.0 },
        point: None,
    };
    let tip = |o: &stress_ref::observation::Observation| *o.probes["tip"].v.last().unwrap();
    let (oe, oq) = (common::run(&explicit), common::run(&qs));
    let static_tip = oe.probes["tip"].v[0];
    assert!(common::rel(tip(&oq), static_tip) < 1e-9, "{} vs {static_tip}", tip(&oq));

    // Adaptive: the breached panel behaves as in the explicit solve (the pulse wakes it).
    let panel = pressure_panel(300e3, "panel");
    let mut adaptive = panel.clone();
    adaptive.sim.solve_mode = SolveMode::Adaptive;
    let (o1, o2) = (common::run(&panel), common::run(&adaptive));
    assert_eq!(common::metric_value(&panel, &o1, "breach"), common::metric_value(&adaptive, &o2, "breach"));
    let (b1, b2) = (o1.values["broken_bonds"], o2.values["broken_bonds"]);
    assert!(common::rel(b2, b1) < 0.10, "{b2} vs {b1}");
    // ...and an idle structure in adaptive mode is settled (no explicit work).
    let mut idle = panel.clone();
    idle.loads.clear();
    idle.sim.solve_mode = SolveMode::Adaptive;
    let mut w = World::new(&idle);
    w.run();
    assert!(w.solver.clusters.iter().all(|c| c.activity == stress_ref::solver::Activity::Settled));
}

/// Debris landing on an intact beam loads it dynamically: the peak support reaction
/// exceeds the block's resting weight by far.
#[test]
fn debris_landing_loads_structure_dynamically() {
    let mut s = new_scene("landing", "block dropped on a fixed-fixed beam");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    let counts = [12, 2, 1];
    let mut chunks = grid(Vec3::new(-0.6, -0.1, 0.0), counts, Vec3::splat(0.1), "concrete");
    for j in 0..2 {
        chunks[grid_index(counts, 0, j, 0)].support = Support::Fixed;
        chunks[grid_index(counts, 11, j, 0)].support = Support::Fixed;
        chunks[grid_index(counts, 0, j, 0)].groups.push("ends".into());
        chunks[grid_index(counts, 11, j, 0)].groups.push("ends".into());
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("beam", chunks, bonds));
    let block = grid(Vec3::new(-0.1, -0.1, 0.6), [2, 2, 2], Vec3::splat(0.1), "concrete");
    let bb = auto_bonds(&block, |_, _| "concrete".into());
    s.bodies.push(body("block", block, bb));
    s.sim.duration = 0.45;
    s.sim.fracture = false;
    s.sim.frame_dt = 1e-3;
    s.sim.sample_interval = Some(1e-4);
    s.probes.push(probe("reaction", ProbeKind::Reaction { body: "beam".into(), chunks: ChunkSelector::Group("ends".into()), axis: [0.0, 0.0, 1.0] }));
    let obs = common::run(&s);
    let r = &obs.probes["reaction"];
    // Reactions include the supported chunks' own weight.
    let beam_weight = 24.0 * 2.4 * 9.81;
    let block_weight = 8.0 * 2.4 * 9.81;
    let peak = r.range_after(0.0).unwrap().1;
    println!("peak reaction {peak}, static {}", beam_weight + block_weight);
    assert!(common::rel(r.v[0], beam_weight) < 1e-3);
    assert!(peak - beam_weight > 3.0 * block_weight, "dynamic landing load {}", peak - beam_weight);
}

/// Stiffness scaling: with the scaled modulus the stress still crosses the structure
/// within two frames, and fracture (stress based) still happens at the same load.
#[test]
fn stiffness_scaling_keeps_strength_and_crossing_time() {
    let m = Material::analytic_test();
    for scale in [1.0, 0.01] {
        let mut s = bond_tension();
        s.sim.stiffness_scale = scale;
        let obs = common::run(&s);
        assert!(common::rel(common::metric_number(&s, &obs, "failure_force"), m.tensile_strength * 0.01) < 0.05);
    }
    let s = wall_impact(10.0, "w");
    let scale = stress_ref::world::min_stiffness_scale(&s, 2.0);
    let c = (s.materials["concrete"].youngs_modulus * scale / s.materials["concrete"].density).sqrt();
    let crossing = 2.1 / c;
    assert!(crossing <= 2.0 * s.sim.frame_dt * 1.0001 && scale < 1.0, "scale {scale}, crossing {crossing}");
}

/// Rigid-body motion is subtracted from chunk loads: a cluster in free fall carries no
/// stress, and a spinning bar's centre bond carries exactly the centripetal tension
/// `sum m w^2 r` of the chunks on one side (linear, angular and centrifugal terms).
#[test]
fn rigid_motion_is_subtracted_from_chunk_loads() {
    // Free fall from rest under gravity: no bond force at all.
    let mut s = new_scene("fall", "");
    s.materials.insert("m".into(), Material { damping_ratio: 0.05, ..Material::analytic_test() });
    let chunks = grid(Vec3::new(-0.5, -0.05, 2.0), [10, 1, 1], Vec3::splat(0.1), "m");
    let bonds = auto_bonds(&chunks, |_, _| "m".into());
    s.bodies.push(body("bar", chunks, bonds));
    s.sim.duration = 0.2;
    s.sim.fracture = false;
    let mut w = World::new(&s);
    w.run();
    let max_force = w.solver.bonds[0].iter().map(|b| b.force.lin.norm()).fold(0.0, f64::max);
    assert!(max_force < 1e-6, "free fall stressed the bar: {max_force} N");

    // Spinning about its centre (z axis): settle the centripetal tension with damping.
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
    let mut w = World::new(&s);
    w.run();
    let centre = w.solver.bonds[0].iter().find(|b| b.geometry.a == 4 && b.geometry.b == 5).unwrap();
    let m = w.solver.structures[0].chunks[0].mass;
    let expected: f64 = (5..10).map(|i| m * omega * omega * ((i as f64 - 4.5) * 0.1)).sum();
    let tension = centre.force.lin.z;
    println!("centre tension {tension} expected {expected}");
    assert!(common::rel(tension, expected) < 0.01, "{tension} vs {expected}");
}
