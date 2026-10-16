//! The real-time alternatives to the explicit ground truth, measured against it and
//! against closed forms: the implicit Newmark step (inertia and true stiffness at a
//! large step) and selective mass scaling (instead of lowering the modulus).

mod common;

use stress_ref::builders::*;
use stress_ref::metrics::MetricValue;
use stress_ref::scene::*;
use stress_ref::world::World;

/// Damped single-degree-of-freedom response to a suddenly applied (released) load:
/// peak over static `1 + exp(-zeta pi / sqrt(1 - zeta^2))`.
fn damped_amplification(zeta: f64) -> f64 {
    1.0 + (-zeta * std::f64::consts::PI / (1.0 - zeta * zeta).sqrt()).exp()
}

fn amplification(scene: &Scene) -> f64 {
    let obs = common::run(scene);
    match common::metric_value(scene, &obs, "dynamic_amplification") {
        MetricValue::Number(x) => x,
        other => panic!("{other:?}"),
    }
}

/// Sudden support loss with the implicit step: the dynamic amplification converges to
/// the damped closed form as the step resolves the period (T = 0.89 ms here); a step
/// longer than the period misses most of it. The explicit solve is the reference.
#[test]
fn implicit_amplification_converges_as_the_step_resolves_the_period() {
    let sudden = support_loss(0.0, "b4_sudden");
    let exact = damped_amplification(0.02);
    let explicit = amplification(&sudden);
    println!("closed form {exact:.4}, explicit {explicit:.4}");
    assert!(common::rel(explicit, exact) < 0.01);
    let mut errors = Vec::new();
    for step in [1e-3, 2e-4, 5e-5, 2e-5] {
        let mut s = sudden.clone();
        s.sim.solve_mode = SolveMode::Implicit;
        s.sim.implicit_dt = Some(step);
        let a = amplification(&s);
        println!("implicit dt {step:e}: {a:.4}");
        errors.push(common::rel(a, exact));
    }
    assert!(errors[0] > 0.3, "one step per period cannot show the amplification");
    assert!(errors.windows(2).all(|w| w[1] < w[0]), "monotone convergence: {errors:?}");
    assert!(errors[3] < 0.02, "{errors:?}");
}

/// Without damping the implicit (trapezoidal) step neither gains nor loses energy:
/// a released column keeps oscillating between the same extremes.
#[test]
fn implicit_step_conserves_the_energy_of_free_vibration() {
    let mut s = support_loss(0.0, "undamped");
    let rock = s.materials.get_mut("rock").unwrap();
    rock.damping_ratio = 0.0;
    s.sim.solve_mode = SolveMode::Implicit;
    s.sim.implicit_dt = Some(5e-5);
    s.sim.duration = 0.05;
    let obs = common::run(&s);
    let series = &obs.probes["reaction"];
    let late: Vec<f64> = series.t.iter().zip(&series.v).filter(|(t, _)| **t > 0.03).map(|(_, v)| *v).collect();
    let early: Vec<f64> = series.t.iter().zip(&series.v).filter(|(t, _)| **t > 0.01 && **t < 0.02).map(|(_, v)| *v).collect();
    let range = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max) - v.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(common::rel(range(&late), range(&early)) < 0.01, "{} vs {}", range(&late), range(&early));
}

/// In the static limit the implicit step is the quasi-static solve: a slowly loaded
/// cantilever reaches the Euler-Bernoulli deflection, and the frame removal matches
/// the explicit solve within the oracle tolerance.
#[test]
fn implicit_step_matches_the_explicit_solve_on_structural_benchmarks() {
    let mut beam = cantilever(20, "c");
    beam.sim.solve_mode = SolveMode::Implicit;
    let obs = common::run(&beam);
    let tip = common::metric_number(&beam, &obs, "tip_deflection");
    let expected = beam.metrics.iter().find(|m| m.name == "tip_deflection").unwrap().expected.clone().unwrap().as_f64().unwrap();
    assert!(common::rel(tip, expected) < 0.01, "{tip} vs {expected}");

    let frame = frame_column_removal_with(0.0, false, "frame_elastic");
    let mut implicit = frame.clone();
    implicit.sim.solve_mode = SolveMode::Implicit;
    let (oe, oi) = (common::run(&frame), common::run(&implicit));
    for m in ["axial_col0_peak", "axial_col2_peak", "beam_moment_peak"] {
        let (e, i) = (common::metric_number(&frame, &oe, m), common::metric_number(&implicit, &oi, m));
        println!("{m}: explicit {e:.4e}, implicit {i:.4e}");
        assert!(common::rel(i, e) < 0.10, "{m}: {i} vs {e}");
    }
}

/// Selective mass scaling raises the stable substep to the target by adding inertia
/// only where needed: statics are untouched, the added mass is reported, and the
/// substep count falls.
#[test]
fn mass_scaling_raises_the_substep_without_touching_statics() {
    let beam = cantilever(40, "c40");
    let w0 = World::new(&beam);
    let dt0 = w0.solver.stable_dt();
    let mut scaled = beam.clone();
    scaled.sim.mass_scaling_dt = Some(4.0 * dt0);
    let w1 = World::new(&scaled);
    let dt1 = w1.solver.stable_dt();
    println!("stable dt {dt0:.3e} -> {dt1:.3e}, added mass {:.1}%", 100.0 * w1.added_mass_fraction);
    // Exactly: the scaling meets its target substep (no allowance).
    assert!(dt1 >= 4.0 * dt0, "{dt1} vs {}", 4.0 * dt0);
    assert!(w1.added_mass_fraction > 0.0);
    let (o0, o1) = (common::run(&beam), common::run(&scaled));
    let tip = |o: &stress_ref::observation::Observation| o.probes["tip"].v[0];
    assert!(common::rel(tip(&o1), tip(&o0)) < 1e-9, "static deflection changed");
    assert_eq!(o1.values["added_mass_fraction"], w1.added_mass_fraction);
    assert!(o1.values["substeps"] < 0.5 * o0.values["substeps"]);
}

/// b5's wall (2 m x 2.1 m, two 0.1 m concrete chunks thick, anchored at its base) and
/// its 1000 kg steel box ram at `speed`, starting `gap` from the wall.
fn wall_and_ram(speed: f64, gap: f64) -> Scene {
    let mut s = new_scene("wall_ram", "b5's wall and ram");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    let counts = [20, 2, 21];
    let mut chunks = grid(stress_ref::math::Vec3::new(-1.0, 0.0, -0.1), counts, stress_ref::math::Vec3::splat(0.1), "concrete");
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            chunks[grid_index(counts, i, j, 0)].support = Support::Fixed;
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    s.bodies.push(body("wall", chunks, bonds));
    s.impactors.push(ImpactorDesc {
        name: "ram".into(),
        shape: ImpactorShape::Box { half_extents: [0.2, 0.25, 0.2] },
        mass: 1000.0,
        position: [0.0, -0.25 - gap, 1.0],
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: [0.0, speed, 0.0],
        angular_velocity: [0.0; 3],
        material: "steel".into(),
        crush: None,
    });
    s
}

/// Chunks off the anchored body, and how many of them move faster than `speed`.
fn detached(w: &World, speed: f64) -> (usize, usize) {
    let off = w.solver.clusters.iter().filter(|c| !c.anchored);
    (off.clone().map(|c| c.chunks.len()).sum(), off.filter(|c| c.velocity.norm() > speed).map(|c| c.chunks.len()).sum())
}

/// `implicit_contact_windows`: implicit mode takes an impact's frames in the explicit
/// integrator, so it punches through where the explicit solve does (b5's wall at
/// 20 m/s: 32 chunks out at about the ram's speed), and the contact ledger books no
/// energy made by contacts.
#[test]
fn contact_windows_resolve_punch_through_as_the_explicit_solve() {
    let mut scene = wall_and_ram(20.0, 0.001);
    scene.sim.duration = 0.05;
    let run = |mode: SolveMode, windows: bool| {
        let mut s = scene.clone();
        s.sim.solve_mode = mode;
        s.sim.methods.implicit_contact_windows = windows;
        s.sim.methods.implicit_failure_windows = windows;
        let mut w = World::new(&s);
        let o = w.run();
        (detached(&w, 10.0), o.values["contact_dissipation"])
    };
    let (explicit, implicit, windowed) = (run(SolveMode::Explicit, false), run(SolveMode::Implicit, false), run(SolveMode::Implicit, true));
    println!("detached (all, fast) and contact dissipation: explicit {explicit:?}, implicit {implicit:?}, windows {windowed:?}");
    assert_eq!(explicit.0, (32, 32));
    assert_eq!(windowed.0, explicit.0);
    assert!(windowed.1 >= 0.0, "contacts made energy: {}", windowed.1);
}

/// A window lasts only while a body can strike: a ram receding from the wall opens it
/// and, once out of reach, the wall returns to the implicit step.
#[test]
fn contact_window_closes_when_the_body_is_out_of_reach() {
    let mut s = wall_and_ram(-20.0, 0.05);
    s.sim.solve_mode = SolveMode::Implicit;
    s.sim.methods.implicit_contact_windows = true;
    let mut w = World::new(&s);
    let wall = |w: &World| w.solver.clusters.iter().position(|c| c.anchored).expect("wall");
    w.step_frame();
    assert!(w.solver.clusters[wall(&w)].explicit_window, "the ram starts within reach");
    for _ in 0..10 {
        w.step_frame();
    }
    assert!(!w.solver.clusters[wall(&w)].explicit_window, "the ram is metres away");
    assert_eq!(w.solver.clusters.len(), 1, "nothing touched the wall");
}

/// `implicit_failure_windows`: b9's blast breach runs in the explicit integrator from
/// the frame its first bond fails (that frame is solved again), so implicit mode breaks
/// the bonds the explicit solve breaks and the fragments leave at its speed. The
/// elastic panel (b9_panel_low) never fails and stays in the implicit step, unchanged.
#[test]
fn failure_windows_resolve_the_breach_as_the_explicit_solve() {
    let run = |scene: &Scene, mode: SolveMode, windows: bool| {
        let mut s = scene.clone();
        s.sim.solve_mode = mode;
        s.sim.methods.implicit_failure_windows = windows;
        s.sim.methods.implicit_contact_windows = windows;
        let mut o = common::run(&s);
        let speed = common::metric_number(&s, &o, "fragment_speed");
        o.values.remove("redone_frames");
        (o.values["broken_bonds"], speed, o)
    };
    let high = pressure_panel(300e3, "b9_panel_high");
    let (explicit, implicit, windowed) = (run(&high, SolveMode::Explicit, false), run(&high, SolveMode::Implicit, false), run(&high, SolveMode::Implicit, true));
    println!("broken bonds, fragment speed: explicit {} {:.4}, implicit {} {:.4}, windows {} {:.4}", explicit.0, explicit.1, implicit.0, implicit.1, windowed.0, windowed.1);
    assert_eq!(windowed.0, explicit.0);
    assert!(common::rel(windowed.1, explicit.1) < 0.01, "{} vs {}", windowed.1, explicit.1);

    let low = pressure_panel(10e3, "b9_panel_low");
    let (plain, windowed) = (run(&low, SolveMode::Implicit, false), run(&low, SolveMode::Implicit, true));
    assert_eq!(windowed.0, 0.0);
    assert_eq!(windowed.2.values, plain.2.values);
}

/// The hand-over between the explicit window and the implicit step is consistent: an
/// undamped stack in free vibration after its support goes, with windows every other
/// or every third frame, keeps its energy (mechanical plus dissipated) to the explicit
/// integrator's own frame-end fluctuation, relative to the vibration's energy (the
/// implicit step alone holds it to round-off).
#[test]
fn window_hand_over_keeps_the_energy_of_free_vibration() {
    let mut s = support_loss(0.0, "undamped");
    s.materials.get_mut("rock").unwrap().damping_ratio = 0.0;
    s.sim.solve_mode = SolveMode::Implicit;
    s.sim.implicit_dt = Some(5e-5);
    // Windows set by hand below, not by the switches.
    s.sim.methods.implicit_contact_windows = false;
    s.sim.methods.implicit_failure_windows = false;
    // Spread of the energy over the vibration (after the removal at 10 ms), over the
    // vibration's largest energy.
    let spread = |window_every: Option<usize>| {
        let mut w = World::new(&s);
        let (mut lo, mut hi, mut vibration) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
        for frame in 0..50 {
            if let Some(k) = window_every {
                for cl in &mut w.solver.clusters {
                    cl.explicit_window = frame % k == 0;
                }
            }
            w.step_frame();
            if frame >= 10 {
                let e = w.mechanical_energy() + w.dissipated_energy();
                (lo, hi) = (lo.min(e), hi.max(e));
                vibration = vibration.max(w.solver.kinetic_energy() + w.solver.elastic_energy());
            }
        }
        (hi - lo) / vibration
    };
    let (implicit, explicit, every_other, every_third) = (spread(None), spread(Some(1)), spread(Some(2)), spread(Some(3)));
    println!("energy spread / vibration energy: implicit {implicit:.3e}, explicit {explicit:.3e}, windows every 2nd frame {every_other:.3e}, every 3rd {every_third:.3e}");
    assert!(implicit < 1e-6, "{implicit}");
    assert!(every_other < 2.5 * explicit, "{every_other} vs {explicit}");
    assert!(every_third < 2.5 * explicit, "{every_third} vs {explicit}");
}

/// The quasi-static solve's energy is booked (`static_discard`), and where its solves
/// converge it never creates energy. A static step minimises the total potential
/// (stored energy less the work of the loads it solves under), so what it removes,
/// net of the loads' work over its displacement, is non-negative.
#[test]
fn a_converged_static_solve_never_creates_energy() {
    for scene in [cantilever(20, "c"), frame_column_removal_with(0.0, false, "frame_elastic")] {
        let mut s = scene.clone();
        s.sim.solve_mode = SolveMode::QuasiStatic;
        let o = common::run(&s);
        let (discard, unconverged) = (o.values["static_discard"], o.values.get("static_unconverged_frames").copied().unwrap_or(0.0));
        println!("{}: static_discard {discard:.4e} J, unconverged frames {unconverged}", s.name);
        assert_eq!(unconverged, 0.0, "{}", s.name);
        assert!(discard >= 0.0, "{}: {discard}", s.name);
    }
}

/// Damping ratio of a decaying oscillation at angular frequency `w`: the ratio of its
/// ranges over `[0.1, 0.5]` s and `[3.5, 4.0]` s is `exp(zeta w 3.4)`. (A range, not
/// peaks: between implicit steps the probe samples the predictor.)
fn envelope_ratio(t: &[f64], v: &[f64], w: f64) -> f64 {
    let range = |a: f64, b: f64| {
        let xs: Vec<f64> = t.iter().zip(v).filter(|(ti, _)| **ti > a && **ti < b).map(|(_, x)| *x).collect();
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) - xs.iter().copied().fold(f64::INFINITY, f64::min)
    };
    (range(0.1, 0.5) / range(3.5, 4.0)).ln() / (w * 3.4)
}

/// `rayleigh_damping`: the cantilever's fundamental mode decays at the material's
/// damping ratio. The stiffness-proportional bond dashpots alone damp it at
/// `zeta w1 / w_b`, a fraction of a percent of that.
#[test]
fn rayleigh_damping_gives_the_fundamental_mode_its_damping_ratio() {
    for (mode, zeta) in [(SolveMode::Explicit, 0.02), (SolveMode::Explicit, 0.05), (SolveMode::Implicit, 0.02)] {
        let run = |rayleigh: bool| {
            let mut s = cantilever(20, "c");
            s.materials.get_mut("beam").unwrap().damping_ratio = zeta;
            s.sim.methods.rayleigh_damping = rayleigh;
            s.sim.solve_mode = mode;
            s.sim.duration = 4.0;
            let w = World::new(&s);
            let w1 = 2.0 * std::f64::consts::PI / w.solver.fundamental_period(0).expect("anchored cantilever");
            let o = common::run(&s);
            let tip = &o.probes["tip"];
            envelope_ratio(&tip.t, &tip.v, w1)
        };
        let (stiffness_only, rayleigh) = (run(false), run(true));
        println!("{mode:?}, material {zeta}: fundamental mode damping {stiffness_only:.3e} (bond dashpots only), {rayleigh:.4} (Rayleigh)");
        assert!(common::rel(rayleigh, zeta) < 0.1, "{mode:?}: {rayleigh} vs {zeta}");
        assert!(stiffness_only.abs() < 0.1 * zeta, "{mode:?}: {stiffness_only}");
    }
}
