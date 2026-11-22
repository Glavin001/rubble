//! The centred damping's correction must not depend on how a beam turns
//! (`ExplicitBeamsSettings::centred_objective`, jbeam-semantics-spec.md §39.6;
//! known-bugs S3).
//!
//! **The bug.** The correction took a beam's change of extension rate as the
//! change of its ends' relative velocity along the beam's direction at the end
//! of the step. A beam that only turns has no extension rate before or after,
//! but its relative velocity turns with it, so that reading is −ω²·L·h: every
//! damped beam of a spinning body was pushed outward each step and the body
//! gained energy. Measured on the CPU reference (2026-11-13): a free triangle
//! of three 0.1 kg nodes 9.5 cm apart (a hub cap; k 68,000 N/m, c 20 N·s/m,
//! its beams at ω·h 0.71) spun at 5 m/s at the rim went from 3.75 J to 1,800 J
//! in 0.75 s and was not finite at 1.5 s. In the crash lab a hub cap that left
//! its wheel went from 25 m/s to 3,700 m/s in 1.4 s, and the car with it.
//!
//! **The invariants** (from the physics, no recorded outputs): a free body's
//! kinetic + gravitational + elastic energy never rises; its angular momentum
//! is constant; a beam that only stretches along its own axis is damped as
//! the analytic oscillator, the same with either form; and the property
//! centred damping exists for (a damped mode stable past 2(√(1+ζ²) − ζ))
//! still holds.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{DVec3, Vec3};

/// A free body 30 m up (it falls 11 m in the longest run): `points` (m) joined by `beams`, every node
/// `kg`, every beam `k` N/m and `c` N·s/m, never yielding or breaking, no
/// contact. Centred damping on; `objective` picks the correction's form.
fn body(
    points: &[[f32; 3]],
    beams: &[(usize, usize)],
    kg: f32,
    k: f32,
    c: f32,
    objective: bool,
) -> ExplicitBeams {
    let nodes: Vec<String> = points
        .iter()
        .enumerate()
        .map(|(i, p)| format!(r#"["n{i}",{},{},{}]"#, p[0], p[1], p[2]))
        .collect();
    let rows: Vec<String> = beams
        .iter()
        .map(|(a, b)| format!(r#"["n{a}","n{b}"]"#))
        .collect();
    let text = format!(
        r#"{{"probe":{{"information":{{"name":"probe"}},"slotType":"main",
        "nodes":[["id","posX","posY","posZ"],{{"nodeWeight":{kg},"collision":false,"selfCollision":false}},{}],
        "beams":[["id1:","id2:"],{{"beamSpring":{k},"beamDamp":{c},"beamDeform":"FLT_MAX","beamStrength":"FLT_MAX"}},{}]}}}}"#,
        nodes.join(","),
        rows.join(",")
    );
    let desc = engine_softbody::jbeam::load_jbeam(&[("probe.jbeam", &text)], None)
        .expect("the probe's jbeam");
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Pressure,
        Vec3::Y * 30.0,
        Vec3::ZERO,
        settings,
    )
    .expect("spawn");
    assert!(
        s.settings.centred_damping,
        "JBeam content runs the BeamNG profile"
    );
    s.settings.centred_objective = objective;
    s
}

/// An equilateral triangle of side `side` in the plane x–y (JBeam frame).
fn triangle(side: f32) -> (Vec<[f32; 3]>, Vec<(usize, usize)>) {
    let r = side / 3f32.sqrt();
    let p = (0..3)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 3.0;
            [r * a.cos(), r * a.sin(), 0.0]
        })
        .collect();
    (p, vec![(0, 1), (1, 2), (2, 0)])
}

/// A regular tetrahedron of edge `side`.
fn tetra(side: f32) -> (Vec<[f32; 3]>, Vec<(usize, usize)>) {
    let s = side / 8f32.sqrt();
    (
        vec![[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]],
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    )
}

fn centre(s: &ExplicitBeams) -> Vec3 {
    s.nodes.iter().map(|n| n.pos).sum::<Vec3>() / s.nodes.len() as f32
}

/// Spin about `axis` through the centre at `rim` m/s at the farthest node,
/// on top of a uniform velocity `drift`.
fn spin(s: &mut ExplicitBeams, axis: Vec3, rim: f32, drift: Vec3) {
    let c = centre(s);
    let far = s
        .nodes
        .iter()
        .map(|n| axis.cross(n.pos - c).length())
        .fold(0.0f32, f32::max);
    for n in &mut s.nodes {
        n.vel = drift + axis.cross(n.pos - c) * (rim / far);
    }
}

/// Runs `s` for `seconds` with the energy ledger: (start energy less the
/// gravitational part, the largest rise of kinetic + gravitational + elastic
/// above its start, whether it stayed finite, the energy's change from 0.25 s
/// to the end).
fn run_late(s: ExplicitBeams, seconds: f32) -> (f64, f64, bool, f64) {
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let l = w.ledger.as_ref().unwrap();
    let (e0, own0) = (l.mechanical(), l.kinetic + l.elastic);
    let (mut rise, mut early, mut e) = (0.0f64, e0, e0);
    for k in 0..(seconds / STEP) as usize {
        w.step();
        e = w.ledger.as_ref().unwrap().mechanical();
        if !e.is_finite() {
            return (own0, f64::INFINITY, false, f64::INFINITY);
        }
        rise = rise.max(e - e0);
        if k == 499 {
            early = e;
        }
    }
    (own0, rise, true, e - early)
}

fn run(s: ExplicitBeams, seconds: f32) -> (f64, f64, bool) {
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.enable_ledger();
    let l = w.ledger.as_ref().unwrap();
    let (e0, own0) = (l.mechanical(), l.kinetic + l.elastic);
    let mut rise = 0.0f64;
    for _ in 0..(seconds / STEP) as usize {
        w.step();
        let e = w.ledger.as_ref().unwrap().mechanical();
        if !e.is_finite() {
            return (own0, f64::INFINITY, false);
        }
        rise = rise.max(e - e0);
    }
    (own0, rise, true)
}

/// **Fails without the fix:** the hub-cap triangle at 5 m/s went 3.75 J →
/// 1,800 J in 0.75 s. Asserted, at rim speeds from 0.05 to 0.46 rad a step,
/// for the cap, a door-sized triangle and a tetrahedron, about a principal
/// axis and tumbling, drifting or not:
/// * once the body has taken its spinning shape (0.25 s) its energy does not
///   rise: over the next 1.25 s by no more than a thousandth of its own
///   energy beyond what the plain step's does for the same body (single
///   precision: an undamped tetrahedron's energy wanders by 0.3 % in a
///   second on the plain step);
/// * before that it is never further above its start than the plain step
///   (centred damping off) takes the same body, plus a thousandth of its
///   energy. (The bodies are set spinning unstretched; at 50 m/s the cap's
///   beams stretch 40 % into their spinning shape, and the symplectic step's
///   energy moves with that shock in either scheme.)
#[test]
fn a_free_spinning_damped_body_never_gains_energy() {
    let cap = triangle(0.095);
    let door = triangle(0.6);
    let tet = tetra(0.3);
    let cases: [(&str, &(Vec<[f32; 3]>, Vec<(usize, usize)>), f32, f32, f32); 3] = [
        ("hub cap", &cap, 0.1, 68_000.0, 20.0),
        ("door-sized triangle", &door, 1.2, 1.4e6, 80.0),
        ("tetrahedron", &tet, 0.5, 5.0e5, 40.0),
    ];
    for (name, (pts, beams), kg, k, c) in cases {
        for rim in [5.0f32, 20.0, 50.0] {
            for drift in [Vec3::ZERO, Vec3::new(30.0, 0.0, 10.0)] {
                for tumbling in [false, true] {
                    let start = |centred: bool| {
                        let mut s = body(pts, beams, kg, k, c, true);
                        s.settings.centred_damping = centred;
                        let normal = (s.nodes[1].pos - s.nodes[0].pos)
                            .cross(s.nodes[2].pos - s.nodes[0].pos)
                            .normalize();
                        let axis = if tumbling {
                            (normal + Vec3::new(0.3, 0.2, 0.4)).normalize()
                        } else {
                            normal
                        };
                        spin(&mut s, axis, rim, drift);
                        s
                    };
                    let what = format!("{name} at {rim} m/s (drift {drift}, tumbling {tumbling})");
                    let (e0, rise, finite, late) = run_late(start(true), 1.5);
                    let (_, plain, _) = run(start(false), 0.25);
                    let (_, _, _, plain_late) = run_late(start(false), 1.5);
                    assert!(finite, "{what}: not finite");
                    // (A body drifting at 30 m/s: the single-precision kinetic
                    // energy of the plain step itself wanders by this much.)
                    assert!(
                        late <= plain_late.max(0.0) + 1e-3 * e0,
                        "{what}: energy rose {late:.4} J from 0.25 s to 1.5 s; the plain step {plain_late:.4} J (own energy {e0:.2} J)"
                    );
                    assert!(
                        rise <= plain + 1e-3 * e0,
                        "{what}: energy rose {rise:.3} J above its start; the plain step {plain:.3} J (own energy {e0:.2} J)"
                    );
                }
            }
        }
    }
}

/// The earlier form, kept as the record of the bug (setting off): the same
/// hub cap at 5 m/s gains more than a hundred times its energy in 0.75 s.
#[test]
fn the_end_direction_form_adds_energy_to_a_spinning_body() {
    let (pts, beams) = triangle(0.095);
    let mut s = body(&pts, &beams, 0.1, 68_000.0, 20.0, false);
    let normal = (s.nodes[1].pos - s.nodes[0].pos)
        .cross(s.nodes[2].pos - s.nodes[0].pos)
        .normalize();
    spin(&mut s, normal, 5.0, Vec3::ZERO);
    let (e0, rise, _) = run(s, 0.75);
    assert!(
        rise > 100.0 * e0,
        "the end-direction form no longer pumps the cap (rise {rise:.1} J of {e0:.2} J): update known-bugs S3"
    );
}

/// A free damped body's angular momentum is constant (its forces are equal,
/// opposite and along its beams). **Fails without the fix** (1.1e13 of its
/// start: the body flies apart). The triangle spins with one node knocked
/// inward, so its beams stretch and the damping acts.
#[test]
fn a_free_spinning_damped_body_keeps_its_angular_momentum() {
    let (pts, beams) = triangle(0.3);
    let mut s = body(&pts, &beams, 0.5, 4.0e5, 60.0, true);
    let normal = (s.nodes[1].pos - s.nodes[0].pos)
        .cross(s.nodes[2].pos - s.nodes[0].pos)
        .normalize();
    spin(&mut s, normal, 20.0, Vec3::ZERO);
    let c = centre(&s);
    let kick = (c - s.nodes[0].pos).normalize() * 3.0;
    s.nodes[0].vel += kick;
    let momentum = |s: &ExplicitBeams| {
        let (mut m, mut p, mut x) = (0.0f64, DVec3::ZERO, DVec3::ZERO);
        for n in &s.nodes {
            m += f64::from(n.mass);
            p += n.vel.as_dvec3() * f64::from(n.mass);
            x += (s.origin.as_dvec3() + n.pos.as_dvec3()) * f64::from(n.mass);
        }
        let (xc, vc) = (x / m, p / m);
        s.nodes.iter().fold(DVec3::ZERO, |l, n| {
            l + (s.origin.as_dvec3() + n.pos.as_dvec3() - xc).cross(n.vel.as_dvec3() - vc)
                * f64::from(n.mass)
        })
    };
    let l0 = momentum(&s);
    let (mut worst, mut settled, mut late) = (0.0f64, l0, 0.0f64);
    for k in 0..3000 {
        s.step();
        let l = momentum(&s);
        worst = worst.max((l - l0).length() / l0.length());
        if k == 1999 {
            settled = l;
        }
        if k > 1999 {
            late = late.max((l - settled).length() / l0.length());
        }
    }
    // Once the knock has rung down nothing acts on the momentum at all.
    assert!(
        late < 1e-5,
        "angular momentum still moving after the ring-down: {late:.2e}"
    );
    // While the beams ring, the correction acts along each beam's direction
    // at the end of the step on the lever of its start: a torque of second
    // order in h (1.2e-3 of the momentum over this ring-down, measured).
    assert!(
        worst < 5e-3,
        "angular momentum moved by {worst:.2e} of its start"
    );
}

/// Two nodes on one beam, stretched and let go: the extension's peaks, and
/// their times.
fn ring(k: f32, c: f32, objective: bool, steps: usize) -> Vec<(usize, f64)> {
    let mut s = body(
        &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        &[(0, 1)],
        1.0,
        k,
        c,
        objective,
    );
    let d = (s.nodes[1].pos - s.nodes[0].pos).normalize();
    let rest = f64::from(s.beams[0].rest);
    s.nodes[1].pos += d * 0.002;
    let mut e: Vec<f64> = Vec::new();
    for _ in 0..steps {
        s.step();
        e.push(f64::from((s.nodes[1].pos - s.nodes[0].pos).length()) - rest);
    }
    (1..e.len() - 1)
        .filter(|&i| e[i] > e[i - 1] && e[i] >= e[i + 1] && e[i] > 0.0)
        .map(|i| (i, e[i]))
        .collect()
}

/// A beam that only stretches along its own axis has nothing to turn: the
/// two forms agree, and both ring down as the analytic damped oscillator
/// (ω₀ = √(k/μ), ζ = c / (2√(kμ)), μ = m/2): period 2π / (ω₀√(1 − ζ²)),
/// log decrement 2πζ / √(1 − ζ²).
#[test]
fn an_axial_oscillation_is_damped_as_the_analytic_oscillator_in_both_forms() {
    let (k, zeta) = (2.0e4f32, 0.05f32);
    let c = 2.0 * zeta * (k * 0.5).sqrt();
    let w0 = f64::from((k / 0.5).sqrt());
    let z = f64::from(zeta);
    let period = std::f64::consts::TAU / (w0 * (1.0 - z * z).sqrt()) / f64::from(STEP);
    let decrement = std::f64::consts::TAU * z / (1.0 - z * z).sqrt();
    let mut peaks = Vec::new();
    for objective in [false, true] {
        let p = ring(k, c, objective, 700);
        assert!(p.len() >= 8, "{} peaks", p.len());
        let t = (p[7].0 - p[1].0) as f64 / 6.0;
        let d = (p[1].1 / p[7].1).ln() / 6.0;
        assert!(
            (t / period - 1.0).abs() < 0.01,
            "objective {objective}: period {t:.2} steps against {period:.2}"
        );
        assert!(
            (d / decrement - 1.0).abs() < 0.03,
            "objective {objective}: log decrement {d:.4} against {decrement:.4}"
        );
        peaks.push(p);
    }
    for (a, b) in peaks[0].iter().zip(&peaks[1]).take(8) {
        assert_eq!(a.0, b.0, "the forms' peaks fall on different steps");
        assert!(
            (a.1 / b.1 - 1.0).abs() < 1e-3,
            "the forms differ on an axial oscillation: {} against {}",
            a.1,
            b.1
        );
    }
}

/// What centred damping is for (§39): a damped mode past the plain step's
/// limit 2(√(1 + ζ²) − ζ) stays stable. At ω·h = 1.95 and ζ = 0.07 (limit
/// 1.865) the plain step diverges and the centred one rings down, in the
/// objective form as before.
#[test]
fn a_damped_mode_past_the_plain_limit_still_rings_down() {
    let (omega_h, zeta) = (1.95f32, 0.07f32);
    let k = (omega_h / STEP).powi(2) * 0.5;
    let c = 2.0 * zeta * (k * 0.5).sqrt();
    for objective in [false, true] {
        let p = ring(k, c, objective, 4000);
        let (first, last) = (p.first().unwrap().1, p.last().unwrap().1);
        assert!(
            last.is_finite() && last < 0.5 * first,
            "objective {objective}: {first:.2e} m to {last:.2e} m"
        );
    }
    // The plain step (centred damping off) at the same values: it grows.
    let mut s = body(
        &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        &[(0, 1)],
        1.0,
        k,
        c,
        false,
    );
    s.settings.centred_damping = false;
    let d = (s.nodes[1].pos - s.nodes[0].pos).normalize();
    let rest = s.beams[0].rest;
    s.nodes[1].pos += d * 0.002;
    let mut worst = 0.0f32;
    for _ in 0..4000 {
        s.step();
        let e = ((s.nodes[1].pos - s.nodes[0].pos).length() - rest).abs();
        if !e.is_finite() {
            worst = f32::INFINITY;
            break;
        }
        worst = worst.max(e);
    }
    assert!(
        worst > 0.02,
        "the plain step no longer diverges here ({worst} m): the case does not test §39"
    );
}
