//! Closed-form checks of the crash measuring kit on hand-built bodies and synthetic
//! samples (no car is rigged here; `engine-softbody-lab/tests/crash_metrics_s02.rs` runs
//! S02 itself).

use super::super::{BeamState, Body, Built, Contact, Layout, Sample, Section, Template, Templates, ZoneMap};
use super::*;
use crate::desc::{SoftBeam, SoftNode, SoftVehicleDesc};
use crate::explicit_beams::{jbeam_force, Beam, ExplicitBeams, ExplicitKind, Node};
use glam::{DMat3, Mat3};

fn node(name: &str, p: [f32; 3], mass: f32) -> SoftNode {
    SoftNode { name: name.into(), position: p, mass, ..Default::default() }
}

fn desc(nodes: Vec<SoftNode>, beams: &[(u32, u32)]) -> SoftVehicleDesc {
    SoftVehicleDesc { nodes, beams: beams.iter().map(|&(a, b)| SoftBeam { a, b, ..Default::default() }).collect(), ..Default::default() }
}

/// The design turned by `yaw` as `CrashScenario::build` spawns it.
fn turned(d: &SoftVehicleDesc, yaw: f32) -> SoftVehicleDesc {
    let (s, c) = yaw.sin_cos();
    let mut o = d.clone();
    for n in &mut o.nodes {
        let [x, y, z] = n.position;
        n.position = [x * c + z * s, y, -x * s + z * c];
    }
    o
}

/// A scene of `designs` (name, design desc, zones, spawn point, yaw, velocity).
fn scene(bodies: &[(&'static str, &SoftVehicleDesc, ZoneMap, Vec3, f32, Vec3)], boxes: Vec<(Vec3, Vec3)>) -> (Templates, Built) {
    let mut templates = Templates::new();
    let mut fleet = Vec::new();
    let mut layout = Layout { boxes, ..Default::default() };
    for (name, d, zones, at, yaw, v) in bodies {
        templates.insert(name.to_string(), Template { desc: (*d).clone(), settings: Default::default(), zones: zones.clone() });
        fleet.push(ExplicitBeams::new(&turned(d, *yaw), *at, *v, Default::default()));
        let mut b = Body::new(*name, *name, *at, *yaw, v.length());
        b.vel = Some(*v);
        layout.bodies.push(b);
    }
    (templates, Built { layout, fleet, contact: Contact::Discrete })
}

fn sample(built: &Built, t: f32) -> Sample {
    let f = &built.fleet;
    Sample {
        t,
        pos: f.iter().map(|s| s.positions()).collect(),
        vel: f.iter().map(|s| s.nodes.iter().map(|n| n.vel).collect()).collect(),
        beams: f.iter().map(|s| s.beams.iter().map(|b| BeamState { rest: b.rest, broken: b.broken }).collect()).collect(),
    }
}

fn run(t: &Templates, b: &Built, xs: &[Sample]) -> BTreeMap<String, f64> {
    let mut r = Recorder::new(t, b);
    for x in xs {
        r.push(x);
    }
    r.finish()
}

fn near(m: &BTreeMap<String, f64>, k: &str, want: f64, tol: f64) {
    let v = m[k];
    println!("{k} = {v:.6} (want {want:.6} ± {tol})");
    assert!((v - want).abs() <= tol, "{k} = {v}, want {want} ± {tol}");
}

/// A 3 × 3 × 5 lattice of unequal masses (1.6 × 1.2 × 4 m).
fn lattice() -> SoftVehicleDesc {
    let mut n = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..5 {
                n.push(node(&format!("n{i}{j}{k}"), [-0.8 + 0.8 * i as f32, 0.3 + 0.6 * j as f32, -2.0 + 1.0 * k as f32], 1.0 + (i + 2 * j + 3 * k) as f32 % 4.0));
            }
        }
    }
    desc(n, &[])
}

#[test]
fn crash_metrics_rigid_rotation_reports_yaw_pitch_roll() {
    let d = lattice();
    for yaw0 in [0.0f32, 0.3, std::f32::consts::FRAC_PI_2, std::f32::consts::PI] {
        let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::new(1.0, 0.0, 2.0), yaw0, Vec3::ZERO)], vec![]);
        let x0 = sample(&b, 0.0);
        let (yaw, pitch, roll) = (25.0f64.to_radians(), 7.0f64.to_radians(), -4.0f64.to_radians());
        // The body's frame at spawn, and the rotation Ry Rx Rz in it, about its COM.
        let f = b.layout.bodies[0].heading().as_dvec3();
        let bm = DMat3::from_cols(DVec3::Y.cross(f), DVec3::Y, f);
        let rb = DMat3::from_rotation_y(yaw) * DMat3::from_rotation_x(pitch) * DMat3::from_rotation_z(roll);
        let r = (bm * rb * bm.transpose()).as_mat3();
        let c = com(&x0.pos[0], &b.fleet[0].nodes.iter().map(|n| n.mass).collect::<Vec<_>>()).as_vec3();
        let mut x1 = x0.clone();
        x1.t = 0.004;
        for p in &mut x1.pos[0] {
            *p = c + r * (*p - c) + Vec3::new(0.3, 0.1, -0.2);
        }
        // One node flown off: the robust fit leaves it out.
        x1.pos[0][7] += Vec3::new(4.0, 3.0, -5.0);
        let m = run(&t, &b, &[x0, x1]);
        near(&m, "car.yaw_deg", 25.0, 1e-2);
        near(&m, "car.yaw_peak_deg", 25.0, 1e-2);
        near(&m, "car.pitch_end_deg", 7.0, 1e-2);
        near(&m, "car.pitch_peak_deg", 7.0, 1e-2);
        near(&m, "car.roll_peak_deg", 4.0, 1e-2);
    }
}

#[test]
fn crash_metrics_fit_recovers_a_known_rotation() {
    let d = lattice();
    let p: Vec<Vec3> = d.nodes.iter().map(|n| Vec3::from(n.position)).collect();
    let w: Vec<f32> = d.nodes.iter().map(|n| n.mass).collect();
    let r = Mat3::from_euler(glam::EulerRot::YXZ, 1.2, -0.4, 2.5);
    let q: Vec<Vec3> = p.iter().map(|x| r * *x + Vec3::new(5.0, -1.0, 2.0)).collect();
    let (fr, ft) = weighted_fit(&p, &q, &w, &vec![true; p.len()]);
    let err = (fr - r.as_dmat3()).to_cols_array().iter().map(|x| x.abs()).fold(0.0, f64::max);
    println!("fit rotation error {err:e}, translation {ft:?}");
    assert!(err < 1e-5);
    assert!((ft - DVec3::new(5.0, -1.0, 2.0)).length() < 1e-4);
}

/// One beam of 1 m yielding: threshold 1 kN, stiffness 1 MN/m.
fn one_beam(kind: ExplicitKind, cap: f32) -> (Templates, Built) {
    let d = desc(vec![node("a", [0.0, 1.0, -0.5], 10.0), node("b", [0.0, 1.0, 0.5], 10.0)], &[(0, 1)]);
    let (t, mut b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![]);
    let beam = &mut b.fleet[0].beams[0];
    (beam.kind, beam.k, beam.tc, beam.tt, beam.limit_k) = (kind, 1.0e6, 1000.0, 1000.0, cap);
    (t, b)
}

fn set_rest(x: &Sample, rest: f32, t: f32) -> Sample {
    let mut y = x.clone();
    y.t = t;
    y.beams[0][0].rest = rest;
    y
}

#[test]
fn crash_metrics_plastic_work_of_one_beam() {
    // No hardening (JBeam cap at the threshold): threshold × set.
    let (t, b) = one_beam(ExplicitKind::JbeamPlain, 1000.0);
    let x0 = sample(&b, 0.0);
    let m = run(&t, &b, &[x0.clone(), set_rest(&x0, 0.9, 0.004)]);
    near(&m, "car.plastic_work_j", 100.0, 1e-3);
    near(&m, "all.plastic_work_j", 100.0, 1e-3);
    // Hardening uncapped: tc L0 ln(L0/L) compressed, tt (L² − L0²)/2L0 stretched, the same
    // in one sample or in many.
    let (t, b) = one_beam(ExplicitKind::JbeamPlain, f32::INFINITY);
    let x0 = sample(&b, 0.0);
    let m = run(&t, &b, &[x0.clone(), set_rest(&x0, 0.9, 0.004)]);
    near(&m, "car.plastic_work_j", 1000.0 * (1.0f64 / 0.9).ln(), 1e-3);
    let steps: Vec<Sample> = (0..=50).map(|i| set_rest(&x0, 1.0 - 0.1 * i as f32 / 50.0, 0.004 * i as f32)).collect();
    near(&run(&t, &b, &steps), "car.plastic_work_j", 1000.0 * (1.0f64 / 0.9).ln(), 1e-3);
    let m = run(&t, &b, &[x0.clone(), set_rest(&x0, 1.1, 0.004)]);
    near(&m, "car.plastic_work_j", 1000.0 * (1.21 - 1.0) / 2.0, 1e-3);
    // Capped at 1.05 kN: hardening to the cap (at L = 1/1.05) and flat past it.
    let (t, b) = one_beam(ExplicitKind::JbeamPlain, 1050.0);
    let x0 = sample(&b, 0.0);
    let m = run(&t, &b, &[x0.clone(), set_rest(&x0, 0.9, 0.004)]);
    let r_star = 1.0 / 1.05f64;
    near(&m, "car.plastic_work_j", 1000.0 * (1.0 / r_star).ln() + 1050.0 * (r_star - 0.9), 1e-3);
    assert_eq!(m["car.bent_frac"], 1.0);
}

#[test]
fn crash_metrics_hardening_matches_the_solver() {
    // The solver's own beam, compressed slowly: its threshold is tc0 · L0 / L.
    let mut beam = Beam::new(0, 1, ExplicitKind::JbeamPlain, crate::explicit_beams::ExplicitType::Deformable, 1.0e6, 0.0, 1.0, 1000.0, 1.0e9, 0.0);
    beam.limit_k = f32::INFINITY;
    let mk = |z: f32| Node { pos: Vec3::new(0.0, 0.0, z), vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 0.0, flags: 0 };
    for i in 0..=2000 {
        let len = 1.0 - 0.1 * i as f32 / 2000.0;
        jbeam_force(&mut beam, &mk(0.0), &mk(len), false);
    }
    let want = 1000.0 / beam.rest;
    println!("rest {} tc {} want {want}", beam.rest, beam.tc);
    assert!((beam.tc - want).abs() < 1e-2 * want);
}

#[test]
fn crash_metrics_momentum_and_kinetic_energy() {
    let d = desc(vec![node("a", [0.0, 0.5, 0.0], 100.0), node("b", [0.0, 0.5, 1.0], 300.0)], &[(0, 1)]);
    let e = desc(vec![node("a", [0.0, 0.5, 0.0], 50.0)], &[]);
    let (t, b) = scene(
        &[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 10.0)), ("box", &e, ZoneMap::default(), Vec3::new(0.0, 0.0, 20.0), 0.0, Vec3::new(2.0, 0.0, -4.0))],
        vec![],
    );
    let x0 = sample(&b, 0.0);
    let m = run(&t, &b, &[x0.clone(), x0]);
    near(&m, "all.ke_start_j", 0.5 * 400.0 * 100.0 + 0.5 * 50.0 * 20.0, 1e-6);
    near(&m, "all.momentum_z_start", 4000.0 - 200.0, 1e-6);
    near(&m, "all.momentum_x_start", 100.0, 1e-6);
    near(&m, "all.ke_lost_j", 0.0, 1e-9);
    near(&m, "car.speed_before", 10.0, 1e-6);
    near(&m, "pair.car_box.com_gap_min", 20.0 - 0.75, 1e-5);
}

#[test]
fn crash_metrics_zones_by_prefix() {
    let d = desc(
        vec![node("fu_hood", [0.5, 1.0, 2.0], 1.0), node("f_rail", [-0.5, 0.4, 2.0], 1.0), node("c_pillar", [0.5, 1.0, 0.0], 1.0), node("x", [-0.5, 1.0, 0.0], 1.0)],
        &[(0, 1), (1, 0), (2, 3), (3, 2), (3, 3)],
    );
    let zm = ZoneMap { parts: vec![("f".into(), Section::FrontLower), ("fu".into(), Section::FrontUpper), ("c".into(), Section::Cabin)], gauges: vec![] };
    let t = Template { desc: d.clone(), settings: Default::default(), zones: zm };
    let s = ExplicitBeams::new(&d, Vec3::ZERO, Vec3::ZERO, Default::default());
    let rest: Vec<Vec3> = d.nodes.iter().map(|n| Vec3::from(n.position)).collect();
    let z: Vec<Section> = assign_zones(&t, &s.beams, &rest).iter().map(|z| z.section).collect();
    // Longest prefix wins; the first node's section; an unnamed first node takes the second's.
    assert_eq!(z, vec![Section::FrontUpper, Section::FrontLower, Section::Cabin, Section::Cabin, Section::Other]);
}

#[test]
fn crash_metrics_zones_by_geometry_keep_left_left_when_yawed() {
    // A 4 m body: front lower left (+X) beam, rear right beam, cabin centre-line beam.
    let d = desc(
        vec![
            node("a", [0.6, 0.3, 1.8], 1.0),
            node("b", [0.6, 0.3, 1.5], 1.0),
            node("c", [-0.6, 1.0, -1.8], 1.0),
            node("d", [-0.6, 1.0, -1.5], 1.0),
            node("e", [0.0, 1.4, 0.0], 1.0),
            node("f", [0.0, 1.4, 0.5], 1.0),
            node("g", [0.0, 0.2, 2.0], 1.0),
            node("h", [0.0, 0.2, -2.0], 1.0),
        ],
        &[(0, 1), (2, 3), (4, 5)],
    );
    for yaw in [0.0f32, std::f32::consts::FRAC_PI_2, std::f32::consts::PI] {
        let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::new(3.0, 0.0, -1.0), yaw, Vec3::ZERO)], vec![]);
        let x0 = sample(&b, 0.0);
        let mut x1 = x0.clone();
        x1.t = 0.004;
        for st in &mut x1.beams[0] {
            st.rest *= 0.9;
        }
        let m = run(&t, &b, &[x0, x1]);
        let w = |k: &str| m[&format!("car.zone.{k}.work_j")];
        println!("yaw {yaw}: front_lower_left {} rear_right {} cabin_left {}", w("front_lower_left"), w("rear_right"), w("cabin_left"));
        assert!(w("front_lower_left") > 0.0 && w("front_lower_right") == 0.0);
        assert!(w("rear_right") > 0.0 && w("rear_left") == 0.0);
        assert!(w("cabin_left") > 0.0 && (w("cabin_left") - w("cabin_right")).abs() < 1e-9);
        near(&m, "car.zone.front.work_frac", w("front_lower_left") / m["car.plastic_work_j"], 1e-12);
        assert_eq!(m["car.zone.front_upper.work_j"], 0.0);
    }
}

#[test]
fn crash_metrics_parts_count_when_a_beam_breaks() {
    let d = desc((0..4).map(|i| node(&format!("n{i}"), [0.0, 0.5, i as f32], 1.0 + i as f32)).collect(), &[(0, 1), (1, 2), (2, 3)]);
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let mut x1 = x0.clone();
    x1.beams[0][2].broken = true;
    let m = run(&t, &b, &[x0, x1]);
    near(&m, "car.detached_parts", 1.0, 0.0);
    near(&m, "car.detached_mass", 4.0, 1e-6);
    near(&m, "car.broken_frac", 1.0 / 3.0, 1e-9);
}

#[test]
fn crash_metrics_box_depth_of_a_node_inside() {
    let d = desc(vec![node("a", [0.0, 0.5, 0.0], 1.0), node("b", [0.0, 0.5, 2.9], 1.0)], &[(0, 1)]);
    // A box from z = 2.8 to 4: node b is 0.1 m in.
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![(Vec3::new(0.0, 1.0, 3.4), Vec3::new(2.0, 2.0, 0.6))]);
    let x0 = sample(&b, 0.0);
    near(&run(&t, &b, &[x0]), "all.box_depth_m", 0.1, 1e-5);
}

#[test]
fn crash_metrics_energy_gain_of_a_stretched_beam() {
    let d = desc(vec![node("a", [-0.5, 1.0, 0.0], 1.0), node("b", [0.5, 1.0, 0.0], 1.0)], &[(0, 1)]);
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![]);
    let k = f64::from(b.fleet[0].beams[0].k);
    let x0 = sample(&b, 0.0);
    let mut x1 = x0.clone();
    x1.pos[0][0].x -= 0.005;
    x1.pos[0][1].x += 0.005;
    near(&run(&t, &b, &[x0, x1]), "all.energy_gain_j", 0.5 * k * 0.01 * 0.01, 1e-3 * 0.5 * k * 1e-4);
}

#[test]
fn crash_metrics_pulse_of_a_constant_deceleration() {
    let d = lattice();
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::new(0.0, 0.0, 10.0))], vec![]);
    let x0 = sample(&b, 0.0);
    // 10 m/s to 0 in 100 ms (from 40 ms), then still.
    let xs: Vec<Sample> = (0..60)
        .map(|i| {
            let t = 0.004 * i as f32;
            let v = (10.0 - 100.0 * (t - 0.04)).clamp(0.0, 10.0);
            let mut x = x0.clone();
            x.t = t;
            x.vel[0].iter_mut().for_each(|u| *u = Vec3::new(0.0, 0.0, v));
            x
        })
        .collect();
    let m = run(&t, &b, &xs);
    near(&m, "car.decel_peak_g", 100.0 / 9.807, 0.02 * 100.0 / 9.807);
    near(&m, "car.decel_mean_g", 100.0 / 9.807, 0.1 * 100.0 / 9.807);
    near(&m, "car.pulse_ms", 104.0, 8.0);
    near(&m, "car.dv_along", -10.0, 1e-5);
    near(&m, "car.speed_end", 0.0, 1e-6);
}

#[test]
fn crash_metrics_bounded_spring_energy() {
    // k = 1 to a bound of 1, ramping to 3 over a zone of 1, 3 past it: ∫₀³ k(x) x dx =
    // 1/2 + 19/6 + 15/2.
    let want = 0.5 + 19.0 / 6.0 + 7.5;
    for e in [3.0, -3.0] {
        let got = super::beams::stored(e, [e.signum(), 2.0 * e.signum()], |x| super::beams::ramp(1.0, 3.0, 1.0, 1.0, x.abs()));
        println!("bounded energy at e = {e}: {got} (want {want})");
        assert!((got - want).abs() < 1e-12);
    }
    let plain = super::beams::stored(-0.4, [1.0, 2.0], |_| 5.0);
    assert!((plain - 0.5 * 5.0 * 0.16).abs() < 1e-12);
}
