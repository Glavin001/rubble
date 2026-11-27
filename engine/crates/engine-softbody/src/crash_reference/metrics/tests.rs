//! Closed-form checks of the crash measuring kit on hand-built bodies and synthetic
//! samples (no car is rigged here; `engine-softbody-lab/tests/crash_metrics_s02.rs` runs
//! S02 itself).

use super::super::{BeamState, Body, Built, Contact, Layout, Sample, Section, Template, Templates, ZoneMap};
use super::*;
use crate::desc::{SoftBeam, SoftNode, SoftTriangle, SoftVehicleDesc};
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
        wheel_spin: Vec::new(),
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
    // |Σ m v| / Σ m: (100, 0, 3800) N·s over 450 kg.
    near(&m, "all.com_speed_end", (100.0f64.powi(2) + 3800.0f64.powi(2)).sqrt() / 450.0, 1e-6);
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

fn near_v(v: f64, want: f64, tol: f64) {
    println!("{v:.6} (want {want:.6} ± {tol})");
    assert!((v - want).abs() <= tol, "{v}, want {want} ± {tol}");
}

#[test]
fn crash_metrics_robust_extent_of_known_lists() {
    // 101 nodes 2 cm apart along +Z, in no order: end to end, 2 m.
    let p: Vec<Vec3> = (0..=100).map(|i| Vec3::new(0.3, 0.0, 0.02 * ((i * 37) % 101) as f32)).collect();
    near_v(robust_extent(&p, |_| true, Vec3::Z, EXTENT_GAP), 2.0, 1e-6);
    // Along another axis: the projection.
    near_v(robust_extent(&p, |_| true, Vec3::new(0.0, 0.6, 0.8), EXTENT_GAP), 1.6, 1e-6);
    // Nodes flung off (one 1 km out, three together 0.3 m past the end) are left out; a
    // node 0.2 m past the end (within the gap) is in.
    let mut q = p.clone();
    q.extend([Vec3::new(0.0, 0.0, 1000.0), Vec3::new(0.0, 0.0, -0.3), Vec3::new(0.0, 0.0, -0.31), Vec3::new(0.0, 0.0, -0.32)]);
    near_v(robust_extent(&q, |_| true, Vec3::Z, EXTENT_GAP), 2.0, 1e-6);
    q.push(Vec3::new(0.0, 0.0, 2.2));
    near_v(robust_extent(&q, |_| true, Vec3::Z, EXTENT_GAP), 2.2, 1e-6);
    // Only the nodes kept.
    near_v(robust_extent(&p, |i| p[i].z <= 1.0 + 1e-4, Vec3::Z, EXTENT_GAP), 1.0, 1e-6);
    assert_eq!(robust_extent(&p[..1], |_| true, Vec3::Z, EXTENT_GAP), 0.0);
}

/// A body of two ladders 1 m apart (left x = +0.5, right x = −0.5), each two rows (y 0.3,
/// 1.0) of 101 nodes 4 cm apart along z ∈ [−2, 2], every row chained, each side's rows
/// joined at every station, and rungs across at every station up to `rungs`. Masses: left
/// `ml`, right `mr`. Node index: side s (0 left, 1 right), row r, station k:
/// (s·2 + r)·101 + k. Beam order: per side, its two rows' 100 chain beams, then its 101
/// row-to-row beams; then the rungs.
fn ladders(ml: f32, mr: f32, rungs: usize) -> SoftVehicleDesc {
    let mut n = Vec::new();
    for s in 0..2 {
        for r in 0..2 {
            for k in 0..101 {
                let x = if s == 0 { 0.5 } else { -0.5 };
                n.push(node(&format!("n{s}{r}_{k}"), [x, [0.3, 1.0][r], -2.0 + 0.04 * k as f32], if s == 0 { ml } else { mr }));
            }
        }
    }
    let id = |s: usize, r: usize, k: usize| ((s * 2 + r) * 101 + k) as u32;
    let mut b = Vec::new();
    for s in 0..2 {
        for r in 0..2 {
            for k in 0..100 {
                b.push((id(s, r, k), id(s, r, k + 1)));
            }
        }
        for k in 0..101 {
            b.push((id(s, 0, k), id(s, 1, k)));
        }
    }
    for k in 0..=rungs {
        for r in 0..2 {
            b.push((id(0, r, k), id(1, r, k)));
        }
    }
    desc(n, &b)
}

/// The rigid motion `r` about `c` then `t`, applied to the sample's body 0.
fn moved(x: &Sample, r: Mat3, c: Vec3, t: Vec3, at: f32) -> Sample {
    let mut y = x.clone();
    y.t = at;
    for p in &mut y.pos[0] {
        *p = c + r * (*p - c) + t;
    }
    y
}

/// Body 0's frame at spawn as a world rotation (columns left, up, forward).
fn frame(b: &Built) -> Mat3 {
    let f = b.layout.bodies[0].heading();
    Mat3::from_cols(Vec3::Y.cross(f), Vec3::Y, f)
}

#[test]
fn crash_metrics_length_and_crush_in_the_fitted_frame() {
    // Compressed along its own length to 0.75 at the deepest and 0.8 at the end, while
    // turned 100°, pitched 15° and thrown 3 m: the world extent along the first heading
    // would read the turn; the fitted frame reads only the crush.
    let d = ladders(1.0, 1.0, 100);
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::new(2.0, 0.0, -1.0), 0.6, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let bm = frame(&b);
    let o = b.layout.bodies[0].at;
    let squash = |f: f32, at: f32| {
        // In the body frame z ← −2 + f (z + 2), then the rigid motion.
        let mut y = x0.clone();
        for p in &mut y.pos[0] {
            let mut l = bm.transpose() * (*p - o);
            l.z = -2.0 + f * (l.z + 2.0);
            *p = o + bm * l;
        }
        let r = bm * Mat3::from_rotation_y(100f32.to_radians()) * Mat3::from_rotation_x(15f32.to_radians()) * bm.transpose();
        moved(&y, r, o, Vec3::new(0.0, 1.0, 3.0), at)
    };
    let m = run(&t, &b, &[x0.clone(), squash(0.75, 0.004), squash(0.8, 0.008)]);
    let rest = 4.0;
    near(&m, "car.length0", rest, 1e-4);
    near(&m, "car.gauge.length.rest", rest, 1e-4);
    near(&m, "car.crush_peak", 0.25 * rest, 1e-4);
    near(&m, "car.crush_residual", 0.2 * rest, 1e-4);
    near(&m, "car.gauge.length.shrink_frac", 0.2, 1e-4);
    near(&m, "car.gauge.length_left.change_residual", -0.2 * rest, 1e-4);
    near(&m, "car.gauge.length_right.change_residual", -0.2 * rest, 1e-4);
    near(&m, "car.yaw_deg", 100.0, 1e-2);
    // Height along the fitted up: the rows 0.7 m apart, unchanged.
    near(&m, "car.gauge.height.rest", 0.7, 1e-5);
    near(&m, "car.gauge.height.change_residual", 0.0, 1e-4);
}

#[test]
fn crash_metrics_side_lengths_leave_out_detached_parts() {
    // The left ladder (light, so the heavy right one sets the fit) crushed to 0.75 of its
    // length; the right ladder's front ten stations (no rungs past station 90) torn off
    // and flung 10 m ahead and 4 m up; the whole turned 30° about the body's up.
    let d = ladders(1e-3, 1000.0, 90);
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, -0.4, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let bm = frame(&b);
    let o = b.layout.bodies[0].at;
    let mut x1 = x0.clone();
    x1.t = 0.004;
    let flung = |i: usize| i >= 202 && (i - 202) % 101 > 90;
    for (i, p) in x1.pos[0].iter_mut().enumerate() {
        let mut l = bm.transpose() * (*p - o);
        if i < 202 {
            l.z = -2.0 + 0.75 * (l.z + 2.0);
        }
        if flung(i) {
            l += Vec3::new(0.0, 4.0, 10.0);
        }
        *p = o + bm * l;
    }
    // Break the chain beams joining station 90 to 91 in both right rows.
    for r in 0..2 {
        x1.beams[0][301 + r * 100 + 90].broken = true;
    }
    let x1 = moved(&x1, bm * Mat3::from_rotation_y(30f32.to_radians()) * bm.transpose(), o, Vec3::new(1.0, 0.0, 0.0), 0.004);
    let m = run(&t, &b, &[x0, x1]);
    let rest = 4.0;
    // The right half still attached: stations 0..=90 of both rows, z ∈ [−2, 1.6].
    let want_right = 3.6;
    near(&m, "car.gauge.length_left.rest", rest, 1e-4);
    near(&m, "car.gauge.length_left.change_residual", -0.25 * rest, 1e-3);
    near(&m, "car.gauge.length_left.shrink_frac", 0.25, 1e-3);
    near(&m, "car.gauge.length_right.rest", rest, 1e-4);
    near(&m, "car.gauge.length_right.change_residual", want_right - rest, 1e-3);
    near(&m, "car.yaw_deg", 30.0, 1e-2);
    near(&m, "car.detached_parts", 1.0, 0.0);
}

#[test]
fn crash_metrics_yaw_unwraps_through_a_full_turn() {
    let d = lattice();
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 1.0, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let bm = frame(&b);
    let c = com(&x0.pos[0], &b.fleet[0].nodes.iter().map(|n| n.mass).collect::<Vec<_>>()).as_vec3();
    let turn = |deg: f32, k: usize| moved(&x0, bm * Mat3::from_rotation_y(deg.to_radians()) * bm.transpose(), c, Vec3::ZERO, 0.004 * k as f32);
    // Up to 400° in 10° steps, then back to 380°: the peak is the unwrapped 400°.
    let mut xs: Vec<Sample> = (0..=40).map(|k| turn(10.0 * k as f32, k)).collect();
    xs.push(turn(380.0, 41));
    let m = run(&t, &b, &xs);
    near(&m, "car.yaw_deg", 380.0, 1e-2);
    near(&m, "car.yaw_peak_deg", 400.0, 1e-2);
    near(&m, "car.yaw_abs_peak_deg", 400.0, 1e-2);
    // The other way: to −50°, then +20°: the peak keeps its sign.
    let mut xs: Vec<Sample> = (0..=5).map(|k| turn(-10.0 * k as f32, k)).collect();
    xs.extend((0..=7).map(|k| turn(-50.0 + 10.0 * k as f32, 6 + k)));
    let m = run(&t, &b, &xs);
    near(&m, "car.yaw_deg", 20.0, 1e-2);
    near(&m, "car.yaw_peak_deg", -50.0, 1e-2);
    near(&m, "car.yaw_abs_peak_deg", 50.0, 1e-2);
}

#[test]
fn crash_metrics_speed_peak_and_box_face() {
    // A body speeding up to 12 m/s and back to 3: the peak is 12.
    let d = desc(vec![node("a", [0.0, 0.5, 0.0], 1.0), node("b", [0.0, 0.5, 2.9], 3.0)], &[(0, 1)]);
    // A box from z = 2.8 to 4: its near face is z = 2.8 seen from −Z, 4.0 from +Z.
    let bx = (Vec3::new(0.0, 1.0, 3.4), Vec3::new(2.0, 2.0, 0.6));
    for (yaw, face_z) in [(0.0f32, 2.8f64), (std::f32::consts::PI, 4.0)] {
        let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, yaw, Vec3::ZERO)], vec![bx]);
        let x0 = sample(&b, 0.0);
        let xs: Vec<Sample> = [0.0f32, 5.0, 12.0, 7.0, 3.0]
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let mut x = x0.clone();
                x.t = 0.004 * k as f32;
                x.vel[0] = vec![Vec3::new(0.6 * v, 0.0, 0.8 * v); 2];
                x
            })
            .collect();
        let m = run(&t, &b, &xs);
        near(&m, "car.speed_peak", 12.0, 1e-5);
        let z = f64::from(x0.pos[0][0].z) * 0.25 + f64::from(x0.pos[0][1].z) * 0.75;
        let h = if yaw == 0.0 { 1.0 } else { -1.0 };
        near(&m, "car.com_end_past_box0", (z - face_z) * h, 1e-5);
    }
    // Without a box: none.
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![]);
    assert!(!run(&t, &b, &[sample(&b, 0.0)]).contains_key("car.com_end_past_box0"));
}

#[test]
fn crash_metrics_reach_past_the_other_com() {
    // Car heading +Z: nodes at z = 0, 1, 2, 1 (masses 1, 1, 2, ~0), its collision
    // triangle on nodes 0, 1, 3 (z ≤ 1). Barrier heading −Z: two nodes (masses 3, 1).
    let mut car = desc(vec![node("a", [0.0, 0.5, 0.0], 1.0), node("b", [0.5, 0.5, 1.0], 1.0), node("c", [0.0, 1.0, 2.0], 2.0), node("d", [-0.5, 0.5, 1.0], 1.0e-6)], &[(0, 1), (1, 2), (1, 3)]);
    car.triangles.push(SoftTriangle { nodes: [0, 1, 3], ..Default::default() });
    let bar = desc(vec![node("a", [0.0, 0.5, 0.15], 3.0), node("b", [0.0, 0.5, -0.45], 1.0)], &[(0, 1)]);
    let (t, b) = scene(&[("car", &car, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO), ("barrier", &bar, ZoneMap::default(), Vec3::new(0.0, 0.0, 1.45), std::f32::consts::PI, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let m = run(&t, &b, &[x0.clone()]);
    let car_com = (1.0 + 4.0 + 1.0e-6) / (4.0 + 1.0e-6);
    let (z0, z1) = (f64::from(x0.pos[1][0].z), f64::from(x0.pos[1][1].z));
    let bar_com = 0.75 * z0 + 0.25 * z1;
    near(&m, "pair.car_barrier.reach_max", 2.0 - bar_com, 1e-5);
    near(&m, "pair.car_barrier.skin_reach_max", 1.0 - bar_com, 1e-5);
    // The barrier heads −Z: its node at the smallest z gets furthest past the car's COM.
    near(&m, "pair.barrier_car.reach_max", car_com - z0.min(z1), 1e-5);
    // Its skin: none (no triangles).
    assert!(!m.contains_key("pair.barrier_car.skin_reach_max"));
    // Later the car alone moves 0.3 m on and back: the reach is the largest over the samples.
    let mut x1 = x0.clone();
    x1.t = 0.004;
    x1.pos[0].iter_mut().for_each(|p| p.z += 0.3);
    let mut x2 = x0.clone();
    x2.t = 0.008;
    let m = run(&t, &b, &[x0, x1, x2]);
    near(&m, "pair.car_barrier.reach_max", 2.3 - bar_com, 1e-5);
}

#[test]
fn crash_metrics_pair_ratios() {
    // Two one-beam bodies: a shortened 0.1 m, b 0.05 m, by yield (no hardening: threshold
    // 1 kN × set) and in position.
    let d = desc(vec![node("a", [0.0, 1.0, -0.5], 10.0), node("b", [0.0, 1.0, 0.5], 10.0)], &[(0, 1)]);
    let (t, mut b) = scene(&[("car_a", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO), ("car_b", &d, ZoneMap::default(), Vec3::new(3.0, 0.0, 0.0), 0.0, Vec3::ZERO)], vec![]);
    for f in &mut b.fleet {
        let beam = &mut f.beams[0];
        (beam.kind, beam.k, beam.tc, beam.tt, beam.limit_k) = (ExplicitKind::JbeamPlain, 1.0e6, 1000.0, 1000.0, 1000.0);
    }
    // The recorder reads beam rates from the fleet as built.
    let x0 = sample(&b, 0.0);
    let mut x1 = x0.clone();
    x1.t = 0.004;
    for (k, dl) in [(0, 0.1f32), (1, 0.05)] {
        x1.beams[k][0].rest = 1.0 - dl;
        x1.pos[k][1].z -= dl;
    }
    let m = run(&t, &b, &[x0, x1]);
    near(&m, "car_a.plastic_work_j", 100.0, 1e-3);
    near(&m, "car_b.plastic_work_j", 50.0, 1e-3);
    near(&m, "pair.car_a_car_b.plastic_work_ratio", 101.0 / 51.0, 1e-4);
    near(&m, "car_a.crush_peak", 0.1, 1e-5);
    near(&m, "pair.car_a_car_b.crush_peak_ratio", (0.1 + 0.001) / (0.05 + 0.001), 1e-3);
    near(&m, "pair.car_a_car_b.yaw_peak_diff_deg", 0.0, 1e-6);
}

#[test]
fn crash_metrics_yaw_peak_difference() {
    let d = lattice();
    let (t, b) = scene(&[("car_a", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO), ("car_b", &d, ZoneMap::default(), Vec3::new(5.0, 0.0, 0.0), std::f32::consts::PI, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let mut x1 = x0.clone();
    x1.t = 0.004;
    for (k, deg) in [(0usize, 20.0f32), (1, -10.0)] {
        let c = com(&x0.pos[k], &b.fleet[k].nodes.iter().map(|n| n.mass).collect::<Vec<_>>()).as_vec3();
        let r = Mat3::from_rotation_y(deg.to_radians());
        x1.pos[k].iter_mut().for_each(|p| *p = c + r * (*p - c));
    }
    let m = run(&t, &b, &[x0, x1]);
    near(&m, "car_a.yaw_peak_deg", 20.0, 1e-2);
    near(&m, "car_b.yaw_peak_deg", -10.0, 1e-2);
    near(&m, "pair.car_a_car_b.yaw_peak_diff_deg", 30.0, 1e-2);
}

#[test]
fn crash_metrics_crossings_of_two_meshes() {
    // a: one horizontal triangle at y = 1; b: one vertical triangle in x = 0 whose two
    // edges from (0, 0.5, 0) pass through a's (at z = 0 and z = 0.25), its third above it.
    let mut a = desc(vec![node("a", [-1.0, 1.0, -1.0], 1.0), node("b", [1.0, 1.0, -1.0], 1.0), node("c", [0.0, 1.0, 1.5], 1.0)], &[(0, 1), (1, 2), (2, 0)]);
    a.triangles.push(SoftTriangle { nodes: [0, 1, 2], ..Default::default() });
    let mut v = desc(vec![node("a", [0.0, 0.5, 0.0], 1.0), node("b", [0.0, 1.5, 0.0], 1.0), node("c", [0.0, 1.5, 0.5], 1.0)], &[(0, 1), (1, 2), (2, 0)]);
    v.triangles.push(SoftTriangle { nodes: [0, 1, 2], ..Default::default() });
    // b spawns 10 m away (no crossing), then is moved into a.
    let (t, b) = scene(&[("a", &a, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO), ("b", &v, ZoneMap::default(), Vec3::new(0.0, 0.0, 10.0), 0.0, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    near(&run(&t, &b, &[x0.clone()]), "pair.a_b.crossings_max", 0.0, 0.0);
    let mut x1 = x0.clone();
    x1.t = 0.004;
    x1.pos[1].iter_mut().for_each(|p| p.z -= 10.0);
    let mut x2 = x0.clone();
    x2.t = 0.008;
    near(&run(&t, &b, &[x0.clone(), x1.clone(), x2]), "pair.a_b.crossings_max", 2.0, 0.0);
    // b's triangle torn (a beam along its edge broken, as the JBeam spawn lists them): its
    // edges are out of the surface.
    let (t, mut b) = (t, b);
    b.fleet[1].tri_breaks = vec![[0, 1, 2, crate::explicit_jbeam_ext::NO_BEAM]];
    let mut x3 = x1;
    x3.beams[1][1].broken = true;
    near(&run(&t, &b, &[x0, x3]), "pair.a_b.crossings_max", 0.0, 0.0);
}

/// The FMVSS 214 barrier as the crash suite spawns it (`crash_rigs::mdb214_jbeam`, JBeam
/// mode), and its nodes' names.
fn barrier() -> (ExplicitBeams, Vec<String>) {
    let text = crate::crash_rigs::mdb214_jbeam();
    let d = crate::jbeam::load_jbeam(&[("mdb214.jbeam", text.as_str())], None).expect("barrier");
    let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, Default::default()).expect("spawn");
    (s, d.nodes.iter().map(|n| n.name.clone()).collect())
}

/// The index of the barrier's beam of `kind` from node `a` to node `b` (either way).
fn barrier_beam(s: &ExplicitBeams, names: &[String], kind: ExplicitKind, a: &str, b: &str) -> usize {
    let id = |n: &str| names.iter().position(|x| x == n).expect("node") as u32;
    let (a, b) = (id(a), id(b));
    s.beams.iter().position(|x| x.kind == kind && ((x.a == a && x.b == b) || (x.a == b && x.b == a))).expect("beam")
}

/// The solver's force on beam `j` of `s` with its nodes at `pos` (at rest: no damping), as
/// the beam pass computes it (`ExplicitBeams::step_rest`): the extension-record path for
/// a beam that has one, else `jbeam_force`. Returns the force along id1–id2 (+ pushes
/// apart) and, for an LBEAM, the arm terms on id1 and id2 (`lbeam_force`).
fn solver_force(s: &mut ExplicitBeams, j: usize, pos: &[Vec3]) -> (f32, Vec3, Vec3, u32) {
    let node = |p: Vec3| Node { pos: p, vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 0.0, flags: 0 };
    let mut beam = s.beams[j];
    let (na, nb) = (node(pos[beam.a as usize]), node(pos[beam.b as usize]));
    let x = s.ext_of.get(j).copied().unwrap_or(0) as usize;
    let out = if beam.kind == ExplicitKind::JbeamLbeam {
        let c = s.beam_ext[x - 1].index(0);
        let o = crate::explicit_jbeam_ext::lbeam_force(&mut beam, &na, &nb, &node(pos[c as usize]), None);
        s.beams[j] = beam;
        return (o.beam.force, o.fa, o.fb, c);
    } else if x > 0 {
        crate::explicit_jbeam_ext::jbeam_ext_force_with(&mut beam, &mut s.beam_ext[x - 1], &na, &nb, None, false)
    } else {
        jbeam_force(&mut beam, &na, &nb, false)
    };
    s.beams[j] = beam;
    (out.force, Vec3::ZERO, Vec3::ZERO, u32::MAX)
}

/// Drives node `m` of beam `j` along `path` (positions, many small steps) quasi-statically
/// and returns the external work done on it (J, trapezoid rule on the solver's force), and
/// the metric's plastic work and stored energy at the end.
fn drive(s: &mut ExplicitBeams, j: usize, pos: &mut [Vec3], m: usize, path: &[Vec3]) -> (f64, f64, f64) {
    let spawn = s.clone();
    let mut plastic = super::beams::can_yield(&s.beams[j]).then(|| super::beams::Plastic::new(&s.beams[j]));
    let mut lplastic = super::beams::LbeamPlastic::new(&s.beams[j]);
    // The force on node m (world): +f along (a − b) on a, −f on b; LBEAM arm terms.
    let force_on = |s: &mut ExplicitBeams, pos: &[Vec3]| -> Vec3 {
        let (f, fa, fb, _) = solver_force(s, j, pos);
        let b = s.beams[j];
        let d = (pos[b.a as usize] - pos[b.b as usize]).normalize();
        if m == b.a as usize {
            d * f + fa
        } else if m == b.b as usize {
            -d * f + fb
        } else {
            -(fa + fb)
        }
    };
    let mut w = 0.0f64;
    let mut f_prev = force_on(s, pos);
    for &p in path {
        let dx = p - pos[m];
        pos[m] = p;
        let f_now = force_on(s, pos);
        // The driver holds the node against the beam: it does −F·dx.
        w -= f64::from((0.5 * (f_prev + f_now)).dot(dx));
        f_prev = f_now;
        if let Some(pl) = plastic.as_mut() {
            pl.advance(s.beams[j].rest);
        }
        if let Some(pl) = lplastic.as_mut() {
            pl.advance(&spawn, j, s.beams[j].rest, pos);
        }
    }
    let stored = super::beams::elastic(&spawn.beams, Some(&spawn), j, s.beams[j].rest, s.beams[j].broken, pos);
    (w, plastic.map_or(0.0, |p| p.work) + lplastic.map_or(0.0, |p| p.work), stored)
}

/// `n` steps from `a` to `b`.
fn line(a: Vec3, b: Vec3, n: usize) -> Vec<Vec3> {
    (1..=n).map(|i| a + (b - a) * (i as f32 / n as f32)).collect()
}

#[test]
fn crash_metrics_barrier_support_column_work_is_force_times_distance() {
    use crate::crash_rigs::{MDB_CRUSH, MDB_FACE};
    let (mut s, names) = barrier();
    let mut pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    // A main-block column inside the face: its crush force is 310 kPa × its cell
    // (1.676 m / 16 × 0.559 m / 6); it never hardens; it densifies at 80 % of its depth.
    let j = barrier_beam(&s, &names, ExplicitKind::JbeamSupport, "f8_3", "p8_3");
    let b0 = s.beams[j];
    let cell = MDB_FACE[0] / 16.0 * (MDB_FACE[1] / 6.0);
    let force = f64::from(MDB_CRUSH[0] * cell);
    let (k, l0) = (f64::from(b0.k), f64::from(b0.rest));
    println!("main column: tc {} N (want {force:.1}), cap {}, k {k:.0} N/m, rest {l0:.4} m, floor {}", b0.tc, b0.limit_k, b0.short_bound);
    assert!((f64::from(b0.tc) - force).abs() < 1e-4 * force);
    assert!((f64::from(b0.limit_k) - force).abs() < 1e-4 * force, "no hardening: deformLimitStress = the crush force");
    let face = names.iter().position(|n| n == "f8_3").unwrap();
    let back = pos[b0.b as usize + b0.a as usize - face];
    let axis = (pos[face] - back).normalize();
    // Crushed 0.15 m (the face node pushed toward the back plate), slowly.
    let d = 0.15f32;
    let start = pos[face];
    let (w, plastic, stored) = drive(&mut s, j, &mut pos, face, &line(start, start - axis * d, 20_000));
    let set = l0 - f64::from(s.beams[j].rest);
    println!("crushed {d} m: external work {w:.3} J, plastic {plastic:.3} J (force × set {:.3}), stored {stored:.3} J, set {set:.5} m (want {:.5})", force * set, f64::from(d) - force / k);
    assert!((set - (f64::from(d) - force / k)).abs() < 1e-4, "the set is the crush less the elastic part");
    assert!((plastic - force * set).abs() < 1e-3 * force * set, "plastic work = crush force × set");
    assert!((stored - 0.5 * force * force / k).abs() < 1e-3 * stored);
    assert!((w - (plastic + stored)).abs() < 2e-3 * w, "the solver's work is the ledger's: {w} vs {}", plastic + stored);
    // Released past its length: a support holds nothing in tension; what was stored comes back.
    let path = line(pos[face], start + axis * 0.05, 20_000);
    let (w2, plastic2, stored2) = drive(&mut s, j, &mut pos, face, &path);
    println!("released: work {w2:.3} J, plastic {plastic2:.3}, stored {stored2:.3}");
    assert!(stored2 == 0.0 && plastic2 == 0.0);
    assert!((w2 + stored).abs() < 2e-3 * stored, "{w2} J back of {stored} J stored");
    // Crushed past densification (80 % of the depth): the set stops at the floor and the
    // column is elastic about it.
    let deep = 0.9 * l0 as f32;
    let rest_before = f64::from(s.beams[j].rest);
    let path = line(pos[face], start - axis * deep, 40_000);
    let (w3, plastic3, stored3) = drive(&mut s, j, &mut pos, face, &path);
    let rest = f64::from(s.beams[j].rest);
    println!("crushed {deep:.3} m: work {w3:.2} J, plastic {plastic3:.2} J (force × set {:.2}), stored {stored3:.2} J, rest {rest:.4} (floor {})", force * (rest_before - rest), s.beams[j].short_bound);
    // The solver's floor is the deform limit or `MIN_REST` (0.1 m), the larger: 0.1 m here,
    // so this column densifies at 74 % of its depth, not the rig's 80 % (0.0762 m).
    let floor = f64::from(b0.short_bound.max(crate::explicit_beams::MIN_REST));
    assert!((rest - floor).abs() < 1e-5, "at the floor {floor}");
    assert!((plastic3 - force * (rest_before - rest)).abs() < 1e-3 * plastic3);
    assert!((w3 - (plastic3 + stored3)).abs() < 2e-3 * w3, "{w3} vs {}", plastic3 + stored3);
}

#[test]
fn crash_metrics_barrier_bumper_and_dense_block_work() {
    use crate::crash_rigs::{MDB_CRUSH, MDB_FACE};
    let (mut s, names) = barrier();
    let mut pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    // A bumper column (bumper node to face node, row 1 inside the band): 1,689 kPa × its
    // cell (1.676 m / 16 × one row, 0.559 m / 6).
    let j = barrier_beam(&s, &names, ExplicitKind::JbeamSupport, "b8_1", "f8_1");
    let b0 = s.beams[j];
    let force = f64::from(MDB_CRUSH[1] * MDB_FACE[0] / 16.0 * (MDB_FACE[1] / 6.0));
    let (k, l0) = (f64::from(b0.k), f64::from(b0.rest));
    println!("bumper column: tc {} N (want {force:.1}), cap {}, k {k:.0}, rest {l0:.4}, floor {}", b0.tc, b0.limit_k, b0.short_bound);
    assert!((f64::from(b0.tc) - force).abs() < 1e-4 * force);
    let m = names.iter().position(|n| n == "b8_1").unwrap();
    let other = (b0.a + b0.b) as usize - m;
    let axis = (pos[m] - pos[other]).normalize();
    let start = pos[m];
    let (w, plastic, stored) = drive(&mut s, j, &mut pos, m, &line(start, start - axis * 0.05, 20_000));
    let set = l0 - f64::from(s.beams[j].rest);
    println!("crushed 0.05 m: work {w:.3} J, plastic {plastic:.3} J (force × set {:.3}), stored {stored:.3} J", force * set);
    assert!((plastic - force * set).abs() < 1e-3 * force * set);
    assert!((w - (plastic + stored)).abs() < 2e-3 * w);
    // Its 0.102 m column can yield only to `MIN_REST` (0.1 m): 2 mm, then it is a spring
    // (the rig's deform limit, 0.0204 m, is below that floor).
    assert!((f64::from(s.beams[j].rest) - f64::from(crate::explicit_beams::MIN_REST)).abs() < 1e-6, "{}", s.beams[j].rest);
    // The densified block (bumper node to back plate, rest 20 % of its length, never
    // yields): elastic only, ½ k e² once pressed past its rest.
    let (mut s, names) = barrier();
    let mut pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    let j = barrier_beam(&s, &names, ExplicitKind::JbeamSupport, "b8_1", "p8_1");
    // Its precompression ramps in over its first steps: run it out at spawn.
    for _ in 0..10_000 {
        solver_force(&mut s, j, &pos);
    }
    let b0 = s.beams[j];
    println!("dense block: tc {} k {} rest {} (length {})", b0.tc, b0.k, b0.rest, pos[b0.a as usize].distance(pos[b0.b as usize]));
    assert!(!b0.tc.is_finite() || b0.tc > 1e30);
    let m = names.iter().position(|n| n == "b8_1").unwrap();
    let other = (b0.a + b0.b) as usize - m;
    let axis = (pos[m] - pos[other]).normalize();
    let start = pos[m];
    let len0 = pos[m].distance(pos[other]);
    let target = start - axis * (len0 - b0.rest + 0.01);
    let (w, plastic, stored) = drive(&mut s, j, &mut pos, m, &line(start, target, 20_000));
    println!("pressed 1 cm past its rest: work {w:.4} J, plastic {plastic}, stored {stored:.4} J (½ k e² = {:.4})", 0.5 * f64::from(b0.k) * 1e-4);
    assert_eq!(plastic, 0.0);
    assert!((stored - 0.5 * f64::from(b0.k) * 1e-4).abs() < 2e-3 * stored);
    assert!((w - stored).abs() < 2e-3 * w);
}

#[test]
fn crash_metrics_barrier_shear_lbeam_work() {
    // A shear LBEAM (face node f8_3, the neighbouring back-plate node p9_3, vertex p8_3):
    // the face node pushed sideways 8 cm, past the plate shear yield (10 % of the 0.381 m
    // depth), and back: the solver's work on it is the ledger's stored energy plus the
    // LBEAM's plastic work, and back at the start it stores only what the set leaves.
    let (mut s, names) = barrier();
    let mut pos: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
    let id = |n: &str| names.iter().position(|x| x == n).unwrap() as u32;
    let j = s.beams.iter().position(|b| b.kind == ExplicitKind::JbeamLbeam && b.a == id("f8_3") && b.b == id("p9_3")).expect("lbeam");
    let b0 = s.beams[j];
    println!("lbeam: k {} tc {} cos {}", b0.k, b0.tc, b0.rest);
    assert!(b0.tc.is_finite());
    let m = b0.a as usize;
    let side = (pos[id("p9_3") as usize] - pos[id("p8_3") as usize]).normalize();
    let start = pos[m];
    let (w, plastic, stored) = drive(&mut s, j, &mut pos, m, &line(start, start + side * 0.08, 40_000));
    println!("sheared 8 cm: work {w:.3} J, plastic {plastic:.3} J, stored {stored:.3} J");
    assert!(plastic > 0.0, "it yields");
    assert!((w - (plastic + stored)).abs() < 5e-3 * w, "{w} vs {}", plastic + stored);
    let path = line(pos[m], start, 40_000);
    let (w2, plastic2, stored2) = drive(&mut s, j, &mut pos, m, &path);
    println!("back: work {w2:.3} J, plastic {plastic2:.3} J, stored {stored2:.3} J");
    let total = w + w2;
    assert!((total - (plastic + plastic2 + stored2)).abs() < 5e-3 * (plastic + plastic2 + stored2), "{total} vs {}", plastic + plastic2 + stored2);
}

/// A rigid body into a wall with a half-sine deceleration a(τ) = −A sin(πτ/T) from first
/// contact (τ = t − t0), sampled every 4 ms as a run is: the series' scalars against their
/// closed forms. v(τ) = v0 − (AT/π)(1 − cos(πτ/T)); it stops where cos(πτ/T) = 1 − πv0/(AT);
/// x(τ) = v0 τ − (AT/π)(τ − (T/π) sin(πτ/T)); after T it leaves at v0 − 2AT/π. The force on
/// the wall is M A sin(πτ/T) and its impulse M (v0 − v(T)).
#[test]
fn crash_series_half_sine_pulse_closed_form() {
    use std::f64::consts::PI;
    let d = lattice();
    let mass: f64 = d.nodes.iter().map(|n| f64::from(n.mass)).sum();
    let yaw = 0.3f32;
    let (t, mut b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, yaw, Vec3::ZERO)], vec![(Vec3::new(0.0, 1.0, 50.0), Vec3::new(5.0, 1.0, 0.5))]);
    let (a, tp, v0, t0) = (40.0 * series::G0, 0.100, 15.6, 0.1);
    b.layout.impact_s = t0 as f32;
    let h = b.layout.bodies[0].heading().as_dvec3();
    let vel = |tau: f64| if tau <= 0.0 { v0 } else { v0 - a * tp / PI * (1.0 - (PI * tau.min(tp) / tp).cos()) };
    let x_at = |tau: f64| {
        let c = |s: f64| v0 * s - a * tp / PI * (s - tp / PI * (PI * s / tp).sin());
        if tau <= 0.0 {
            v0 * tau
        } else if tau <= tp {
            c(tau)
        } else {
            c(tp) + vel(tp) * (tau - tp)
        }
    };
    let x0 = sample(&b, 0.0);
    let dt = f64::from(super::super::DT * super::super::SAMPLE_STEPS as f32);
    let xs: Vec<Sample> = (0..=100)
        .map(|k| {
            let tk = k as f64 * dt;
            let mut x = x0.clone();
            x.t = tk as f32;
            let (dx, v) = ((h * (x_at(tk - t0) - x_at(-t0))).as_vec3(), (h * vel(tk - t0)).as_vec3());
            x.pos[0].iter_mut().for_each(|p| *p += dx);
            x.vel[0].iter_mut().for_each(|w| *w = v);
            x
        })
        .collect();
    let mut r = Recorder::new(&t, &b);
    for x in &xs {
        r.push(x);
    }
    let (m, s) = r.finish_with_series();
    let stop = tp / PI * (1.0 - PI * v0 / (a * tp)).acos();
    let leave = vel(tp);
    // The 8 ms box-car keeps a sine to sin(πΔ/T)/(πΔ/T), Δ = 4 ms (0.26 %), and the
    // samples nearest the peak are 2 ms either side of it (τ = 48 and 52 ms: 0.20 % more).
    let boxcar = (PI * dt / tp).sin() / (PI * dt / tp);
    let near_peak = (PI * 0.048 / tp).sin();
    near(&m, "car.ref.nodes", 0.0, 0.0);
    near(&m, "car.com.peak_decel_g", 40.0 * boxcar * near_peak, 1e-3);
    // (48 or 52 ms: the two are equal but for rounding.)
    assert!((m["car.com.peak_ms"] - 50.0).abs() - 2.0 < 1e-3, "{}", m["car.com.peak_ms"]);
    near(&m, "car.com.stop_ms", 1000.0 * stop, 0.3);
    near(&m, "car.com.dyn_crush", x_at(stop), 3e-4);
    near(&m, "car.com.rebound", -leave, 1e-4);
    near(&m, "car.com.restitution", -leave / v0, 1e-5);
    near(&m, "car.com.vy_peak", 0.0, 1e-5);
    near(&m, "all.box_force_peak_n", mass * a * boxcar * near_peak, 1e-4 * mass * a);
    near(&m, "all.box_impulse_ns", mass * (v0 - leave), 1e-5 * mass * v0);
    near(&m, "all.box_force_pre_n", 0.0, 1e-5 * mass * a);
    // The curves: on the sample grid from first contact, and ∫F dt over the pulse = M Δv.
    let k0 = s.t_ms.iter().position(|x| x.abs() < 1e-3).unwrap();
    assert!((s.bodies[0].com.dx[k0]).abs() < 1e-5 && (s.bodies[0].com.vx[k0] - v0).abs() < 1e-5);
    let impulse = signal::integrate(&s.box_force_n[k0..], dt, 0.0).last().copied().unwrap();
    near(&BTreeMap::from([("impulse".to_string(), impulse)]), "impulse", mass * (v0 - leave), 2e-3 * mass * v0);
    for (k, &tk) in s.t_ms.iter().enumerate() {
        let tau = tk / 1000.0;
        assert!((s.bodies[0].com.vx[k] - vel(tau)).abs() < 1e-5, "v at {tk}");
        assert!((s.bodies[0].com.dx[k] - (x_at(tau) - x_at(0.0))).abs() < 5e-5, "x at {tk}");
    }
}

/// The reference location: the cabin floor behind the B-pillar (the `cabin_width` gauge's
/// node) and ahead of the rear axle, within 5 cm of the cabin's lowest node.
#[test]
fn crash_series_reference_nodes_by_position() {
    let p = |n: &str, z: f32, y: f32| node(n, [0.5, y, z], 1.0);
    let mut d = desc(
        vec![
            p("cab_f1", -1.0, 0.2),
            p("cab_f2", -2.0, 0.2),
            p("cab_f3", -2.4, 0.23),
            p("cab_f4", -2.9, 0.2),
            p("cab_hi", -2.0, 0.5),
            p("cab_bp", -1.6, 0.9),
            p("rr_low", -2.2, 0.1),
            p("hub_f1", 0.0, 0.3),
            p("hub_f2", 0.0, 0.3),
            p("hub_r1", -2.8, 0.3),
            p("hub_r2", -2.8, 0.3),
        ],
        &[],
    );
    d.wheels = vec![crate::desc::SoftWheel { node1: 7, node2: 8, ..Default::default() }, crate::desc::SoftWheel { node1: 9, node2: 10, ..Default::default() }];
    let zones = ZoneMap { parts: vec![("cab_".into(), Section::Cabin), ("rr_".into(), Section::Rear), ("hub_".into(), Section::Wheel)], gauges: vec![super::super::Gauge { name: "cabin_width".into(), a: "cab_bp".into(), b: "cab_bp".into() }] };
    let t = Template { desc: d, settings: Default::default(), zones };
    assert_eq!(reference_nodes(&t), vec![1, 2]);
    // Without wheels there is no rear axle: the centre of mass is the reference.
    let mut bare = t.clone();
    bare.desc.wheels.clear();
    assert!(reference_nodes(&bare).is_empty());
}

/// Side crush: a door-band skin node pushed 0.1 m in on the left reads 0.1 m of left crush
/// (residual and peak), nothing on the right, after the body has also turned and moved.
#[test]
fn crash_series_side_crush_in_the_fitted_frame() {
    let mut n = Vec::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                n.push(node(&format!("n{i}{j}{k}"), [-0.8 + 0.8 * i as f32, 0.5 * j as f32, -1.0 + k as f32], 1.0));
            }
        }
    }
    let mut d = desc(n, &[]);
    let id = |i: usize, j: usize, k: usize| (9 * i + 3 * j + k) as u32;
    // Left (+X) face and right face: a triangle with two door-band nodes (y = 0.5).
    d.triangles.push(SoftTriangle { nodes: [id(2, 1, 0), id(2, 1, 1), id(2, 2, 1)], ..Default::default() });
    d.triangles.push(SoftTriangle { nodes: [id(0, 1, 0), id(0, 1, 1), id(0, 2, 1)], ..Default::default() });
    let (t, b) = scene(&[("car", &d, ZoneMap::default(), Vec3::ZERO, 0.0, Vec3::ZERO)], vec![]);
    let x0 = sample(&b, 0.0);
    let mut x1 = x0.clone();
    x1.t = 0.004;
    x1.pos[0][id(2, 1, 1) as usize].x -= 0.1;
    // Then the whole body turned 20° about +Y and moved.
    let x2 = moved(&x1, Mat3::from_rotation_y(0.35), Vec3::ZERO, Vec3::new(1.0, 0.0, 2.0), 0.008);
    let m = run(&t, &b, &[x0, x1, x2]);
    near(&m, "car.side_crush_left_residual", 0.1, 1e-4);
    near(&m, "car.side_crush_left_peak", 0.1, 1e-4);
    near(&m, "car.side_crush_right_residual", 0.0, 1e-4);
}

// (Split out: this file is at the 1000-line limit.)
mod com_order;
