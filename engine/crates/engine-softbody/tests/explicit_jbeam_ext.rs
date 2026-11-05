//! JBeam mode Part II (`engine/docs/jbeam-semantics-spec.md` §16–§27), from the
//! documented semantics and physical invariants, on small original vehicles.
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, JbeamWheels, Node, STEP,
};
use engine_softbody::explicit_jbeam_ext::{
    jbeam_ext_force, lbeam_force, pressure_tri_force, BeamExt, ATMOSPHERE, PSI,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::{
    PART2_CAR, PART2_GROUPS, PART2_PANEL, PART2_RAIL, PART2_RAMP,
};
use engine_softbody::SoftVehicleDesc;
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    }
}

fn spawn_text(
    text: &str,
    wheels: JbeamWheels,
    v: Vec3,
) -> (ExplicitBeams, SoftVehicleDesc, Vec<String>) {
    let desc = load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let (s, w) =
        ExplicitBeams::from_desc_jbeam_with(&desc, wheels, Vec3::new(0.0, 1.0, 0.0), v, settings())
            .unwrap();
    (s, desc, w)
}

fn soft() -> JbeamWheels {
    JbeamWheels::Soft(engine_softbody::truck_export::TyreDefaults::for_vehicle(
        &SoftVehicleDesc::default(),
    ))
}

fn node(d: &SoftVehicleDesc, name: &str) -> u32 {
    d.nodes.iter().position(|n| n.name == name).unwrap() as u32
}

fn beam(s: &ExplicitBeams, d: &SoftVehicleDesc, a: &str, b: &str) -> usize {
    let (a, b) = (node(d, a), node(d, b));
    s.beams
        .iter()
        .position(|x| (x.a, x.b) == (a, b) || (x.a, x.b) == (b, a))
        .unwrap()
}

fn node_at(pos: Vec3, vel: Vec3) -> Node {
    Node {
        pos,
        vel,
        force: Vec3::ZERO,
        mass: 1.0,
        friction: 1.0,
        flags: 0,
    }
}

#[test]
fn several_groups_broken_by_any_without_propagation() {
    let (mut s, d, _) = spawn_text(PART2_GROUPS, soft(), Vec3::ZERO);
    let (a, b, c) = (
        beam(&s, &d, "a0", "a1"),
        beam(&s, &d, "b0", "b1"),
        beam(&s, &d, "c0", "c1"),
    );
    assert_eq!(
        s.beam_ext.len(),
        1,
        "only the beam with a table has an extension"
    );
    assert_ne!(s.ext_of[b], 0);
    // Pull a1 off: beam a breaks on its own, sets off g1, which takes b (in g1 and
    // g2), but not c (g2 only: groups do not propagate).
    s.nodes[node(&d, "a1") as usize].pos.x += 0.05;
    for _ in 0..3 {
        s.step();
    }
    assert!(s.beams[a].broken && s.beams[b].broken);
    assert!(
        !s.beams[c].broken,
        "a beam broken by g1 does not set off g2"
    );
}

#[test]
fn a_beam_in_several_groups_sets_them_all_off_the_next_step() {
    let (mut s, d, _) = spawn_text(PART2_GROUPS, soft(), Vec3::ZERO);
    let (a, b, c) = (
        beam(&s, &d, "a0", "a1"),
        beam(&s, &d, "b0", "b1"),
        beam(&s, &d, "c0", "c1"),
    );
    s.nodes[node(&d, "b1") as usize].pos.x += 0.05;
    s.step();
    assert!(s.beams[b].broken);
    assert!(
        !s.beams[a].broken && !s.beams[c].broken,
        "set off one step later"
    );
    s.step();
    assert!(s.beams[a].broken && s.beams[c].broken, "both of its groups");
}

#[test]
fn triangles_break_with_their_sides_unless_disabled() {
    let (mut s, d, _) = spawn_text(PART2_PANEL, soft(), Vec3::ZERO);
    assert_eq!(s.tri_breaks.len(), 2);
    assert!(
        s.tri_breaks[1].iter().all(|&j| j == u32::MAX),
        "disableTriangleBreaking"
    );
    s.nodes[node(&d, "p1") as usize].pos.x += 0.05;
    s.nodes[node(&d, "q1") as usize].pos.x += 0.05;
    s.step();
    assert!(s.beams[beam(&s, &d, "p0", "p1")].broken && s.beams[beam(&s, &d, "q0", "q1")].broken);
    assert!(!s.tri_dead[0], "dead from the step after");
    s.step();
    assert!(s.tri_dead[0] && !s.tri_dead[1]);
}

#[test]
fn deform_groups_trigger_on_yield_past_the_ratio_or_on_break() {
    let (mut s, d, _) = spawn_text(PART2_PANEL, soft(), Vec3::ZERO);
    assert_eq!(s.deform_names, vec!["glass".to_string()]);
    assert_eq!(s.deform_triggered(), &[0]);
    // A small elastic stretch: below the yield, no trigger.
    s.nodes[node(&d, "p2") as usize].pos.z -= 0.002;
    s.step();
    assert_eq!(s.deform_triggered(), &[0]);
    // Past the deform force: the rest length moves by more than 1 %.
    s.nodes[node(&d, "p2") as usize].pos.z -= 0.012;
    for _ in 0..2 {
        s.step();
    }
    assert_eq!(s.deform_triggered(), &[1]);
}

fn ext_beam(
    kind: ExplicitKind,
    k: f32,
    c: f32,
    rest: f32,
) -> engine_softbody::explicit_beams::Beam {
    let mut b = engine_softbody::explicit_beams::Beam::new(
        0,
        1,
        kind,
        engine_softbody::explicit_beams::ExplicitType::Deformable,
        k,
        c,
        rest,
        f32::INFINITY,
        f32::INFINITY,
        0.0,
    );
    b.long_bound = f32::INFINITY;
    b
}

#[test]
fn anisotropic_rates_switch_at_rest_over_the_transition_zone() {
    let mut b = ext_beam(ExplicitKind::JbeamAniso, 1000.0, 0.0, 1.0);
    let mut x = BeamExt {
        e1: [5000.0, 0.0, 0.1, f32::INFINITY],
        ..Default::default()
    };
    let o = node_at(Vec3::ZERO, Vec3::ZERO);
    let f = |b: &mut _, x: &mut _, l: f32| {
        jbeam_ext_force(b, x, &node_at(Vec3::X * l, Vec3::ZERO), &o, None).force
    };
    assert!(
        (f(&mut b, &mut x, 0.98) - 20.0).abs() < 1e-3,
        "compression: k"
    );
    assert!(
        (f(&mut b, &mut x, 1.2) + 1000.0).abs() < 1e-2,
        "past the zone: k expansion"
    );
    // Halfway through the zone (5 cm): k + (ke − k)·0.5.
    assert!((f(&mut b, &mut x, 1.05) + 0.05 * 3000.0).abs() < 1e-2);
    // Breaks past its length.
    x.e1[3] = 1.5;
    jbeam_ext_force(
        &mut b,
        &mut x,
        &node_at(Vec3::X * 1.6, Vec3::ZERO),
        &o,
        None,
    );
    assert!(b.broken);
}

#[test]
fn pressured_beam_follows_the_gas_law_on_its_surface() {
    let p0 = 2.0 * ATMOSPHERE; // gauge
    let mut b = ext_beam(ExplicitKind::JbeamPressured, 0.0, 0.0, 1.0);
    let mut x = BeamExt {
        e1: [0.01, ATMOSPHERE + p0, 1.0, 1.0],
        e2: [f32::INFINITY, f32::INFINITY, 0.0, 0.0],
        ..Default::default()
    };
    let o = node_at(Vec3::ZERO, Vec3::ZERO);
    let at = |b: &mut _, x: &mut _, l: f32| {
        jbeam_ext_force(b, x, &node_at(Vec3::X * l, Vec3::ZERO), &o, None)
    };
    let f1 = at(&mut b, &mut x, 1.0).force;
    assert!((f1 - p0 * 0.01).abs() < 1e-2, "spawn: gauge × surface");
    // Half the length: twice the absolute pressure.
    let f2 = at(&mut b, &mut x, 0.5).force;
    let want = (2.0 * (ATMOSPHERE + p0) - ATMOSPHERE) * 0.01;
    assert!((f2 - want).abs() < 0.1, "{f2} {want}");
    // volumeCoef 0: pressure never changes.
    x.e1[2] = 0.0;
    assert!((at(&mut b, &mut x, 0.5).force - f1).abs() < 1e-3);
    // maxPressure breaks it.
    x.e1[2] = 1.0;
    x.e2[1] = 3.0 * ATMOSPHERE;
    assert!(at(&mut b, &mut x, 0.5).check_break);
    assert!(PSI > 6894.0 && PSI < 6895.0);
}

#[test]
fn rebound_damping_takes_the_expansion_values() {
    let mut b = ext_beam(ExplicitKind::JbeamBounded, 0.0, 100.0, 1.0);
    b.tc = f32::INFINITY; // split
    b.tt = 100.0;
    b.short_bound = 1.0;
    b.long_bound = 1.0;
    b.limit_k = 0.0;
    let mut x = BeamExt {
        mask: engine_softbody::explicit_jbeam_ext::EXT_REBOUND,
        e2: [300.0, 300.0, f32::INFINITY, 0.0],
        ..Default::default()
    };
    let o = node_at(Vec3::ZERO, Vec3::ZERO);
    let at = |b: &mut _, x: &mut _, v: f32| {
        jbeam_ext_force(b, x, &node_at(Vec3::X, Vec3::X * v), &o, None).force
    };
    assert!(
        (at(&mut b, &mut x, -1.0) - 100.0).abs() < 1e-3,
        "compression: beamDamp"
    );
    assert!(
        (at(&mut b, &mut x, 1.0) + 300.0).abs() < 1e-3,
        "expansion: beamDampRebound"
    );
}

#[test]
fn precompression_ramps_in_over_its_time() {
    let (mut s, _, _) = spawn_text(PART2_RAMP, soft(), Vec3::ZERO);
    assert!((s.beams[0].rest - 1.0).abs() < 1e-6);
    assert!((s.beams[0].rest0 - 0.8).abs() < 1e-6);
    let n = (0.01 / STEP).round() as usize;
    for _ in 0..n / 2 {
        s.step();
    }
    assert!((s.beams[0].rest - 0.9).abs() < 1e-4, "{}", s.beams[0].rest);
    for _ in 0..n {
        s.step();
    }
    assert!((s.beams[0].rest - 0.8).abs() < 1e-4);
}

#[test]
fn lbeam_resists_angle_not_arm_length_and_conserves_momentum() {
    // Vertex at the origin, arms along x and y: right angle.
    let mut b = ext_beam(ExplicitKind::JbeamLbeam, 1000.0, 0.0, 0.0);
    b.tc = f32::INFINITY;
    b.rest = 0.0; // cos 90°
    let c = node_at(Vec3::ZERO, Vec3::ZERO);
    let at = |b: &mut _, pa: Vec3, pb: Vec3| {
        lbeam_force(
            b,
            &node_at(pa, Vec3::ZERO),
            &node_at(pb, Vec3::ZERO),
            &c,
            None,
        )
    };
    assert!(at(&mut b, Vec3::X, Vec3::Y).force.abs() < 1e-3);
    assert!(
        at(&mut b, Vec3::X * 1.5, Vec3::Y).force.abs() < 1e-3,
        "arm stretch: no force"
    );
    // Close the angle to 60°: the ends are pushed apart (positive force).
    let pb = Vec3::new(0.5, 3f32.sqrt() / 2.0, 0.0);
    let o = at(&mut b, Vec3::X, pb);
    assert!(o.force > 0.0);
    // Equal and opposite along id1–id2: momentum and angular momentum.
    let push = o.d * (o.force * o.inv);
    let torque = Vec3::X.cross(push) + pb.cross(-push);
    assert!(torque.length() < 1e-3);
}

#[test]
fn pressure_on_a_closed_surface_sums_to_zero() {
    // A tetrahedron, outward wound.
    let x = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let tris = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mut sum = Vec3::ZERO;
    for t in tris {
        sum += 3.0 * pressure_tri_force(x[t[0]], x[t[1]], x[t[2]], 1000.0);
    }
    assert!(sum.length() < 1e-3, "{sum}");
}

fn lowest(s: &ExplicitBeams) -> f32 {
    s.nodes
        .iter()
        .map(|n| n.pos.y)
        .fold(f32::INFINITY, f32::min)
}

fn kinetic(s: &ExplicitBeams) -> f32 {
    s.nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum()
}

#[test]
fn pressure_wheel_car_stands_on_its_tyres() {
    let (mut s, d, w) = spawn_text(PART2_CAR, JbeamWheels::Pressure, Vec3::ZERO);
    assert_eq!(d.pressure_wheels.len(), 4);
    assert!(w.iter().any(|w| w.contains("pressure wheels")), "{w:?}");
    let wheel_nodes = s.nodes.len() - d.nodes.len();
    assert_eq!(wheel_nodes, 4 * 4 * 12);
    assert!(s.beams.iter().any(|b| b.kind == ExplicitKind::JbeamLbeam));
    assert!(s.beams.iter().any(|b| b.kind == ExplicitKind::JbeamAniso));
    assert_eq!(
        s.beams
            .iter()
            .filter(|b| b.kind == ExplicitKind::JbeamPressureGroup)
            .count(),
        4
    );
    // Lowest node 2 cm above the ground.
    let drop = lowest(&s) - 0.02;
    for n in &mut s.nodes {
        n.pos.y -= drop;
    }
    let chassis_y0 = s.nodes[node(&d, "flt") as usize].pos.y;
    s.run(4.0);
    let ke = kinetic(&s);
    println!("kinetic energy at 4 s: {ke} J");
    // Settled to chatter: about 1 J of the car rocking on its 12-ray tyres'
    // flats (±3 cm at about 1 Hz, spec §28) since the ground rule's normal part;
    // 0.25 J before, when the tread nodes' bounce on the ground damped it.
    assert!(ke < 2.0, "settled: {ke} J");
    assert_eq!(s.damage().0, 0, "nothing broke");
    let sag = chassis_y0 - s.nodes[node(&d, "flt") as usize].pos.y;
    assert!(sag > 0.0 && sag < 0.08, "tyres carry the car: sag {sag}");
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
}

#[test]
fn pressure_wheel_car_rolls_straight() {
    let v = Vec3::new(0.0, 0.0, 5.0);
    let (mut s, _, _) = spawn_text(PART2_CAR, JbeamWheels::Pressure, v);
    let drop = lowest(&s) - 0.005;
    for n in &mut s.nodes {
        n.pos.y -= drop;
    }
    s.roll_tyres(v);
    let mid = |s: &ExplicitBeams| s.nodes[..8].iter().map(|n| n.pos.x).sum::<f32>() / 8.0;
    let (x0, z0) = (mid(&s), s.nodes[0].pos.z);
    s.run(1.0);
    let travel = s.nodes[0].pos.z - z0;
    assert!(travel > 3.5, "rolls on: {travel} m in 1 s at 5 m/s");
    let side = mid(&s) - x0;
    assert!(side.abs() < 0.1, "straight: {side} m off its line");
    assert_eq!(s.damage().0, 0);
}

#[test]
fn slide_node_hangs_on_its_rail() {
    let (mut s, d, _) = spawn_text(PART2_RAIL, soft(), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(s.slides.len(), 1);
    let i = node(&d, "s") as usize;
    s.nodes[i].vel = Vec3::new(0.5, 0.0, 0.0);
    s.run(0.5);
    let p = s.nodes[i].pos - Vec3::new(0.0, 1.0, 0.0);
    assert!(
        p.y.abs() < 1e-3 && p.z.abs() < 1e-3,
        "held on the rail under gravity: {p}"
    );
    assert!(p.x > 0.2, "free along it: {p}");
}

/// Rest creep (spec §28): the horizontal centre of mass sampled every 0.1 s over
/// `window` s after `settle` s; returns (fitted drift over the window, the sway
/// about that fit), both the larger of x and z, in metres. Exact physics: at rest
/// on flat ground no net horizontal force acts, so a straight-line fit of the
/// centre of mass moves no more than the sway (the at-rest chatter) around it.
fn creep(s: &mut ExplicitBeams, settle: f32, window: f32) -> (f32, f32) {
    s.run(settle);
    let mass: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let mut samples = Vec::new();
    let n = (window / 0.1).round() as usize;
    for k in 0..=n {
        if k > 0 {
            s.run(0.1);
        }
        let c = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / mass;
        samples.push((0.1 * k as f64, c));
    }
    let (mut drift, mut sway) = (0.0f32, 0.0f32);
    for axis in [0usize, 2] {
        let t: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let y: Vec<f64> = samples.iter().map(|s| s.1[axis] as f64).collect();
        let (tm, ym) = (t.iter().sum::<f64>() / t.len() as f64, y.iter().sum::<f64>() / y.len() as f64);
        let slope = t.iter().zip(&y).map(|(a, b)| (a - tm) * (b - ym)).sum::<f64>()
            / t.iter().map(|a| (a - tm).powi(2)).sum::<f64>();
        let resid = t
            .iter()
            .zip(&y)
            .map(|(a, b)| (b - ym - slope * (a - tm)).abs())
            .fold(0.0, f64::max);
        drift = drift.max((slope * window as f64).abs() as f32);
        sway = sway.max(resid as f32);
    }
    (drift, sway)
}

#[test]
fn pressure_wheel_car_does_not_creep_at_rest() {
    let (mut s, _, _) = spawn_text(PART2_CAR, JbeamWheels::Pressure, Vec3::ZERO);
    let drop = lowest(&s) - 0.02;
    for n in &mut s.nodes {
        n.pos.y -= drop;
    }
    let (drift, sway) = creep(&mut s, 2.0, 4.0);
    println!("fixture car: drift {drift} m over 4 s, sway {sway} m");
    // Rounding: 1 mm.
    assert!(drift <= sway + 1e-3, "creeps {drift} m (sway {sway} m)");
}

/// The normal part of the pressure-wheel ground rule (spec §28): a light node
/// under a heavy load, 9 mm deep, rises at most half its depth in a step.
#[test]
fn loaded_tread_node_is_pushed_to_the_surface_not_past_it() {
    use engine_softbody::explicit_beams::{contact_force, ContactSurface};
    use engine_softbody::explicit_jbeam_ext::friction_limit;
    let (m, f, p) = (0.12, Vec3::new(0.0, -5000.0, 0.0), 0.009);
    for vy in [-0.5f32, 0.0, 3.0] {
        let v = Vec3::new(0.0, vy, 0.0);
        let c = contact_force(f, v, m, Vec3::Y, STEP, &ContactSurface::GROUND, p, 1.0);
        let c = friction_limit(c, f, v, m, Vec3::Y, STEP, p);
        let out = vy + (f.y + c.y) * STEP / m;
        assert!(out * STEP <= 0.5 * p * (1.0 + 1e-4), "vy {vy}: rises {out} m/s");
        assert!(c.y >= 0.0);
    }
}

/// Community vehicles (opt-in, `RUBBLE_BEAMNG_MODS=<inspiration/beamng-mods>`):
/// no rest creep with pressure wheels.
#[test]
fn community_vehicles_do_not_creep_at_rest() {
    let Ok(mods) = std::env::var("RUBBLE_BEAMNG_MODS") else {
        eprintln!("skip: RUBBLE_BEAMNG_MODS not set");
        return;
    };
    let m = std::path::Path::new(&mods);
    for (dir, name, pc) in [
        ("monster_supercar/vehicles", "supercar", "config.pc"),
        ("qlourie_astral_r/vehicles", "qlourie_astral_r", "wrc17_tarmac_balanced.pc"),
        ("rls_forklift/vehicles", "TPUMedForklift", "FL45D Triplex.pc"),
    ] {
        let dir = m.join(dir);
        let Ok(files) = engine_softbody::jbeam::read_vehicle_dir(&dir, name) else {
            continue;
        };
        let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
        let desc = load_jbeam(&refs, Some(&pcs)).unwrap();
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
            &desc,
            JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings(),
        )
        .unwrap();
        let drop = lowest(&s) - 0.02;
        for n in &mut s.nodes {
            n.pos.y -= drop;
        }
        let (drift, sway) = creep(&mut s, 2.0, 8.0);
        println!("{name}: drift {drift} m over 8 s, sway {sway} m");
        assert!(drift <= sway + 1e-3, "{name} creeps {drift} m (sway {sway} m)");
    }
}


/// Triangle aerodynamics (spec §31), exact for one plate in still air: normal to
/// its motion, drag ½ρ·1.28·(dragCoef/100)·A·v² against the motion and no lift; at
/// 30° to it, drag with sin α and lift with sin α cos α across the motion, toward
/// the side the flow pushes; past the stall the lift falls as sin α_s cos α.
#[test]
fn triangle_drag_and_lift_follow_the_flat_plate_law() {
    use engine_softbody::explicit_jbeam_ext::{aero_tri_force, AIR_DENSITY, FLAT_PLATE_CD};
    let k = 0.5 * AIR_DENSITY * FLAT_PLATE_CD * 0.01;
    let ext = BeamExt { e1: [0.0, k * 100.0, k * 80.0, 0.58f32.sin()], ..Default::default() };
    // A 1 m × 1 m right triangle in the x–y plane (normal +z), area 0.5 m².
    let x = [Vec3::ZERO, Vec3::X, Vec3::Y];
    let v = 20.0;
    let f = 3.0 * aero_tri_force(x, [Vec3::Z * v; 3], &ext);
    let drag = 0.5 * AIR_DENSITY * FLAT_PLATE_CD * 0.5 * v * v;
    assert!((f.z + drag).abs() < 1e-3 * drag && f.x.abs() < 1e-3 && f.y.abs() < 1e-3, "{f:?} vs {drag}");
    // At 30°: moving along (0, −cos 30°, −sin 30°)·v: the flow comes from below-ahead.
    let a = 30f32.to_radians();
    let dir = Vec3::new(0.0, -a.cos(), -a.sin());
    let f = 3.0 * aero_tri_force(x, [dir * v; 3], &ext);
    let q = 0.5 * AIR_DENSITY * FLAT_PLATE_CD * 0.5 * v * v;
    let along = f.dot(-dir);
    let across = (f - (-dir) * along).length();
    assert!((along - q * a.sin()).abs() < 1e-3 * q, "drag {along} vs {}", q * a.sin());
    assert!((across - 0.8 * q * a.sin() * a.cos()).abs() < 1e-3 * q, "lift {across}");
    // The flow (along −dir: +y, +z) pushes the plate toward +z.
    assert!(f.z > 0.0);
    // Past the stall (0.58 rad), at 60°: lift 0.8·q·sin 0.58·cos 60°.
    let a = 60f32.to_radians();
    let dir = Vec3::new(0.0, -a.cos(), -a.sin());
    let f = 3.0 * aero_tri_force(x, [dir * v; 3], &ext);
    let along = f.dot(-dir);
    let across = (f - (-dir) * along).length();
    assert!((across - 0.8 * q * 0.58f32.sin() * a.cos()).abs() < 1e-3 * q, "stalled lift {across}");
}
