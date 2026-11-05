//! JBeam mode Part II (`engine/docs/jbeam-semantics-spec.md` §16–§27), from the
//! documented semantics and physical invariants, on small original vehicles.
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, JbeamWheels, Node, STEP,
};
use engine_softbody::explicit_jbeam_ext::{
    jbeam_ext_force, lbeam_force, pressure_tri_force, BeamExt, ATMOSPHERE, PSI,
};
use engine_softbody::jbeam::load_jbeam;
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

/// A ladder of nodes held up by fixed nodes (no gravity effects on the tests that
/// pull it apart by hand).
const GROUPS: &str = r#"
{ "t": {
  "information": {"name": "groups"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["a0", 0, 0, 0], ["a1", 1, 0, 0], ["b0", 0, 1, 0], ["b1", 1, 1, 0], ["c0", 0, 2, 0], ["c1", 1, 2, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 100000, "beamDamp": 0, "beamDeform": "FLT_MAX", "beamStrength": 1000},
            {"breakGroup": "g1"}, ["a0", "a1"],
            {"breakGroup": ["g1", "g2"]}, ["b0", "b1"],
            {"breakGroup": "g2"}, ["c0", "c1"],
            {"breakGroup": ""}, ["a0", "b0"], ["b0", "c0"], ["a1", "b1"], ["b1", "c1"]]
}}"#;

#[test]
fn several_groups_broken_by_any_without_propagation() {
    let (mut s, d, _) = spawn_text(GROUPS, soft(), Vec3::ZERO);
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
    let (mut s, d, _) = spawn_text(GROUPS, soft(), Vec3::ZERO);
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

const PANEL: &str = r#"
{ "t": {
  "information": {"name": "panel"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["p0", 0, 0, 0], ["p1", 1, 0, 0], ["p2", 0, 1, 0], ["q0", 2, 0, 0], ["q1", 3, 0, 0], ["q2", 2, 1, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 100000, "beamDamp": 0, "beamDeform": 500, "beamStrength": 1000,
             "deformGroup": "glass", "deformationTriggerRatio": 0.002},
            ["p0", "p1"], ["p1", "p2"], ["p2", "p0"],
            {"deformGroup": "", "disableTriangleBreaking": true},
            ["q0", "q1"], ["q1", "q2"], ["q2", "q0"]],
  "triangles": [["id1:", "id2:", "id3:"], ["p0", "p1", "p2"], ["q0", "q1", "q2"]]
}}"#;

#[test]
fn triangles_break_with_their_sides_unless_disabled() {
    let (mut s, d, _) = spawn_text(PANEL, soft(), Vec3::ZERO);
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
    let (mut s, d, _) = spawn_text(PANEL, soft(), Vec3::ZERO);
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

const RAMP: &str = r#"
{ "t": {
  "information": {"name": "ramp"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false, "fixed": true},
            ["a", 0, 0, 0], ["b", 1, 0, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 1000, "beamDamp": 0, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX",
             "beamPrecompression": 0.8, "beamPrecompressionTime": 0.01}, ["a", "b"]]
}}"#;

#[test]
fn precompression_ramps_in_over_its_time() {
    let (mut s, _, _) = spawn_text(RAMP, soft(), Vec3::ZERO);
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

/// A box chassis on four pressure wheels (original Rubble content).
const CAR: &str = r#"
{ "t": {
  "information": {"name": "pw car"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 40, "frictionCoef": 1.0},
            ["fl", 0.7, -1.2, 0.45], ["fr", -0.7, -1.2, 0.45], ["rl", 0.7, 1.2, 0.45], ["rr", -0.7, 1.2, 0.45],
            ["flt", 0.5, -1.2, 0.9], ["frt", -0.5, -1.2, 0.9], ["rlt", 0.5, 1.2, 0.9], ["rrt", -0.5, 1.2, 0.9],
            {"nodeWeight": 8},
            ["fli", 0.55, -1.2, 0.45], ["fri", -0.55, -1.2, 0.45], ["rli", 0.55, 1.2, 0.45], ["rri", -0.55, 1.2, 0.45],
            ["flo", 0.85, -1.2, 0.45], ["fro", -0.85, -1.2, 0.45], ["rlo", 0.85, 1.2, 0.45], ["rro", -0.85, 1.2, 0.45]],
  "beams": [["id1:", "id2:"], {"beamSpring": 2000000, "beamDamp": 200, "beamDeform": "FLT_MAX", "beamStrength": "FLT_MAX"},
            ["fl", "fr"], ["rl", "rr"], ["fl", "rl"], ["fr", "rr"], ["fl", "rr"], ["fr", "rl"],
            ["flt", "frt"], ["rlt", "rrt"], ["flt", "rlt"], ["frt", "rrt"], ["flt", "rrt"],
            ["fl", "flt"], ["fr", "frt"], ["rl", "rlt"], ["rr", "rrt"], ["fl", "frt"], ["rl", "rrt"], ["fl", "rlt"], ["fr", "rrt"],
            ["fli", "flo"], ["fri", "fro"], ["rli", "rlo"], ["rri", "rro"],
            ["fli", "fl"], ["flo", "fl"], ["fli", "flt"], ["flo", "flt"], ["fli", "rl"], ["flo", "rl"],
            ["fri", "fr"], ["fro", "fr"], ["fri", "frt"], ["fro", "frt"], ["fri", "rr"], ["fro", "rr"],
            ["rli", "rl"], ["rlo", "rl"], ["rli", "rlt"], ["rlo", "rlt"], ["rli", "fl"], ["rlo", "fl"],
            ["rri", "rr"], ["rro", "rr"], ["rri", "rrt"], ["rro", "rrt"], ["rri", "fr"], ["rro", "fr"]],
  "pressureWheels": [["name", "hubGroup", "group", "node1:", "node2:", "nodeS", "nodeArm:", "wheelDir"],
        {"numRays": 12, "radius": 0.32, "hubRadius": 0.2, "tireWidth": 0.2, "hubWidth": 0.22,
         "hubNodeWeight": 1.0, "nodeWeight": 0.25, "pressurePSI": 30, "frictionCoef": 1.0,
         "hubBeamSpring": 800000, "hubBeamDamp": 60, "hubSideBeamSpring": 800000, "hubSideBeamDamp": 60,
         "hubTreadBeamSpring": 800000, "hubTreadBeamDamp": 60, "hubPeripheryBeamSpring": 800000, "hubPeripheryBeamDamp": 60,
         "wheelSideBeamSpring": 20000, "wheelSideBeamDamp": 150, "wheelSideBeamSpringExpansion": 200000,
         "wheelSideBeamDampExpansion": 150, "wheelSideTransitionZone": 0.05,
         "wheelTreadBeamSpring": 200000, "wheelTreadBeamDamp": 30, "wheelPeripheryBeamSpring": 200000,
         "wheelPeripheryBeamDamp": 30, "enableTireLbeams": true, "wheelLbeamSpring": 5000, "wheelLbeamDamp": 5,
         "enableTreadReinfBeams": true, "wheelTreadReinfBeamSpring": 100000, "wheelTreadReinfBeamDamp": 20},
        ["FL", "wheel_FL", "tire_FL", "fli", "flo", 9999, "flt", 1],
        ["FR", "wheel_FR", "tire_FR", "fri", "fro", 9999, "frt", -1],
        ["RL", "wheel_RL", "tire_RL", "rli", "rlo", 9999, "rlt", 1],
        ["RR", "wheel_RR", "tire_RR", "rri", "rro", 9999, "rrt", -1]]
}}"#;

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
    let (mut s, d, w) = spawn_text(CAR, JbeamWheels::Pressure, Vec3::ZERO);
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
    assert!(ke < 1.0, "settled: {ke} J");
    assert_eq!(s.damage().0, 0, "nothing broke");
    let sag = chassis_y0 - s.nodes[node(&d, "flt") as usize].pos.y;
    assert!(sag > 0.0 && sag < 0.08, "tyres carry the car: sag {sag}");
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
}

#[test]
fn pressure_wheel_car_rolls_straight() {
    let v = Vec3::new(0.0, 0.0, 5.0);
    let (mut s, _, _) = spawn_text(CAR, JbeamWheels::Pressure, v);
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

const RAIL: &str = r#"
{ "t": {
  "information": {"name": "rail"}, "slotType": "main",
  "nodes": [["id", "posX", "posY", "posZ"], {"nodeWeight": 1, "collision": false},
            ["r0", -1, 0, 0, {"fixed": true}], ["r1", 1, 0, 0, {"fixed": true}], ["s", 0, 0, 0]],
  "beams": [["id1:", "id2:"], {"beamSpring": 1000, "beamDamp": 0}, ["r0", "r1"]],
  "rails": {"guide": {"links:": ["r0", "r1"], "looped": false, "capped": true}},
  "slidenodes": [["id:", "railName", "attached", "fixToRail", "tolerance", "spring", "strength", "capStrength"],
                 ["s", "guide", true, true, 0, 100000, "FLT_MAX", "FLT_MAX"]]
}}"#;

#[test]
fn slide_node_hangs_on_its_rail() {
    let (mut s, d, _) = spawn_text(RAIL, soft(), Vec3::new(0.0, 0.0, 0.0));
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
