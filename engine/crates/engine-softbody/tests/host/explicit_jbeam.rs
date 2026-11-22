//! The explicit solver's JBeam mode (`engine/docs/jbeam-semantics-spec.md`) on
//! Rubble's own fixture (`jbeam_fixture::latched_box_jbeam`): the rules one at a
//! time, then the whole vehicle at rest and in a crash.

use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{
    jbeam_force, ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, StaticBox,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::latched_box_jbeam;
use engine_softbody::truck_export::TyreDefaults;
use glam::Vec3;

fn desc() -> SoftVehicleDesc {
    let text = latched_box_jbeam();
    load_jbeam(&[("box.jbeam", text.as_str())], None).unwrap()
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    }
}

/// The fixture 1 cm above the ground, moving at `v`.
fn spawn(v: Vec3) -> (ExplicitBeams, SoftVehicleDesc) {
    let d = desc();
    let low = d
        .nodes
        .iter()
        .map(|n| n.position[1])
        .fold(f32::MAX, f32::min);
    let (s, _) = ExplicitBeams::from_desc_jbeam(
        &d,
        TyreDefaults::default(),
        Vec3::new(0.0, 0.01 - low, 0.0),
        v,
        settings(),
    )
    .unwrap();
    (s, d)
}

fn node(d: &SoftVehicleDesc, name: &str) -> u32 {
    d.nodes.iter().position(|n| n.name == name).unwrap() as u32
}

fn beam(s: &ExplicitBeams, d: &SoftVehicleDesc, a: &str, b: &str) -> usize {
    let (a, b) = (node(d, a), node(d, b));
    s.beams
        .iter()
        .position(|x| (x.a == a && x.b == b) || (x.a == b && x.b == a))
        .unwrap()
}

#[test]
fn kinds_precompression_and_groups_load() {
    let (s, d) = spawn(Vec3::ZERO);
    let pop = &s.beams[beam(&s, &d, "b2", "d4")];
    assert_eq!(pop.kind, ExplicitKind::JbeamSupport);
    assert!(pop.group < 0, "breakGroupType 1: passive");
    let spawn_len = s.nodes[pop.a as usize]
        .pos
        .distance(s.nodes[pop.b as usize].pos);
    assert!(
        (pop.rest / spawn_len - 1.02).abs() < 1e-5,
        "precompression on a support"
    );
    assert!(
        (pop.long_bound - 26.0 * spawn_len).abs() < 1e-4,
        "break past (1 + 25) × spawn length"
    );
    let hinge = &s.beams[beam(&s, &d, "d1", "bh1")];
    assert_eq!(hinge.kind, ExplicitKind::JbeamPlain);
    assert_eq!(
        -pop.group, hinge.group,
        "the pop-open support is a passive member of hinge_a"
    );
    let detent = &s.beams[beam(&s, &d, "d3", "bh1")];
    assert_eq!(detent.kind, ExplicitKind::JbeamBounded);
    assert_eq!(
        (detent.limit_k, detent.limit_c, detent.plastic),
        (0.0, 0.0, 0.01)
    );
    let limiter = &s.beams[beam(&s, &d, "d3", "bh2")];
    assert_eq!(
        (limiter.tc, limiter.tt),
        (0.1, 50.0),
        "velocity split and fast damping"
    );
    assert!(limiter.shock > 0, "a damping filter (dampCutoffHz)");
    assert_eq!(
        s.beams
            .iter()
            .filter(|b| b.kind == ExplicitKind::JbeamCoupler)
            .count(),
        1,
        "the latch is coupled at spawn"
    );
    let shape = &s.beams[beam(&s, &d, "d1", "d2")];
    assert!(
        (shape.long_bound / shape.rest - 1.1).abs() < 1e-5,
        "deformLimitExpansion 0.1"
    );
    assert!(
        (shape.short_bound / shape.rest - 0.7).abs() < 1e-5,
        "deformLimit 0.3"
    );
    assert!(
        !s.settings.fast_inv_sqrt,
        "JBeam mode measures lengths exactly"
    );
}

#[test]
fn passive_members_break_with_their_group_but_never_set_it_off() {
    // The passive pop-open support breaking by force leaves its group alone.
    let (mut s, d) = spawn(Vec3::ZERO);
    let pop = beam(&s, &d, "b2", "d4");
    let hinge = beam(&s, &d, "d1", "bh1");
    s.beams[pop].strength = 1e-3;
    s.beams[pop].gate = 1e-3;
    for _ in 0..400 {
        s.step();
    }
    assert!(s.beams[pop].broken);
    assert!(
        !s.beams[hinge].broken,
        "a passive break does not set off its group"
    );
    // An active member breaking takes the group, passive members included.
    let (mut s, d) = spawn(Vec3::ZERO);
    s.beams[hinge].strength = 1e-3;
    s.beams[hinge].gate = 1e-3;
    for _ in 0..400 {
        s.step();
    }
    for (a, b) in [("d1", "bh1"), ("d1", "bh2"), ("dr", "bh1"), ("b2", "d4")] {
        assert!(
            s.beams[beam(&s, &d, a, b)].broken,
            "{a}-{b} breaks with hinge_a"
        );
    }
    for (a, b) in [("d2", "bh2"), ("d2", "bh1"), ("dr", "bh2")] {
        assert!(!s.beams[beam(&s, &d, a, b)].broken, "{a}-{b} is in hinge_b");
    }
}

#[test]
fn supports_push_only_and_break_by_length() {
    let (s, d) = spawn(Vec3::ZERO);
    let j = beam(&s, &d, "d1", "b2");
    let mut na = s.nodes[s.beams[j].a as usize];
    let nb = s.nodes[s.beams[j].b as usize];
    let dir = (na.pos - nb.pos).normalize();
    let rest = s.beams[j].rest;
    // Stretched (and moving apart): no force at all.
    na.pos = nb.pos + dir * rest * 1.5;
    na.vel = dir * 3.0;
    let mut b = s.beams[j];
    let out = jbeam_force(&mut b, &na, &nb, false);
    assert_eq!(out.force, 0.0);
    assert!(!b.broken);
    // Past (1 + 6) × the spawn length: broken, whatever the force.
    na.pos = nb.pos + dir * (b.long_bound + 0.01);
    let out = jbeam_force(&mut b, &na, &nb, false);
    assert!(b.broken && out.force == 0.0 && !out.check_break);
    // Compressed: pushes.
    let mut b = s.beams[j];
    na.pos = nb.pos + dir * rest * 0.99;
    na.vel = Vec3::ZERO;
    assert!(jbeam_force(&mut b, &na, &nb, false).force > 0.0);
}

#[test]
fn bounded_beams_move_to_their_limit_rates_over_the_bound_zone() {
    let (s, d) = spawn(Vec3::ZERO);
    let j = beam(&s, &d, "d3", "bh1");
    let mut na = s.nodes[s.beams[j].a as usize];
    let nb = s.nodes[s.beams[j].b as usize];
    let dir = (na.pos - nb.pos).normalize();
    let rest = s.beams[j].rest;
    let force_at = |x: f32, na: &mut engine_softbody::explicit_beams::Node| {
        na.pos = nb.pos + dir * x;
        let mut b = s.beams[j];
        jbeam_force(&mut b, na, &nb, false).force
    };
    // Inside the bounds: the beam's own spring (800 N/m).
    let f = force_at(rest * 0.95, &mut na);
    assert!((f - 800.0 * 0.05 * rest).abs() < 1e-2 * f.abs(), "{f}");
    // Past the bound by more than the 1 cm zone: the limit spring, 0.
    assert_eq!(force_at(rest * 0.85, &mut na), 0.0);
    // Halfway through the zone: half the spring.
    let e = -(0.1 * rest + 0.005);
    let f = force_at(rest + e, &mut na);
    assert!((f - 400.0 * -e).abs() < 1e-2 * f.abs(), "{f}");
}

#[test]
fn latched_box_stands_still() {
    let (mut s, _) = spawn(Vec3::ZERO);
    for _ in 0..4000 {
        s.step();
    }
    let (broken, bent) = s.damage();
    let ke: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    eprintln!("latched box after 2 s at rest: broken {broken}, bent {bent}, KE {ke:.4} J");
    assert_eq!((broken, bent), (0, 0));
    assert!(ke < 0.05, "KE {ke} J");
}

#[test]
fn latched_box_crash_breaks_without_exploding() {
    let (mut s, _) = spawn(Vec3::new(0.0, 0.0, 15.0));
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.05 + 0.5),
        Vec3::new(3.0, 2.0, 0.5),
    ));
    for _ in 0..2000 {
        s.step();
    }
    let (broken, bent) = s.damage();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    eprintln!(
        "latched box 15 m/s into a wall, 1 s: broken {broken}, bent {bent}, fastest {vmax:.2} m/s"
    );
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
    assert!(bent > 0, "the crash deforms the box");
    assert!(vmax < 20.0, "no node faster than the impact ({vmax} m/s)");
}

#[test]
fn self_collision_skips_the_nodes_own_group() {
    let (s, d) = spawn(Vec3::ZERO);
    assert!(!s.self_groups.is_empty());
    let door_tri = s
        .triangles
        .iter()
        .position(|t| t.nodes.contains(&node(&d, "d1")))
        .unwrap();
    let body_tri = s
        .triangles
        .iter()
        .position(|t| t.nodes.contains(&node(&d, "b2")))
        .unwrap();
    let g = s.triangle_self_group(door_tri);
    assert_eq!(g, s.self_groups[node(&d, "d2") as usize]);
    assert_ne!(s.triangle_self_group(body_tri), g);
    // Only selfCollision nodes are candidates.
    assert!(!s.candidates.contains(&node(&d, "dl")));
    assert!(s.candidates.contains(&node(&d, "d1")));
}

#[test]
fn dihedral_gradient_matches_the_f64_reference() {
    use engine_softbody::elements;
    use engine_softbody::explicit_beams::{dihedral, dihedral_jacobian};
    let x = [
        Vec3::new(0.3, -0.2, 0.9),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.1, 1.0, 0.05),
        Vec3::new(-0.7, 1.2, 0.4),
    ];
    let xd = x.map(|p| p.as_dvec3());
    assert!((dihedral(x) as f64 - elements::dihedral(xd)).abs() < 1e-6);
    for (g, r) in dihedral_jacobian(x)
        .iter()
        .zip(elements::dihedral_jacobian(xd))
    {
        assert!((g.as_dvec3() - r).length() < 1e-5, "{g} vs {r}");
    }
}

#[test]
fn torsion_bar_holds_its_angle_and_breaks_with_its_hinge() {
    let (s, d) = spawn(Vec3::ZERO);
    let t = s
        .beams
        .iter()
        .position(|b| b.kind == ExplicitKind::JbeamTorsion)
        .expect("the hinge torsion bar");
    assert_eq!(s.beams[t + 1].kind, ExplicitKind::JbeamTorsionPair);
    // Its links are watched: the detent d3-bh1's group (hinge_a, of which it is a
    // passive member), and the axis bh1-bh2, which got a group of its own.
    let watch = s.torsion_watch[s.beams[t].shock as usize - 1];
    let detent = beam(&s, &d, "d3", "bh1");
    let axis = beam(&s, &d, "bh1", "bh2");
    assert_eq!(watch[0], s.beams[detent].group.unsigned_abs());
    assert!(s.beams[axis].group > 0 && watch[1] == s.beams[axis].group as u32);
    assert_eq!(
        s.beams
            .iter()
            .filter(|b| b.group == s.beams[axis].group)
            .count(),
        1,
        "a group of its own"
    );
    // Twisted, it pushes back; the torque is −k·Δφ.
    let mut b = s.beams[t];
    let mut nodes = [b.a, b.b, s.beams[t + 1].a, s.beams[t + 1].b].map(|n| s.nodes[n as usize]);
    nodes[0].pos += Vec3::new(0.01, 0.0, 0.0);
    let f = engine_softbody::explicit_beams::torsion_force(&mut b, &nodes);
    assert!(f.iter().any(|f| f.length() > 0.0));
    assert!(
        b.force.abs() > 0.0 && b.force.abs() <= 2000.0 + 1e-3,
        "torque {}",
        b.force
    );
    // Its hinge group breaking (an active member, d1-bh1) breaks it the step after.
    let (mut s, _) = spawn(Vec3::ZERO);
    let hinge = beam(&s, &d, "d1", "bh1");
    s.beams[hinge].strength = 1e-3;
    s.beams[hinge].gate = 1e-3;
    for _ in 0..400 {
        s.step();
        if s.beams[hinge].broken {
            break;
        }
    }
    assert!(s.beams[hinge].broken && s.beams[detent].broken);
    assert!(!s.beams[t].broken, "not in the same step");
    s.step();
    assert!(s.beams[t].broken && s.beams[t + 1].broken);
}

/// BeamNG node couplers (spec §7.1) on `jbeam_fixture::NODE_COUPLERS`: which pairs
/// couple at spawn, their strength and groups; then the lid thrown off the body: the
/// active latch breaks past its strength and sets off its group, the passive coupler
/// (unbreakable alone) is broken by its second group's hinge beam.
#[test]
fn node_couplers_couple_at_spawn_and_break_by_strength_and_group() {
    use engine_softbody::jbeam_fixture::NODE_COUPLERS;
    let d = load_jbeam(&[("nc.jbeam", NODE_COUPLERS)], None).unwrap();
    let id = |n: &str| node(&d, n);
    let group = |g: u32| d.break_groups[g as usize - 1].as_str();
    assert_eq!(d.couplers.len(), 2, "{:?}", d.couplers);
    let latch = d.couplers.iter().find(|c| c.a == id("lc1")).unwrap();
    assert_eq!(latch.b, id("bc1"));
    assert_eq!(latch.strength, 2500.0);
    assert_eq!(group(latch.break_group), "latch_a");
    assert!(!latch.break_group_passive);
    let hinge = d.couplers.iter().find(|c| c.a == id("lc2")).unwrap();
    assert_eq!(hinge.b, id("bc2"));
    assert!(hinge.strength.is_infinite());
    assert!(hinge.break_group_passive);
    assert_eq!(
        [group(hinge.break_group), group(hinge.extra_break_groups[0])],
        ["hinge_l", "hinge_r"]
    );
    let warn = d.warnings.join("\n");
    assert!(
        warn.contains("1 node couplers have no couplerStartRadius"),
        "{warn}"
    );
    assert!(
        warn.contains("1 node couplers have no free tag node"),
        "{warn}"
    );

    let (mut s, _) = ExplicitBeams::from_desc_jbeam(
        &d,
        TyreDefaults::default(),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    let coupler = |s: &ExplicitBeams, a: &str| {
        s.beams
            .iter()
            .position(|b| b.kind == ExplicitKind::JbeamCoupler && b.a == id(a))
            .unwrap()
    };
    let (jl, jh) = (coupler(&s, "lc1"), coupler(&s, "lc2"));
    // At rest in free fall nothing breaks.
    s.run(0.1);
    assert!(s.beams.iter().all(|b| !b.broken));
    // Throw the lid up at 3 m/s.
    let lid = ["l1", "l2", "l3", "l4", "lc1", "lc2", "lc3", "lc4"].map(|n| id(n) as usize);
    for &i in &lid {
        s.nodes[i].vel.y += 3.0;
    }
    s.run(0.3);
    let latch_beam = beam(&s, &d, "l2", "b4");
    let hinge_beam = beam(&s, &d, "l1", "b4");
    assert!(s.beams[jl].broken, "the latch holds");
    assert!(
        s.beams[latch_beam].broken,
        "the latch's group was not set off"
    );
    assert!(s.beams[hinge_beam].broken, "the hinge beam holds");
    assert!(s.beams[jh].broken, "the passive coupler survived its group");
    let gap = s.nodes[id("lc2") as usize]
        .pos
        .distance(s.nodes[id("bc2") as usize].pos);
    assert!(gap > 0.2, "the lid did not part: {gap}");
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
}

/// The torsion law (spec §9) is the exact one: the forces τ·∂φ/∂xᵢ put τ/r⊥ on each
/// lever end along its plane's normal (r⊥ its distance from the axis line), and the
/// axis nodes' reactions leave no net force and no net torque (momentum and angular
/// momentum conserved); the field is −∇ of ½k(φ − φ₀)² (checked by central
/// differences in `dihedral_gradient_matches_the_f64_reference`).
#[test]
fn torsion_forces_are_the_exact_lever_law() {
    use engine_softbody::explicit_beams::dihedral_jacobian;
    let cases = [
        [
            Vec3::new(0.0, 0.267, 0.0),
            Vec3::ZERO,
            Vec3::new(1.7, 0.0, 0.0),
            Vec3::new(1.7, 0.267, 0.05),
        ],
        [
            Vec3::new(-0.3, 0.4, 0.2),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.2, 0.0, 1.1),
            Vec3::new(0.9, -0.2, 1.4),
        ],
    ];
    for x in cases {
        let g = dihedral_jacobian(x);
        let axis = (x[2] - x[1]).normalize();
        let perp = |p: Vec3| {
            let d = p - x[1];
            (d - axis * d.dot(axis)).length()
        };
        assert!((g[0].length() - 1.0 / perp(x[0])).abs() < 1e-4 * g[0].length());
        assert!((g[3].length() - 1.0 / perp(x[3])).abs() < 1e-4 * g[3].length());
        assert!(g[0].dot(axis).abs() < 1e-5 && g[3].dot(axis).abs() < 1e-5);
        let f: Vec3 = g.iter().copied().sum();
        let m: Vec3 = (0..4).map(|i| x[i].cross(g[i])).sum();
        assert!(f.length() < 1e-4, "net force {f}");
        assert!(m.length() < 1e-4, "net torque {m}");
    }
}
