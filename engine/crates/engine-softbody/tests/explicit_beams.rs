//! Rules of the explicit-beams solver (`engine/docs/explicit-beams.md`), each
//! checked from its definition: the numerical approximations, yield and hardening,
//! the halfway force and breaking, the break exception, detacher groups, tension
//! weakening, supports, ropes, limited springs, the contact force against a box,
//! tyre geometry and spawn order.

use engine_softbody::explicit_beams::*;
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
};

/// Unit-mass nodes at `points`, no ground contact, no drag.
fn world(points: &[Vec3]) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    for &p in points {
        s.nodes.push(Node {
            pos: p,
            vel: Vec3::ZERO,
            force: Vec3::ZERO,
            mass: 1.0,
            friction: 1.0,
            flags: NODE_NO_GROUND,
        });
    }
    s
}

fn plain(a: u32, b: u32, rest: f32, k: f32, deform: f32, strength: f32) -> Beam {
    Beam::new(
        a,
        b,
        ExplicitKind::Plain,
        ExplicitType::Deformable,
        k,
        0.0,
        rest,
        deform,
        strength,
        0.0,
    )
}

fn node(p: Vec3, v: Vec3) -> Node {
    Node {
        pos: p,
        vel: v,
        force: Vec3::ZERO,
        mass: 1.0,
        friction: 1.0,
        flags: 0,
    }
}

// --- approximations ---

#[test]
fn fast_inverse_square_root_is_within_two_tenths_of_a_percent() {
    for i in 1..2000 {
        let x = i as f32 * 0.37;
        let exact = 1.0 / x.sqrt();
        let rel = (fast_inv_sqrt(x) - exact).abs() / exact;
        assert!(rel < 2e-3, "x {x}: {rel}");
    }
}

#[test]
fn approximate_square_root_is_exact_at_even_powers_of_two_and_close_between() {
    for (x, r) in [(1.0f32, 1.0f32), (4.0, 2.0), (16.0, 4.0), (0.25, 0.5)] {
        assert_eq!(approx_sqrt(x), r);
    }
    for i in 1..1000 {
        let x = i as f32 * 0.123;
        let rel = (approx_sqrt(x) - x.sqrt()).abs() / x.sqrt();
        assert!(rel < 0.07, "x {x}: {rel}");
    }
}

#[test]
fn approximate_exp_clamps_and_tracks_exp() {
    assert_eq!(approx_exp(-15.5), 0.0);
    assert_eq!(approx_exp(88.5), 1e38);
    let mut last = 0.0;
    for i in -140..=140 {
        let x = i as f32 * 0.1;
        let y = approx_exp(x);
        let rel = (y - x.exp()).abs() / x.exp();
        assert!(rel < 0.07, "x {x}: {y} vs {}", x.exp());
        assert!(y >= last, "monotone at {x}");
        last = y;
    }
}

#[test]
fn approximate_pow_is_exact_on_powers_of_two_and_close_between() {
    assert_eq!(approx_pow(2.0, 2.0), 4.0);
    assert_eq!(approx_pow(4.0, 0.5), 2.0);
    assert_eq!(approx_pow(1.0, 3.7), 1.0);
    for i in 1..200 {
        let x = i as f32 * 0.05;
        let rel = (approx_pow(x, 2.0) - x * x).abs() / (x * x);
        assert!(rel < 0.13, "x {x}: {rel}");
    }
}

#[test]
fn turbulence_noise_is_the_documented_sequence() {
    let mut state = 1;
    let first = noise_draw(&mut state);
    assert_eq!(state, 16807);
    assert_eq!(first, f32::from_bits(0x4000_0000 | 16807) - 3.0);
    let mut sum = 0.0f64;
    for _ in 0..30_000 {
        let v = noise_draw(&mut state);
        assert!((-1.0..1.0).contains(&v));
        sum += v as f64;
    }
    assert!((sum / 30_000.0).abs() < 0.05, "mean {}", sum / 30_000.0);
}

#[test]
fn frames_carry_the_remainder() {
    let mut c = FrameClock::default();
    let steps: Vec<u32> = (0..6).map(|_| c.next()).collect();
    assert_eq!(steps, vec![33, 33, 34, 33, 33, 34]);
    assert_eq!(
        ExplicitBeamsSettings::default(),
        ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag: true,
            turbulence: true,
        }
    );
}

// --- beams ---

#[test]
fn compression_yield_moves_rest_length_halves_the_excess_and_hardens() {
    let mut b = plain(0, 1, 1.0, 1.0e6, 1000.0, 1.0e9);
    let a = node(Vec3::new(0.99, 0.0, 0.0), Vec3::ZERO);
    let o = node(Vec3::ZERO, Vec3::ZERO);
    let out = beam_force(&mut b, &a, &o, false);
    // e = −0.01: F = 10 kN > Tc. δ = e + Tc/k.
    let delta = -0.01f32 + 1000.0 / 1.0e6;
    assert!((b.rest - (1.0 + delta)).abs() < 1e-6, "{}", b.rest);
    // Halfway between the force and the threshold.
    let f = 1.0e6 * 0.01f32;
    assert!(
        (out.force - (f - (f - 1000.0) / 2.0)).abs() < 1.0,
        "{}",
        out.force
    );
    // Hardened: Tc scales with L_old / L, and the gate follows.
    assert!((b.tc - 1000.0 / b.rest).abs() < 1e-2, "{}", b.tc);
    assert_eq!(b.gate, b.tc.min(b.tt).min(b.strength));
    assert!(!out.check_break);
    // Plastic coefficient 1: the beam yields all the way to its current length.
    let mut p = plain(0, 1, 1.0, 1.0e6, 1000.0, 1.0e9);
    p.plastic = 1.0;
    beam_force(&mut p, &a, &o, false);
    assert!((p.rest - 0.99).abs() < 1e-6);
}

#[test]
fn compression_yield_keeps_a_minimum_rest_length() {
    let mut b = plain(0, 1, 0.2, 1.0e6, 10.0, 1.0e12);
    b.plastic = 1.0;
    let a = node(Vec3::new(0.01, 0.0, 0.0), Vec3::ZERO);
    let o = node(Vec3::ZERO, Vec3::ZERO);
    beam_force(&mut b, &a, &o, false);
    assert_eq!(b.rest, MIN_REST);
}

#[test]
fn breaking_above_strength_after_the_halfway_force() {
    // F = 10 kN, halfway force 5.5 kN: strength 6 kN holds, 5 kN breaks.
    for (strength, breaks) in [(6000.0f32, false), (5000.0, true)] {
        let mut s = world(&[Vec3::new(0.99, 0.0, 0.0), Vec3::ZERO]);
        s.beams.push(plain(0, 1, 1.0, 1.0e6, 1000.0, strength));
        s.finish();
        s.step();
        assert_eq!(s.beams[0].broken, breaks, "strength {strength}");
        let fx = s.nodes[0].force.x;
        if breaks {
            assert_eq!(fx, 0.0, "a breaking beam applies no force");
        } else {
            assert!((fx - 5500.0).abs() < 1.0, "{fx}");
        }
    }
}

#[test]
fn a_beam_at_a_thinly_held_triangle_node_does_not_break() {
    // Node 0 is on a collision triangle with two plain beams: the overloaded one is
    // saved and its strength becomes twice its gate.
    let pts = [
        Vec3::new(0.99, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let mut s = world(&pts);
    s.nodes[0].flags |= NODE_ON_TRIANGLE;
    s.beams.push(plain(0, 1, 1.0, 1.0e6, 1000.0, 5000.0));
    let r = pts[0].distance(pts[2]);
    s.beams.push(plain(0, 2, r, 1.0e6, 1.0e9, 1.0e9));
    s.finish();
    s.step();
    assert!(!s.beams[0].broken);
    assert_eq!(s.beams[0].strength, 2.0 * s.beams[0].gate);
    // A third live plain beam at node 0: now it breaks.
    let mut s = world(&pts);
    s.nodes[0].flags |= NODE_ON_TRIANGLE;
    s.beams.push(plain(0, 1, 1.0, 1.0e6, 1000.0, 5000.0));
    s.beams.push(plain(0, 2, r, 1.0e6, 1.0e9, 1.0e9));
    s.beams
        .push(plain(0, 3, pts[0].distance(pts[3]), 1.0e6, 1.0e9, 1.0e9));
    s.finish();
    s.step();
    assert!(s.beams[0].broken);
}

#[test]
fn detacher_group_master_breaks_its_group_and_later_beams_skip_this_step() {
    // Pairs of nodes far apart, one beam each.
    let mut pts = Vec::new();
    for i in 0..5 {
        let x = 10.0 * i as f32;
        pts.push(Vec3::new(x + 0.99, 0.0, 0.0));
        pts.push(Vec3::new(x, 0.0, 0.0));
    }
    let mut s = world(&pts);
    let overloaded = |a, b| plain(a, b, 1.0, 1.0e6, 1000.0, 5000.0);
    let strong = |a, b| plain(a, b, 1.0, 1.0e6, 1.0e9, 1.0e9);
    // 0: follower of group 1 before the master, 1: the master, 2: follower after
    // it, 3: a beam of group 2, 4: a follower of group 2, overloaded.
    let mut b0 = strong(0, 1);
    b0.group = -1;
    let mut b1 = overloaded(2, 3);
    b1.group = 1;
    let mut b2 = strong(4, 5);
    b2.group = -1;
    let mut b3 = strong(6, 7);
    b3.group = 2;
    let mut b4 = overloaded(8, 9);
    b4.group = -2;
    s.beams.extend([b0, b1, b2, b3, b4]);
    s.finish();
    s.step();
    let broken: Vec<bool> = s.beams.iter().map(|b| b.broken).collect();
    assert_eq!(broken, vec![true, true, true, false, true]);
    let gravity = Vec3::new(0.0, GRAVITY, 0.0);
    // The earlier follower kept the force it applied; the later one was skipped.
    assert!((s.nodes[0].force.x - 10_000.0).abs() < 1.0);
    assert_eq!(s.nodes[4].force, gravity);
    // A follower breaking alone does not take its group.
    assert!(!s.beams[3].broken);
}

#[test]
fn tension_yield_hardens_and_weakens_by_its_plastic_stretch() {
    let mut b = plain(0, 1, 1.0, 1.0e6, 1000.0, 1.0e6);
    let a = node(Vec3::new(1.01, 0.0, 0.0), Vec3::ZERO);
    let o = node(Vec3::ZERO, Vec3::ZERO);
    let out = beam_force(&mut b, &a, &o, false);
    let delta = 0.01f32 - 1000.0 / 1.0e6;
    assert!((b.rest - (1.0 + delta)).abs() < 1e-6, "{}", b.rest);
    let f = -1.0e6 * 0.01f32;
    assert!(
        (out.force - (f - (f + 1000.0) / 2.0)).abs() < 1.0,
        "{}",
        out.force
    );
    assert!((b.tt - 1000.0 * b.rest).abs() < 1e-2, "{}", b.tt);
    assert!(
        (b.strength - (1.0e6 - delta * 1.0e6)).abs() < 1.0,
        "{}",
        b.strength
    );
    // Compression threshold untouched.
    assert_eq!(b.tc, 1000.0);
}

#[test]
fn supports_push_only_and_break_far_out() {
    let mk = || {
        let mut b = Beam::new(
            0,
            1,
            ExplicitKind::Support,
            ExplicitType::Deformable,
            1.0e6,
            100.0,
            1.0,
            1.0e9,
            1.0e9,
            0.0,
        );
        b.long_bound = 2.0;
        b
    };
    let o = node(Vec3::ZERO, Vec3::ZERO);
    // Stretched and separating at 1 m/s: no spring, a tenth of the damping.
    let mut b = mk();
    let out = beam_force(&mut b, &node(Vec3::new(1.5, 0.0, 0.0), Vec3::X), &o, false);
    assert!((out.force - (-10.0)).abs() < 1e-3, "{}", out.force);
    assert!(!b.broken);
    // Compressed: a full spring.
    let mut b = mk();
    let out = beam_force(
        &mut b,
        &node(Vec3::new(0.9, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert!((out.force - 1.0e5).abs() < 1.0);
    // Past L·factor of extension: broken, with its (spring-free) force still applied.
    let mut b = mk();
    let out = beam_force(&mut b, &node(Vec3::new(3.5, 0.0, 0.0), Vec3::X), &o, false);
    assert!(b.broken);
    assert!((out.force - (-10.0)).abs() < 1e-3);
    // Without a factor the default is 4 rest lengths.
    let mut b = mk();
    b.long_bound = 0.0;
    beam_force(
        &mut b,
        &node(Vec3::new(4.5, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert!(!b.broken);
    beam_force(
        &mut b,
        &node(Vec3::new(5.5, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert!(b.broken);
}

#[test]
fn ropes_pull_only() {
    let mk = || {
        Beam::new(
            0,
            1,
            ExplicitKind::Rope,
            ExplicitType::Deformable,
            1.0e6,
            100.0,
            1.0,
            1.0e9,
            1.0e9,
            0.0,
        )
    };
    let o = node(Vec3::ZERO, Vec3::ZERO);
    let mut b = mk();
    let out = beam_force(&mut b, &node(Vec3::new(0.5, 0.0, 0.0), -Vec3::X), &o, false);
    assert!((out.force - 10.0).abs() < 1e-3, "{}", out.force);
    let mut b = mk();
    let out = beam_force(
        &mut b,
        &node(Vec3::new(1.1, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert!((out.force + 1.0e5).abs() < 1.0);
}

#[test]
fn limited_springs_stiffen_past_their_bounds_and_never_yield() {
    let mk = || {
        let mut b = Beam::new(
            0,
            1,
            ExplicitKind::LimitedSpring,
            ExplicitType::Actuator,
            1000.0,
            0.0,
            1.0,
            1.0e9,
            1.0e12,
            0.0,
        );
        b.short_bound = 0.2;
        b.long_bound = 0.1;
        b.limit_k = 1.0e6;
        b.limit_c = 1000.0;
        b
    };
    let o = node(Vec3::ZERO, Vec3::ZERO);
    // Inside the bounds: its own spring.
    let mut b = mk();
    let out = beam_force(
        &mut b,
        &node(Vec3::new(1.05, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert!((out.force + 50.0).abs() < 1e-2, "{}", out.force);
    // 5 cm past the long bound: k moves 0.05 of the way to the limit spring.
    let mut b = mk();
    let out = beam_force(
        &mut b,
        &node(Vec3::new(1.15, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    let k = 1000.0 + (1.0e6 - 1000.0) * 0.05f32;
    assert!((out.force + k * 0.15).abs() < 2.0, "{}", out.force);
    // 10 cm past the short bound.
    let mut b = mk();
    let out = beam_force(
        &mut b,
        &node(Vec3::new(0.7, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    let k = 1000.0 + (1.0e6 - 1000.0) * 0.1f32;
    assert!((out.force - k * 0.3).abs() < 5.0, "{}", out.force);
    assert_eq!(b.rest, 1.0, "limited springs never yield");
    // Deformable limited springs (tyre spokes) do not yield either.
    let mut b = mk();
    b.ty = ExplicitType::Deformable;
    b.tc = 10.0;
    b.tt = 10.0;
    b.gate = 10.0;
    beam_force(
        &mut b,
        &node(Vec3::new(0.7, 0.0, 0.0), Vec3::ZERO),
        &o,
        false,
    );
    assert_eq!(b.rest, 1.0);
}

// --- contact ---

#[test]
fn a_node_driven_into_a_box_stops_without_bouncing() {
    let mut s = world(&[Vec3::new(0.0, 0.5, -0.01)]);
    s.nodes[0].flags = 0;
    s.nodes[0].vel = Vec3::new(0.0, 0.0, 5.0);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::splat(1.0),
    ));
    s.finish();
    let mut deepest = 0.0f32;
    for _ in 0..400 {
        s.step();
        let n = &s.nodes[0];
        assert!(n.vel.z > -1e-3, "bounced: {}", n.vel.z);
        deepest = deepest.max(n.pos.z);
    }
    // 80 % of the approach speed goes each step: it stops within a few mm.
    assert!(deepest < 0.01, "went {deepest} m in");
    assert!(s.nodes[0].vel.z.abs() < 1e-3);
}

#[test]
fn box_faces_are_picked_nearest_first_in_order() {
    let b = StaticBox {
        lo: Vec3::ZERO,
        hi: Vec3::ONE,
    };
    assert_eq!(box_face(&b, Vec3::new(0.5, 0.5, 0.1)), Some(Vec3::NEG_Z));
    assert_eq!(box_face(&b, Vec3::new(0.5, 0.95, 0.5)), Some(Vec3::Y));
    assert_eq!(box_face(&b, Vec3::new(0.05, 0.5, 0.5)), Some(Vec3::NEG_X));
    // Ties keep the earlier face.
    assert_eq!(box_face(&b, Vec3::splat(0.5)), Some(Vec3::NEG_Z));
    assert_eq!(box_face(&b, Vec3::new(0.5, 0.5, 1.0)), None);
}

#[test]
fn ground_contact_holds_a_node_up() {
    let mut s = world(&[Vec3::new(0.0, 0.0, 0.0)]);
    s.ground = 0.0;
    s.nodes[0].flags = 0;
    s.finish();
    for _ in 0..2000 {
        s.step();
    }
    assert!(s.nodes[0].pos.y > -0.01, "{}", s.nodes[0].pos.y);
    assert!(s.nodes[0].vel.length() < 0.05);
}

// --- spawning ---

const TYRE_TRUCK: &str = "tyre test
globals
100, 0, tracks/semi
nodes
0, 0, 0.5, 0
1, 0, 0.5, 0.3
2, 0.5, 0.5, 0.25
3, 0, 1.0, 0.15
beams
0, 1
0, 2
1, 2
0, 3
1, 3
2, 3
wheels
0.4, 0.2, 6, 0, 1, 2, 1, 0, 9999, 30, 50000, 300
end
";

#[test]
fn tyres_are_two_rings_on_their_axle_with_eight_or_nine_beams_per_ray() {
    let s = ExplicitBeams::from_truck(TYRE_TRUCK, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let n = 6usize;
    assert_eq!(s.nodes.len(), 4 + 2 * n);
    let a1 = s.nodes[0].pos;
    let a2 = s.nodes[1].pos;
    let axis = (a2 - a1).normalize();
    for i in 0..n {
        let o = &s.nodes[4 + 2 * i];
        let inn = &s.nodes[4 + 2 * i + 1];
        for (t, centre) in [(o, a1), (inn, a2)] {
            let r = t.pos - centre;
            assert!((r.length() - 0.4).abs() < 1e-5, "radius {}", r.length());
            assert!(r.dot(axis).abs() < 1e-5, "in the axle's plane");
            assert!(t.has(NODE_TYRE) && t.has(NODE_CONTACTER));
            assert!((t.mass - 30.0 / 12.0).abs() < 1e-6);
        }
    }
    // Rays are evenly spaced, outer and inner staggered by half a ray.
    let ang = |p: Vec3, c: Vec3, q: Vec3, d: Vec3| (p - c).angle_between(q - d).to_degrees();
    assert!((ang(s.nodes[4].pos, a1, s.nodes[5].pos, a2) - 30.0).abs() < 1e-3);
    assert!((ang(s.nodes[4].pos, a1, s.nodes[6].pos, a1) - 60.0).abs() < 1e-3);
    // Rigidity node 2 is nearer the second axle node: it braces the inner ring.
    // 9 beams per ray, tyre beams before the file's.
    let tyre_beams = s.beams.iter().filter(|b| b.a >= 4 || b.b >= 4).count();
    assert_eq!(tyre_beams, 9 * n);
    assert!(s.beams[..9 * n].iter().all(|b| b.a >= 4 || b.b >= 4));
    let per_ray = &s.beams[..9];
    assert_eq!(per_ray[0].kind, ExplicitKind::LimitedSpring);
    assert_eq!((per_ray[0].a, per_ray[0].b), (0, 4));
    assert_eq!((per_ray[1].a, per_ray[1].b), (1, 5));
    assert_eq!(per_ray[0].short_bound, 0.66);
    assert_eq!(per_ray[0].limit_k, 9_000_000.0);
    assert_eq!(per_ray[2].kind, ExplicitKind::Plain);
    assert_eq!(per_ray[8].ty, ExplicitType::Virtual);
    assert_eq!((per_ray[8].a, per_ray[8].b), (2, 5));
    assert_eq!(per_ray[8].gate, per_ray[8].strength);
    assert!(per_ray.iter().all(|b| b.k == 50000.0 && b.c == 300.0));
    // Without a rigidity node: 8 per ray.
    let src = TYRE_TRUCK.replace("0, 1, 2, 1, 0, 9999", "0, 1, 9999, 1, 0, 9999");
    let s = ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    assert_eq!(s.beams.len(), 8 * n + 6);
    // Other wheel sections are an error.
    let src = TYRE_TRUCK.replace("wheels\n", "meshwheels\n");
    assert!(ExplicitBeams::from_truck(&src, Vec3::ZERO, Vec3::ZERO, EXACT).is_err());
}

const ORDER_TRUCK: &str = "order test
globals
100, 0, tracks/semi
nodes
0, 0, 0.5, 0
1, 1, 0.5, 0
2, 0, 1.5, 0
3, 0, 0.5, 1, c
4, 1, 1.5, 1
5, 1, 0.5, 1
6, 0, 1.5, 1
7, 1, 1.5, 0
hydros
0, 4, 0.1
shocks
1, 2, 10000, 500, 0.2, 0.3, 0.9
beams
0, 1
0, 2, s
1, 2, r
cinecam
0.5, 1.0, 0.5, 0, 1, 2, 3, 4, 5, 6, 7, 8000, 800, 20
contacters
2
submesh
cab
0, 1, 2, c
end
";

#[test]
fn spawn_order_flags_and_rest_lengths() {
    let pos = Vec3::new(12_345.6, 7.0, -9_876.5);
    let s = ExplicitBeams::from_truck(ORDER_TRUCK, pos, Vec3::X, ExplicitBeamsSettings::default())
        .unwrap();
    // Cinecam beams, then plain/support/rope, then shocks, then hydros.
    assert_eq!(s.beams.len(), 8 + 3 + 1 + 1);
    assert!(s.beams[..8].iter().all(|b| b.a == 8));
    let kinds: Vec<_> = s.beams[8..].iter().map(|b| (b.kind, b.ty)).collect();
    assert_eq!(
        kinds,
        vec![
            (ExplicitKind::Plain, ExplicitType::Deformable),
            (ExplicitKind::Support, ExplicitType::Deformable),
            (ExplicitKind::Rope, ExplicitType::Deformable),
            (ExplicitKind::LimitedSpring, ExplicitType::Actuator),
            (ExplicitKind::Plain, ExplicitType::Actuator),
        ]
    );
    // Rest lengths from the spawned single-precision positions; the shock's
    // precompressed.
    for b in &s.beams {
        let l = s.nodes[b.a as usize]
            .pos
            .distance(s.nodes[b.b as usize].pos);
        let expect = if b.kind == ExplicitKind::LimitedSpring {
            l * 0.9
        } else {
            l
        };
        assert_eq!(b.rest, expect);
        assert_eq!(b.rest0, b.rest);
    }
    let shock = &s.beams[11];
    assert_eq!((shock.short_bound, shock.long_bound), (0.2, 0.3));
    assert_eq!(shock.limit_c, 12_000.0);
    assert_eq!(shock.gate, shock.strength);
    let support = &s.beams[9];
    assert_eq!(support.long_bound, 4.0);
    // Positions: spawn position plus offset in single precision; velocity given.
    assert_eq!(s.nodes[4].pos, pos + Vec3::new(1.0, 1.5, 1.0));
    assert_eq!(s.nodes[4].vel, Vec3::X);
    // Flags: only the listed contacter; `c` nodes and the cinecam skip the ground;
    // the cab's nodes are on a collision triangle.
    let contacters: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| s.nodes[i].has(NODE_CONTACTER))
        .collect();
    assert_eq!(contacters, vec![2]);
    assert!(s.nodes[3].has(NODE_NO_GROUND) && s.nodes[8].has(NODE_NO_GROUND));
    assert!(!s.nodes[4].has(NODE_NO_GROUND));
    assert!((0..3).all(|i| s.nodes[i].has(NODE_ON_TRIANGLE)));
    assert!(!s.nodes[3].has(NODE_ON_TRIANGLE));
    assert_eq!(s.triangles.len(), 1);
    assert_eq!(s.candidates, vec![2]);
}

#[test]
fn origin_rebases_to_node_zero_past_one_hundred_metres() {
    let pos = Vec3::new(500.0, 0.5, 0.0);
    let mut s = ExplicitBeams::from_truck(ORDER_TRUCK, pos, Vec3::ZERO, EXACT).unwrap();
    let before = s.positions();
    s.rebase_origin();
    assert_eq!(s.origin, before[0]);
    assert_eq!(s.nodes[0].pos, Vec3::ZERO);
    for (a, b) in before.iter().zip(s.positions()) {
        assert!(a.distance(b) < 1e-4);
    }
    // Near the origin nothing moves.
    let mut s = ExplicitBeams::from_truck(ORDER_TRUCK, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    s.rebase_origin();
    assert_eq!(s.origin, Vec3::ZERO);
}

#[test]
fn self_collision_pushes_a_contacter_out_of_a_triangle() {
    // A big triangle in y = 0 and a contacter just above it, falling.
    let mut s = world(&[
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.015, 0.0),
    ]);
    for n in &mut s.nodes[..3] {
        n.flags |= NODE_IMMOVABLE | NODE_ON_TRIANGLE;
    }
    s.nodes[3].flags |= NODE_CONTACTER;
    s.nodes[3].vel = Vec3::new(0.0, -1.0, 0.0);
    s.triangles.push(Triangle {
        nodes: [0, 1, 2],
        wait: 0,
        idle: 0,
    });
    s.finish();
    for _ in 0..200 {
        s.step();
        // Never through the face; it settles at the edge of the range.
        assert!(s.nodes[3].pos.y > 0.0, "through at {}", s.nodes[3].pos.y);
    }
    let y = s.nodes[3].pos.y;
    assert!(y < s.collision_range + 0.01, "contacter at {y}");
    assert!(s.triangles[0].wait < 0, "a hit resets the throttle");
}

#[test]
fn an_unhit_triangle_backs_off_its_tests() {
    let mut s = world(&[
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(5.0, 5.0, 5.0),
    ]);
    for n in &mut s.nodes {
        n.flags |= NODE_IMMOVABLE;
    }
    s.nodes[3].flags |= NODE_CONTACTER;
    s.triangles.push(Triangle {
        nodes: [0, 1, 2],
        wait: 0,
        idle: 0,
    });
    s.finish();
    let mut seen = Vec::new();
    for _ in 0..8 {
        s.step();
        seen.push((s.triangles[0].wait, s.triangles[0].idle));
    }
    // Tested, skipped once, tested, skipped twice, tested, …
    assert_eq!(
        seen,
        vec![
            (1, 0),
            (0, 1),
            (2, 0),
            (1, 1),
            (0, 2),
            (3, 0),
            (2, 1),
            (1, 2)
        ]
    );
}

/// The self-collision grid only skips candidates outside a triangle's padded box
/// and keeps candidate order: a run with it is bit for bit the run without it.
#[test]
fn collision_grid_matches_every_pair() {
    use engine_softbody::hires::{hires_car_jbeam, HiresParams};
    let (jbeam, _) = hires_car_jbeam(&HiresParams {
        node_weight: 2.0,
        spring: 1.5e6,
        damp: 150.0,
        stiffness_length: 0.15,
        ..Default::default()
    });
    let files = [("hires.jbeam", jbeam.as_str())];
    // Moving into a box, so contacts start and the throttle runs.
    let spawn = || {
        let (mut sim, _, _) = ExplicitBeams::from_jbeam(
            &files,
            None,
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.0, 0.0, 15.0),
            ExplicitBeamsSettings::default(),
        )
        .unwrap();
        sim.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, 3.2),
            Vec3::new(2.0, 1.0, 0.3),
        ));
        sim
    };
    let mut grid = spawn();
    let mut all = spawn();
    assert!(
        grid.grid_cell > 0.0,
        "the high-resolution car uses the grid"
    );
    all.grid_cell = 0.0;
    for i in 0..400 {
        if i % 33 == 0 {
            grid.rebase_origin();
            all.rebase_origin();
        }
        grid.step();
        all.step();
    }
    assert!(
        grid.damage().0 + grid.damage().1 > 0,
        "the crash deforms the car"
    );
    for (a, b) in grid.positions().iter().zip(all.positions()) {
        assert_eq!(
            a.to_array().map(f32::to_bits),
            b.to_array().map(f32::to_bits)
        );
    }
}
