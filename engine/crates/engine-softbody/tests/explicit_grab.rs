//! The node grab (`ExplicitBeams::grab`, Rubble only): an external spring–damper
//! pulling one node toward a world point, added to the node's force after the node
//! pass's restart. Checked from its definition (the force after one step), its
//! physics (a lone node settles at the target, sagging m·g/k), that no grab or an
//! out-of-range node leaves every step unchanged, and that a strong grab tears a
//! part off a car.

use engine_softbody::explicit_beams::*;
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
};

/// One node of `mass` at `p`, no ground contact, no drag.
fn lone(p: Vec3, v: Vec3, mass: f32) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(EXACT);
    s.ground = -1.0e3;
    s.nodes.push(Node {
        pos: p,
        vel: v,
        force: Vec3::ZERO,
        mass,
        friction: 1.0,
        flags: NODE_NO_GROUND,
    });
    s.finish();
    s
}

fn sedan() -> ExplicitBeams {
    let text = car_truck_bare(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, EXACT).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    ExplicitBeams::from_truck(&text, Vec3::new(0.0, 0.01 - low, 0.0), Vec3::ZERO, EXACT).unwrap()
}

#[test]
fn grab_force_is_added_after_the_restart() {
    let target = Vec3::new(1.0, 2.0, -0.5);
    let (k, c) = (400.0, 3.0);
    let mut s = lone(Vec3::new(0.1, 0.2, 0.3), Vec3::new(1.0, -2.0, 0.5), 2.0);
    s.origin = Vec3::new(10.0, 0.0, -4.0);
    s.grab = Some(Grab {
        node: 0,
        target,
        k,
        c,
    });
    s.step();
    let n = s.nodes[0];
    let x = s.origin + n.pos;
    let want = Vec3::new(0.0, n.mass * GRAVITY, 0.0) + ((target - x) * k - n.vel * c);
    assert_eq!(
        n.force, want,
        "the restart (gravity) plus the grab, bit for bit"
    );
    // The next step consumes it.
    let v0 = n.vel;
    s.step();
    assert_eq!(s.nodes[0].vel, v0 + want / n.mass * STEP);
}

#[test]
fn a_lone_node_settles_at_the_target_less_its_sag() {
    let target = Vec3::new(0.0, 1.0, 0.0);
    let (m, k) = (1.0f32, 1000.0f32);
    // Critically damped.
    let c = 2.0 * (k * m).sqrt();
    let mut s = lone(Vec3::new(0.5, 0.0, -0.3), Vec3::ZERO, m);
    s.grab = Some(Grab {
        node: 0,
        target,
        k,
        c,
    });
    s.run(2.0);
    let rest = target + Vec3::new(0.0, m * GRAVITY / k, 0.0);
    let p = s.origin + s.nodes[0].pos;
    assert!(p.distance(rest) < 1e-4, "{p} vs {rest}");
    assert!(s.nodes[0].vel.length() < 1e-3);
}

#[test]
fn no_grab_or_a_missing_node_changes_nothing() {
    let mut a = sedan();
    let mut b = a.clone();
    let mut c = a.clone();
    b.grab = None;
    c.grab = Some(Grab {
        node: c.nodes.len() as u32 + 7,
        target: Vec3::new(0.0, 5.0, 0.0),
        k: 1.0e5,
        c: 100.0,
    });
    for _ in 0..30 {
        a.frame();
        b.frame();
        c.frame();
    }
    assert_eq!(a.positions(), b.positions());
    assert_eq!(a.positions(), c.positions());
    assert_eq!(
        a.nodes.iter().map(|n| n.vel).collect::<Vec<_>>(),
        c.nodes.iter().map(|n| n.vel).collect::<Vec<_>>()
    );
}

#[test]
fn a_strong_grab_tears_a_part_off() {
    let mut s = sedan();
    // The frontmost node (bumper), pulled 4 m ahead of where it starts and 2 m up.
    let (front, _) = s
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (i, n.pos.z))
        .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
    let start = s.origin + s.nodes[front].pos;
    let mass = s.nodes[front].mass;
    s.grab = Some(Grab {
        node: front as u32,
        target: start + Vec3::new(0.0, 2.0, 4.0),
        k: 2.0e5,
        c: 0.5 * mass / STEP,
    });
    let before = s.damage().0;
    s.run(1.0);
    let (broken, bent) = s.damage();
    let gone = (s.origin + s.nodes[front].pos).distance(start);
    eprintln!("grab: {broken} broken, {bent} bent, node moved {gone:.2} m");
    assert_eq!(before, 0);
    assert!(broken > 0, "the pull breaks beams");
    assert!(
        gone > 2.0,
        "the node is pulled toward the target ({gone} m)"
    );
    assert!(
        s.nodes
            .iter()
            .all(|n| n.pos.is_finite() && n.vel.is_finite()),
        "stays finite"
    );
}
