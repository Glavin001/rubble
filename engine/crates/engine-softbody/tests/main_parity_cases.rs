//! Cases of `main`'s (30ab0e1d) tests whose input or reference the branch edited,
//! kept here as main had them (PR #105 regression audit), so the edit does not drop
//! what main tested.

use engine_softbody::explicit_beams::*;
use glam::Vec3;

fn exact() -> ExplicitBeamsSettings {
    let mut s = ExplicitBeamsSettings::default();
    s.fast_inv_sqrt = false;
    s.drag = false;
    s.turbulence = false;
    s
}

/// `explicit_beams.rs::self_collision_pushes_a_contacter_out_of_a_triangle` as on
/// main: the contacter falls at 1 m/s (the branch's copy uses 10 m/s).
#[test]
fn main_self_collision_pushes_a_contacter_out_of_a_triangle_at_1_m_s() {
    let mut s = ExplicitBeams::empty(exact());
    s.ground = -1.0e3;
    for p in [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.015, 0.0),
    ] {
        s.nodes.push(Node { pos: p, vel: Vec3::ZERO, force: Vec3::ZERO, mass: 1.0, friction: 1.0, flags: NODE_NO_GROUND });
    }
    for n in &mut s.nodes[..3] {
        n.flags |= NODE_IMMOVABLE | NODE_ON_TRIANGLE;
    }
    s.nodes[3].flags |= NODE_CONTACTER;
    s.nodes[3].vel = Vec3::new(0.0, -1.0, 0.0);
    s.triangles.push(Triangle { nodes: [0, 1, 2], wait: 0, idle: 0 });
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

/// The exact path's beam length against main's formula `q * (1 / sqrt(q))`: the
/// branch computes it another way (`exact_length`), which changes last bits. It must
/// stay within two ulp of main's value (measured: 92 of 174 sampled values differ, at
/// most 2 ulp) and no further from the true length.
#[test]
fn the_exact_length_is_within_two_ulp_of_mains() {
    let mut worst = 0u32;
    let mut differing = 0u32;
    let mut n = 0u32;
    let mut x = 1.0e-6f32;
    while x < 1.0e6 {
        for q in [x, x * 1.37, x * 2.9] {
            let main = q * (1.0 / q.sqrt());
            let (len, inv) = exact_length(q);
            let d = (len.to_bits() as i64 - main.to_bits() as i64).unsigned_abs() as u32;
            worst = worst.max(d);
            differing += u32::from(d != 0);
            n += 1;
            let truth = f64::from(q).sqrt();
            assert!(
                (f64::from(len) - truth).abs() <= (f64::from(main) - truth).abs() + f64::EPSILON * truth,
                "q {q}: exact_length {len} is further from sqrt than main's {main}"
            );
            assert!((f64::from(inv) * truth - 1.0).abs() < 4.0e-7, "q {q}: inverse {inv}");
        }
        x *= 1.618;
    }
    eprintln!("exact length: {differing}/{n} values differ from main's formula, at most {worst} ulp");
    assert!(worst <= 2, "exact_length is {worst} ulp from main's q * (1 / sqrt(q))");
}
