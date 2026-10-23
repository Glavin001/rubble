//! Focused CCD checks on the shared engine, each in the native host's contact
//! modes and the browser's (canonical contacts, world-host.mjs's step modes):
//! the time of impact, the momentum exchange, sleeping targets, resting and
//! rolling CCD bodies, and the loud capacity failure.

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY, OVERFLOW_CCD_BODIES};
use engine_webgpu::WgpuSim;

#[path = "support/phase.rs"]
mod phase;

const DT: f32 = 1.0 / 60.0;

fn body(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: inv_i, flags, linear_vel: v, sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn collider(body: u32, shape: ShapeType, p: [f32; 4]) -> Collider {
    Collider { body, shape: shape as u32, material: 0, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: p }
}
fn world(browser: bool, gravity: f32) -> WgpuSim {
    let cap = WorldCapacity { max_bodies: 2048, max_colliders: 2048, max_contacts: 1 << 17, max_candidates: 1 << 17, max_colors: 32, ..WorldCapacity::small() };
    let mut sim = if browser { WgpuSim::new_with_compact_contacts(cap, false).unwrap() } else { WgpuSim::new_with_persistent_contacts(cap).unwrap() };
    if browser {
        sim.use_control_modes();
    }
    let mut p = AvbdParams::default();
    p.dt = DT;
    p.gravity_y = gravity;
    sim.set_params(p);
    sim
}
fn add_box(sim: &mut WgpuSim, pos: [f32; 3], h: [f32; 3], mass: f32, flags: u32, v: [f32; 3]) -> u32 {
    let ii = if mass > 0.0 { [3.0 / (mass * (h[1] * h[1] + h[2] * h[2])), 3.0 / (mass * (h[0] * h[0] + h[2] * h[2])), 3.0 / (mass * (h[0] * h[0] + h[1] * h[1]))] } else { [0.0; 3] };
    let id = sim.add_body(body(pos, if mass > 0.0 { 1.0 / mass } else { 0.0 }, ii, flags, v));
    sim.add_collider(collider(id, ShapeType::Box, [h[0], h[1], h[2], (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt()]));
    id
}
fn add_ball(sim: &mut WgpuSim, pos: [f32; 3], r: f32, mass: f32, flags: u32, v: [f32; 3]) -> u32 {
    let i = 2.5 / (mass * r * r);
    let id = sim.add_body(body(pos, 1.0 / mass, [i; 3], flags, v));
    sim.add_collider(collider(id, ShapeType::Sphere, [r, 0.0, 0.0, r]));
    id
}
fn speed(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// A CCD ball at a static 5 cm wall from 0.5, 1.5 and 2.5 steps' travel away,
/// 30-2,000 m/s: untouched before its impact step, then stopped at the face
/// (centre a radius in front, never past it).
#[test]
fn the_time_of_impact_is_exact() {
    for browser in [false, true] {
        for v in [30.0f32, 250.0, 2000.0] {
            for gap_steps in [0.5f32, 1.5, 2.5] {
                let mut sim = world(browser, 0.0);
                add_box(&mut sim, [0.0, 0.0, 0.025], [2.0, 2.0, 0.025], 0.0, 0, [0.0; 3]);
                let r = 0.1;
                let z0 = -(r + gap_steps * v * DT);
                let ball = add_ball(&mut sim, [0.0, 0.0, z0], r, 5.0, 9 | BODY_CCD, [0.0, 0.0, v]);
                let impact = gap_steps.floor() as usize;
                for step in 0..6 {
                    sim.step().unwrap();
                    let b = sim.read_body(ball);
                    if step < impact {
                        assert!((b.linear_vel[2] - v).abs() < 1e-3 * v, "browser {browser} v {v} gap {gap_steps}: slowed before its impact step {step}: {:?}", b.linear_vel);
                    }
                    assert!(b.position[2] <= -r + 0.02, "browser {browser} v {v} gap {gap_steps}: past the face at step {step}: z {}", b.position[2]);
                    if step >= impact {
                        assert!(b.position[2] >= -r - 0.05, "browser {browser} v {v} gap {gap_steps}: stopped short at step {step}: z {}", b.position[2]);
                    }
                }
            }
        }
    }
}

/// Whether `p` is inside the box `h` of body `b` (its local frame).
fn inside_box(p: [f32; 3], b: &BodyState, h: f32) -> bool {
    let [x, y, z, w] = b.orientation;
    let d = [p[0] - b.position[0], p[1] - b.position[1], p[2] - b.position[2]];
    // Rotate d by the conjugate quaternion.
    let (qx, qy, qz) = (-x, -y, -z);
    let t = [2.0 * (qy * d[2] - qz * d[1]), 2.0 * (qz * d[0] - qx * d[2]), 2.0 * (qx * d[1] - qy * d[0])];
    let l = [d[0] + w * t[0] + (qy * t[2] - qz * t[1]), d[1] + w * t[1] + (qz * t[0] - qx * t[2]), d[2] + w * t[2] + (qx * t[1] - qy * t[0])];
    l.iter().all(|c| c.abs() < h)
}

/// A 5 kg CCD ball into a free 2 kg box in empty space, the box asleep (held
/// still in the impact step, the response resolves the impact) or awake (the
/// solve does), 30-600 m/s:
/// - head on: total momentum kept to 1% and the box leaves at the perfectly
///   inelastic speed;
/// - off centre (0.05 m inside the box's edge): momentum kept to 1% up to the
///   speeds the solve converges at (awake 120 m/s, asleep 200 m/s). Faster,
///   the struck box spins at up to a turn a step and the ten-iteration solve
///   loses some of the re-contact's momentum (a plain ball does the same):
///   momentum then never grows;
/// - always: kinetic energy never grows and the ball's centre never ends a
///   step inside the box.
#[test]
fn the_impact_conserves_momentum_and_never_adds_energy() {
    for browser in [false, true] {
        for asleep in [true, false] {
            for v in [30.0f32, 120.0, 200.0, 300.0, 600.0] {
                for offset in [0.0f32, 0.35] {
                    let label = format!("browser {browser} asleep {asleep} {v} m/s {}", if offset == 0.0 { "head on" } else { "off centre" });
                    let mut sim = world(browser, 0.0);
                    let (mb, mo, h) = (5.0f32, 2.0f32, 0.4f32);
                    let target = add_box(&mut sim, [offset, 0.0, 0.0], [h; 3], mo, if asleep { 1 | 4 } else { 1 | 8 }, [0.0; 3]);
                    let ball = add_ball(&mut sim, [0.0, 0.0, -3.0], 0.1, mb, 9 | BODY_CCD, [0.0, 0.0, v]);
                    let ke0 = 0.5 * mb * v * v;
                    let p0 = mb * v;
                    let ib = mo * (2.0 * h) * (2.0 * h) / 6.0;
                    let (mut ke_max, mut p_max, mut p_end, mut box_v) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for step in 0..20 {
                        sim.step().unwrap();
                        let (a, b) = (sim.read_body(ball), sim.read_body(target));
                        let ke = 0.5 * mb * speed(a.linear_vel).powi(2) + 0.5 * mo * speed(b.linear_vel).powi(2) + 0.5 * ib * speed(b.angular_vel).powi(2);
                        ke_max = ke_max.max(ke);
                        p_end = mb * a.linear_vel[2] + mo * b.linear_vel[2];
                        p_max = p_max.max(p_end);
                        box_v = b.linear_vel[2];
                        assert!(!inside_box(a.position, &b, h), "{label}: the ball's centre is inside the box at step {step}: ball {:?} box {:?}", a.position, b.position);
                    }
                    println!("{label}: momentum {p_end:.1} of {p0:.1}, peak KE {:.3} of start, box {box_v:.1} m/s (inelastic {:.1})", ke_max / ke0, p0 / (mb + mo));
                    assert!(ke_max <= ke0 * 1.001, "{label}: energy grew to {} of the start", ke_max / ke0);
                    assert!(p_max <= p0 * 1.01, "{label}: momentum grew to {p_max} of {p0}");
                    let converged = offset == 0.0 || v <= if asleep { 200.0 } else { 120.0 };
                    if converged {
                        assert!((p_end - p0).abs() < 0.01 * p0, "{label}: momentum {p_end} vs {p0}");
                    }
                    if offset == 0.0 {
                        let vc = p0 / (mb + mo);
                        assert!(box_v >= 0.9 * vc, "{label}: the box should leave at about the inelastic speed {vc} ({box_v})");
                    }
                }
            }
        }
    }
}

/// A 600 m/s CCD ball at a sleeping slab (its wake deferred, so held still in
/// the impact step): the ball never passes it, and the slab moves after.
#[test]
fn a_sleeping_target_is_never_passed_and_takes_the_hit() {
    for browser in [false, true] {
        let mut sim = world(browser, 0.0);
        let slab = add_box(&mut sim, [0.0, 0.0, 0.05], [1.0, 1.0, 0.05], 20.0, 1 | 4, [0.0; 3]);
        let ball = add_ball(&mut sim, [0.0, 0.0, -4.0], 0.1, 5.0, 9 | BODY_CCD, [0.0, 0.0, 600.0]);
        let cols: Vec<Collider> = (0..sim.collider_count()).map(|i| sim.collider(i)).collect();
        let mut prev = sim.download_bodies();
        for step in 0..15 {
            sim.step().unwrap();
            let now = sim.download_bodies();
            assert!(phase::crossing(&cols, &prev, &now, ball as usize).is_none(), "browser {browser} step {step}: passed the sleeping slab");
            prev = now;
        }
        let s = sim.read_body(slab);
        assert!(s.position[2] > 0.5, "browser {browser}: the slab should be knocked away (z {})", s.position[2]);
    }
}

/// A CCD ball resting on the floor stays put and sleeps; one rolling at
/// 10 m/s keeps rolling (the post-solve check never holds back contact sink).
#[test]
fn resting_and_rolling_ccd_balls_are_left_alone() {
    for browser in [false, true] {
        let mut sim = world(browser, -10.0);
        let mut p = sim.params();
        p.flags |= FLAG_SPEC_VELOCITY;
        sim.set_params(p);
        add_box(&mut sim, [0.0, -1.0, 0.0], [50.0, 1.0, 50.0], 0.0, 0, [0.0; 3]);
        let rest = add_ball(&mut sim, [0.0, 0.25, 0.0], 0.25, 5.0, 9 | BODY_CCD, [0.0; 3]);
        let roll = add_ball(&mut sim, [5.0, 0.25, -20.0], 0.25, 5.0, 9 | BODY_CCD, [0.0, 0.0, 10.0]);
        // The same roll without CCD: the reference (sliding friction turns it
        // into rolling at 5/7 of the speed).
        let plain = add_ball(&mut sim, [-5.0, 0.25, -20.0], 0.25, 5.0, 9, [0.0, 0.0, 10.0]);
        let mut top = 0.0f32;
        for _ in 0..180 {
            sim.step().unwrap();
            top = top.max(sim.read_body(rest).position[1]);
        }
        let (a, b, c) = (sim.read_body(rest), sim.read_body(roll), sim.read_body(plain));
        println!("browser {browser}: resting y {:.4} (peak {top:.4}), CCD ball rolled to z {:.2} at {:.2} m/s, plain ball to z {:.2} at {:.2} m/s", a.position[1], b.position[2], speed(b.linear_vel), c.position[2], speed(c.linear_vel));
        assert!(top < 0.27 && a.position[1] > 0.22, "browser {browser}: the resting ball moved (peak y {top}, y {})", a.position[1]);
        assert!((b.position[2] - c.position[2]).abs() < 0.5, "browser {browser}: the CCD ball rolls as the plain one does (z {} vs {})", b.position[2], c.position[2]);
    }
}

/// More awake CCD bodies than CCD_BODY_CAP fail the step loudly.
#[test]
fn too_many_ccd_bodies_fail_loudly() {
    let cap = WorldCapacity { max_bodies: 1200, max_colliders: 1200, max_contacts: 1 << 16, max_candidates: 1 << 16, max_colors: 32, ..WorldCapacity::small() };
    let mut sim = WgpuSim::new(cap).unwrap();
    let mut p = AvbdParams::default();
    p.gravity_y = 0.0;
    sim.set_params(p);
    for k in 0..(engine_schema::CCD_BODY_CAP + 8) {
        add_ball(&mut sim, [(k % 40) as f32, (k / 40) as f32, 0.0], 0.1, 1.0, 9 | BODY_CCD, [0.0, 0.0, 1.0]);
    }
    match sim.step() {
        Err(engine_exec::ExecError::Overflow(f)) => assert!(f & OVERFLOW_CCD_BODIES != 0, "overflow flags {f:#x}"),
        other => panic!("expected an overflow, got {other:?}"),
    }
}
