//! Breakable joints on the native wgpu host (same WGSL as Chrome).
//!
//! A joint's converged multiplier is the constraint force it carries. Below the
//! break threshold the joint holds indefinitely; above it the joint breaks once,
//! reports `EVENT_BREAK` with the force that exceeded the threshold, and its
//! bodies separate.

use engine_schema::{AvbdParams, BodyState, Collider, Joint, JointType, ShapeType, WorldCapacity, JOINT_FLAG_BROKEN};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const G: f32 = 10.0;

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 1024,
        max_colliders: 2048,
        max_contacts: 16_384,
        max_joints: 64,
        max_candidates: 32_768,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    // Measure the steady load, not the value frozen when the pair falls asleep.
    p.sleep_time = 1e6;
    sim.set_params(p);
    sim
}

fn add_box(sim: &mut WgpuSim, pos: Vec3, half: Vec3, mass: f32) -> u32 {
    let body = if mass > 0.0 {
        let i = Vec3::new(half.y * half.y + half.z * half.z, half.x * half.x + half.z * half.z, half.x * half.x + half.y * half.y)
            * (mass / 3.0);
        sim.add_body(BodyState::dynamic(pos, Quat::IDENTITY, mass, i))
    } else {
        sim.add_body(BodyState {
            position: pos.to_array(),
            inv_mass: 0.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [0.0; 3],
            flags: 0,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
            initial_pos: pos.to_array(),
            initial_pad: 0.0,
            initial_rot: [0.0, 0.0, 0.0, 1.0],
        })
    };
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half.x, half.y, half.z, half.length()],
    });
    body
}

fn joint(kind: JointType, a: u32, b: u32, anchor_a: Vec3, anchor_b: Vec3, break_force: f32, break_torque: f32) -> Joint {
    Joint {
        body_a: a,
        body_b: b,
        joint_type: kind as u32,
        local_anchor_a: anchor_a.to_array(),
        local_anchor_b: anchor_b.to_array(),
        stiffness: f32::INFINITY,
        axis_a: [0.0, 1.0, 0.0],
        axis_b: [0.0, 1.0, 0.0],
        k: [1.0e4; 6],
        break_force,
        break_torque,
        ..Default::default()
    }
}

/// A 10 kg box hangs 1 m below a static block on a ball joint (load 100 N).
fn hanging_box(break_force: f32) -> (WgpuSim, u32, u32) {
    let mut sim = world();
    let anchor = add_box(&mut sim, Vec3::new(0.0, 6.0, 0.0), Vec3::splat(0.2), 0.0);
    let bob = add_box(&mut sim, Vec3::new(0.0, 5.0, 0.0), Vec3::splat(0.25), 10.0);
    let j = sim.add_joint(joint(JointType::Ball, anchor, bob, Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.5, 0.0), break_force, 0.0));
    (sim, bob, j)
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

#[test]
fn a_joint_below_its_break_force_holds() {
    // 150 N threshold, 100 N load: the joint must carry the weight for 5 s.
    let (mut sim, bob, j) = hanging_box(150.0);
    run(&mut sim, 300);
    let joint = sim.read_joint(j);
    let lambda = Vec3::new(joint.lambda[0], joint.lambda[1], joint.lambda[2]);
    let b = sim.read_body(bob);
    eprintln!("λ {lambda:?} |λ| {:.2} N, bob y {:.4} v {:?}", lambda.length(), b.position[1], b.linear_vel);
    assert_eq!(joint.flags & JOINT_FLAG_BROKEN, 0, "the joint holds");
    assert!(sim.poll_joint_breaks().is_empty(), "no break events");
    assert!((lambda.length() - 10.0 * G).abs() < 0.05 * 10.0 * G, "converged force is the weight: {}", lambda.length());
    assert!((b.position[1] - 5.0).abs() < 0.01, "still hanging at the anchor: y {}", b.position[1]);
}

#[test]
fn a_joint_above_its_break_force_breaks_once_and_reports_it() {
    // 60 N threshold, 100 N load.
    let (mut sim, bob, j) = hanging_box(60.0);
    let mut breaks = Vec::new();
    for _ in 0..120 {
        sim.step().expect("step");
        breaks.extend(sim.poll_joint_breaks());
    }
    let joint = sim.read_joint(j);
    let b = sim.read_body(bob);
    eprintln!("breaks {breaks:?}, bob y {:.3} vy {:.3}", b.position[1], b.linear_vel[1]);
    assert_ne!(joint.flags & JOINT_FLAG_BROKEN, 0, "the joint is broken");
    assert_eq!(breaks.len(), 1, "exactly one EVENT_BREAK");
    assert_eq!(breaks[0].joint, j);
    assert!(breaks[0].force > 60.0, "the reported force exceeds the threshold: {}", breaks[0].force);
    // Free fall from the break: well below the anchor after 2 s.
    assert!(b.position[1] < 0.0, "the box fell: y {}", b.position[1]);
}

/// A 1 m cantilever (20 kg, centre 0.5 m from the weld) welded to a wall: the weld
/// carries 200 N and a 100 N·m moment. `arm_first` makes the arm the joint's body A.
fn cantilever(break_torque: f32, arm_first: bool) -> (Joint, Vec<engine_schema::JointBreak>, f32) {
    let mut sim = world();
    let wall = add_box(&mut sim, Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.25, 1.0, 1.0), 0.0);
    let arm = add_box(&mut sim, Vec3::new(0.75, 3.0, 0.0), Vec3::new(0.5, 0.1, 0.1), 20.0);
    let (wall_anchor, arm_anchor) = (Vec3::new(0.25, 0.0, 0.0), Vec3::new(-0.5, 0.0, 0.0));
    let mut jt = if arm_first {
        joint(JointType::Fixed, arm, wall, arm_anchor, wall_anchor, 1000.0, break_torque)
    } else {
        joint(JointType::Fixed, wall, arm, wall_anchor, arm_anchor, 1000.0, break_torque)
    };
    // Weld axis along the arm: the angular rows resist the droop.
    jt.axis_a = [1.0, 0.0, 0.0];
    jt.axis_b = [1.0, 0.0, 0.0];
    let j = sim.add_joint(jt);
    let mut breaks = Vec::new();
    for _ in 0..180 {
        sim.step().expect("step");
        breaks.extend(sim.poll_joint_breaks());
    }
    let arm_y = sim.read_body(arm).position[1];
    (sim.read_joint(j), breaks, arm_y)
}

#[test]
fn a_weld_carries_its_moment_as_a_converged_torque() {
    // Hard angular rows: the multiplier converges to the 100 N·m moment and the arm
    // stays level, whichever body is A (the rows' Jacobians are exact).
    for arm_first in [false, true] {
        let (held, breaks, y) = cantilever(200.0, arm_first);
        let lin = Vec3::from_slice(&held.lambda[..3]).length();
        let ang = Vec3::from_slice(&held.lambda[3..]).length();
        eprintln!("arm first {arm_first}: |λ_lin| {lin:.1} N |λ_ang| {ang:.1} N·m, arm y {y:.4}");
        assert_eq!(held.flags & JOINT_FLAG_BROKEN, 0);
        assert!(breaks.is_empty());
        assert!((lin - 200.0).abs() < 4.0, "weld carries the weight: {lin}");
        assert!((ang - 100.0).abs() < 5.0, "weld carries the moment: {ang}");
        assert!((y - 3.0).abs() < 0.01, "arm stays level: {y}");
    }
}

#[test]
fn a_fixed_joint_breaks_on_torque_not_force() {
    // 1000 N force threshold (5x the load) never trips; a 50 N·m torque threshold
    // (half the moment) does.
    let (snapped, breaks, y) = cantilever(50.0, false);
    eprintln!("snapped: breaks {breaks:?}, arm y {y:.3}");
    assert_ne!(snapped.flags & JOINT_FLAG_BROKEN, 0);
    assert_eq!(breaks.len(), 1);
    assert!(breaks[0].torque > 50.0 && breaks[0].force < 1000.0, "broken by torque: {:?}", breaks[0]);
    assert!(y < 1.0, "the arm fell: {y}");
}
