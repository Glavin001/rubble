//! Kinematic bodies and GPU impulses (character audit Phase 0, CCD audit row 4).
//!
//! A kinematic body (`BODY_KINEMATIC`, no `BODY_DYNAMIC`) is moved by its given
//! velocity each step on the GPU. Contact rows see that motion, so friction
//! carries what rides on it, and a moving kinematic wakes the sleepers it
//! touches. Its velocity, and impulses on dynamic bodies, are records the
//! step applies on the GPU (`apply_impulses`): no body upload, so the host
//! keeps manifold reuse.

use engine_schema::{AvbdParams, BodyFlags, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const KINEMATIC: u32 = BodyFlags::Kinematic as u32 | BodyFlags::Awake as u32;

fn world(dt: f32) -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 64,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.dt = dt;
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim
}

fn body(pos: Vec3, inv_mass: f32, inv_i: [f32; 3], flags: u32) -> BodyState {
    BodyState {
        position: pos.to_array(),
        inv_mass,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: inv_i,
        flags,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos.to_array(),
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    }
}

fn add_box(sim: &mut WgpuSim, b: BodyState, half: [f32; 3]) -> u32 {
    let id = sim.add_body(b);
    let r = Vec3::from_array(half).length();
    sim.add_collider(Collider {
        body: id,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], r],
    });
    id
}

fn dynamic_box(sim: &mut WgpuSim, pos: Vec3, half: [f32; 3], mass: f32) -> u32 {
    let inv_i = [
        3.0 / (mass * (half[1] * half[1] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[1] * half[1])),
    ];
    add_box(sim, body(pos, 1.0 / mass, inv_i, 9), half)
}

/// Cargo drift (m, in the platform frame) after driving the platform for 3 s
/// at 1 m/s while turning at 0.5 rad/s. `teleport`: the old way, a static body
/// written every step (`write_bodies`); otherwise a kinematic body moved on the
/// GPU by its velocity.
fn platform_ride(teleport: bool) -> (f32, Vec3) {
    let mut sim = world(1.0 / 60.0);
    add_box(&mut sim, body(Vec3::new(0.0, -0.5, 0.0), 0.0, [0.0; 3], 0), [50.0, 0.5, 50.0]);
    let flags = if teleport { 0 } else { KINEMATIC };
    let platform = add_box(&mut sim, body(Vec3::new(0.0, 1.0, 0.0), 0.0, [0.0; 3], flags), [2.0, 0.1, 2.0]);
    let cargo = dynamic_box(&mut sim, Vec3::new(1.0, 1.3, 0.0), [0.2, 0.2, 0.2], 20.0);
    for _ in 0..30 {
        sim.step().expect("step");
    }
    let local0 = Vec3::from_array(sim.read_body(cargo).position) - Vec3::new(0.0, 1.0, 0.0);
    let (v, w) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0));
    let steps = 180;
    if !teleport {
        sim.set_kinematic_velocity(platform, v.to_array(), w.to_array());
    }
    let mut pose = sim.read_body(platform);
    for _ in 0..steps {
        if teleport {
            let p = Vec3::from_array(pose.position) + v / 60.0;
            let q = Quat::from_array(pose.orientation);
            let q = (q + Quat::from_xyzw(w.x / 120.0, w.y / 120.0, w.z / 120.0, 0.0) * q).normalize();
            pose.position = p.to_array();
            pose.orientation = q.to_array();
            pose.initial_pos = pose.position;
            pose.initial_rot = pose.orientation;
            sim.write_bodies(platform, &[pose]);
        }
        sim.step().expect("step");
    }
    let p = sim.read_body(platform);
    let c = sim.read_body(cargo);
    let t = steps as f32 / 60.0;
    let pp = Vec3::from_array(p.position);
    assert!((pp - Vec3::new(t, 1.0, 0.0)).length() < 1e-3, "platform at {pp}");
    let q = Quat::from_array(p.orientation);
    let local = q.inverse() * (Vec3::from_array(c.position) - pp);
    let point_vel = v + w.cross(Vec3::from_array(c.position) - pp);
    (( local - local0).length(), Vec3::from_array(c.linear_vel) - point_vel)
}

#[test]
fn a_box_rides_a_turning_kinematic_platform() {
    let (drift, vel_err) = platform_ride(false);
    let (teleported, _) = platform_ride(true);
    println!("cargo drift: kinematic {drift:.3} m (velocity error {vel_err}), teleported static {teleported:.3} m");
    assert!(drift < 0.1, "cargo drifted {drift} m on the kinematic platform");
    assert!(vel_err.length() < 0.1, "cargo does not move with the platform: {vel_err}");
    assert!(teleported > 1.0, "control: a teleported static platform should leave the cargo behind ({teleported} m)");
}

#[test]
fn a_moving_kinematic_wakes_a_sleeping_pile() {
    let mut sim = world(1.0 / 60.0);
    add_box(&mut sim, body(Vec3::new(0.0, -0.5, 0.0), 0.0, [0.0; 3], 0), [50.0, 0.5, 50.0]);
    let mut pile = Vec::new();
    for k in 0..3 {
        pile.push(dynamic_box(&mut sim, Vec3::new(0.0, 0.25 + 0.5 * k as f32, 0.0), [0.25; 3], 10.0));
    }
    let pusher = add_box(&mut sim, body(Vec3::new(-3.0, 0.5, 0.0), 0.0, [0.0; 3], KINEMATIC), [0.25, 0.5, 1.0]);
    for _ in 0..240 {
        sim.step().expect("step");
    }
    for &b in &pile {
        assert!(sim.read_body(b).is_sleeping(), "pile settled asleep");
    }
    sim.set_kinematic_velocity(pusher, [2.0, 0.0, 0.0], [0.0; 3]);
    let mut woke = false;
    for _ in 0..120 {
        sim.step().expect("step");
        woke |= pile.iter().any(|&b| !sim.read_body(b).is_sleeping());
    }
    assert!(woke, "the kinematic pusher woke the pile");
    let bottom = sim.read_body(pile[0]);
    println!("bottom box at {:?}", bottom.position);
    assert!(bottom.position[0] > 0.5, "the pusher moved the pile: {:?}", bottom.position);
    // The pusher never passed through the bottom box.
    let px = sim.read_body(pusher).position[0];
    assert!(px + 0.25 <= bottom.position[0] - 0.25 + 0.05, "pusher {px} inside box at {}", bottom.position[0]);
}

#[test]
fn gpu_impulses_change_velocity_and_wake() {
    let mut sim = world(1.0 / 60.0);
    let mut p = sim.params();
    p.gravity_y = 0.0;
    sim.set_params(p);
    let a = dynamic_box(&mut sim, Vec3::new(0.0, 5.0, 0.0), [0.5; 3], 2.0);
    let b = dynamic_box(&mut sim, Vec3::new(5.0, 5.0, 0.0), [0.5; 3], 2.0);
    sim.step().expect("step");
    // Two records on `a` sum in order; one off-centre record on `b` spins it.
    sim.apply_impulse(a, [4.0, 0.0, 0.0], [0.0, 5.0, 0.0]);
    sim.apply_impulse(a, [0.0, 0.0, 2.0], [0.0, 5.0, 0.0]);
    sim.apply_impulse(b, [0.0, 0.0, 1.0], [5.5, 5.0, 0.0]);
    sim.step().expect("step");
    let va = Vec3::from_array(sim.read_body(a).linear_vel);
    assert!((va - Vec3::new(2.0, 0.0, 1.0)).length() < 1e-3, "Δv = J/m: {va}");
    let wb = Vec3::from_array(sim.read_body(b).angular_vel);
    // r × J = (0.5,0,0) × (0,0,1) = (0,-0.5,0); I = m·(2·0.5²)/3·... box: I = m(h²+h²)/3.
    let i = 2.0 * (0.25 + 0.25) / 3.0;
    assert!((wb - Vec3::new(0.0, -0.5 / i, 0.0)).length() < 1e-3, "Δω = I⁻¹ r×J: {wb}");
    let vb = Vec3::from_array(sim.read_body(b).linear_vel);
    assert!((vb - Vec3::new(0.0, 0.0, 0.5)).length() < 1e-3, "{vb}");
}
