//! Character controllers (character audit Phases 1-3), on the native wgpu host:
//! the kinematic mover in `character_step.slang` against Jolt-style scenarios.

use engine_schema::character::*;
use engine_schema::{AvbdParams, BodyFlags, BodyState, Collider, Material, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use engine_webgpu::{CharacterHandle, MotionEvent, WgpuSim};
use glam::{Quat, Vec3};

const DT: f32 = 1.0 / 60.0;

fn world() -> WgpuSim {
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
    p.dt = DT;
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 0.8, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim
}

fn body(pos: Vec3, rot: Quat, inv_mass: f32, inv_i: [f32; 3], flags: u32) -> BodyState {
    BodyState {
        position: pos.to_array(),
        inv_mass,
        orientation: rot.to_array(),
        inv_inertia: inv_i,
        flags,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos.to_array(),
        initial_pad: 0.0,
        initial_rot: rot.to_array(),
    }
}

fn add_box(sim: &mut WgpuSim, b: BodyState, half: [f32; 3]) -> u32 {
    let id = sim.add_body(b);
    sim.add_collider(Collider {
        body: id,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], Vec3::from_array(half).length()],
    });
    id
}

fn static_box(sim: &mut WgpuSim, pos: Vec3, rot: Quat, half: [f32; 3]) -> u32 {
    add_box(sim, body(pos, rot, 0.0, [0.0; 3], 0), half)
}

fn dynamic_box(sim: &mut WgpuSim, pos: Vec3, half: [f32; 3], mass: f32) -> u32 {
    let inv_i = [
        3.0 / (mass * (half[1] * half[1] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[2] * half[2])),
        3.0 / (mass * (half[0] * half[0] + half[1] * half[1])),
    ];
    add_box(sim, body(pos, Quat::IDENTITY, 1.0 / mass, inv_i, 9), half)
}

/// Flat static ground with its top at y = 0.
fn ground(sim: &mut WgpuSim) -> u32 {
    static_box(sim, Vec3::new(0.0, -0.5, 0.0), Quat::IDENTITY, [60.0, 0.5, 60.0])
}

fn desc() -> CharacterDesc {
    CharacterDesc::default()
}

/// Centre height of a character standing on y = `floor`.
fn stand_y(d: &CharacterDesc, floor: f32) -> f32 {
    floor + d.radius + d.half_height + d.skin
}

fn walk(sim: &mut WgpuSim, h: CharacterHandle, v: [f32; 3], steps: u32) -> Vec<CharacterState> {
    sim.set_character_input(h, CharacterInput { velocity: v, jump_seq: 0 });
    (0..steps)
        .map(|_| {
            sim.step().expect("step");
            sim.character_state(h)
        })
        .collect()
}

#[test]
fn walks_on_flat_ground_at_its_input_speed() {
    let mut sim = world();
    ground(&mut sim);
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 0.0) + 0.2, 0.0]);
    // Falls onto the ground and stands there.
    let settle = walk(&mut sim, h, [0.0; 3], 60);
    let s = settle.last().unwrap();
    assert_eq!(s.ground, GroundState::OnGround, "{s:?}");
    assert!((s.position[1] - stand_y(&d, 0.0)).abs() < 0.01, "standing height {}", s.position[1]);
    let run = walk(&mut sim, h, [3.0, 0.0, 0.0], 120);
    let s = run.last().unwrap();
    assert!((s.position[0] - 6.0).abs() < 0.05, "walked {} m in 2 s at 3 m/s", s.position[0]);
    assert!((s.velocity[0] - 3.0).abs() < 0.05, "{:?}", s.velocity);
    assert!(run.iter().all(|s| s.ground == GroundState::OnGround), "never left the ground");
    assert!((s.position[1] - stand_y(&d, 0.0)).abs() < 0.01, "height {}", s.position[1]);
    // The inner capsule body follows.
    let inner = sim.read_body(sim.character_body(h));
    let pos = sim.character_state(h).position;
    assert!((Vec3::from_array(inner.position) - Vec3::from_array(pos)).length() < 1e-3, "inner {:?} vs {pos:?}", inner.position);
}

#[test]
fn climbs_stairs_without_ground_flicker() {
    let mut sim = world();
    ground(&mut sim);
    // 0.3 m rise, 0.5 m run, five steps from x = 1.
    let (rise, run) = (0.3f32, 0.5f32);
    for k in 0..5 {
        let top = rise * (k + 1) as f32;
        static_box(&mut sim, Vec3::new(1.0 + run * k as f32 + 0.5 * run + 5.0, 0.5 * top, 0.0), Quat::IDENTITY,
            [0.5 * run + 5.0, 0.5 * top, 2.0]);
    }
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 0.0), 0.0]);
    walk(&mut sim, h, [0.0; 3], 20);
    let states = walk(&mut sim, h, [1.5, 0.0, 0.0], 240);
    let s = states.last().unwrap();
    println!("stairs: end {:?} state {:?}", s.position, s.ground);
    assert!(s.position[0] > 3.6, "made it up the stairs: x {}", s.position[0]);
    assert!((s.position[1] - stand_y(&d, 1.5)).abs() < 0.03, "on the top step: y {}", s.position[1]);
    let air = states.iter().filter(|s| s.ground != GroundState::OnGround).count();
    let steep = states.iter().filter(|s| s.ground == GroundState::OnSteepGround).count();
    println!("stairs: {air} steps off the ground, {steep} on steep ground");
    assert!(air <= 2, "ground state flickered {air} times on the stairs");
    assert_eq!(steep, 0, "steep-ground flicker on the stairs");
    assert!(states.iter().any(|s| s.out_flags & CO_STEPPED_UP != 0), "stepped up");
}

fn slope_world(angle_deg: f32) -> (WgpuSim, CharacterHandle, CharacterDesc) {
    let mut sim = world();
    let a = angle_deg.to_radians();
    // A ramp rotated about z; its top surface through the origin.
    let rot = Quat::from_rotation_z(a);
    let n = rot * Vec3::Y;
    static_box(&mut sim, -n * 0.5, rot, [20.0, 0.5, 4.0]);
    let d = desc();
    let c = n * (d.radius + d.half_height * n.y.abs() + d.skin + 0.02) + Vec3::Y * 0.02;
    let h = sim.add_character(&d, c.to_array());
    (sim, h, d)
}

#[test]
fn holds_on_a_30_degree_slope() {
    let (mut sim, h, _) = slope_world(30.0);
    walk(&mut sim, h, [0.0; 3], 30);
    let p0 = Vec3::from_array(sim.character_state(h).position);
    let states = walk(&mut sim, h, [0.0; 3], 120);
    let s = states.last().unwrap();
    let drift = (Vec3::from_array(s.position) - p0).length();
    println!("30°: drift {drift:.4} m, state {:?}", s.ground);
    assert_eq!(s.ground, GroundState::OnGround);
    assert!(drift < 0.01, "slid {drift} m on a walkable slope");
}

#[test]
fn slides_on_a_55_degree_slope() {
    let (mut sim, h, _) = slope_world(55.0);
    walk(&mut sim, h, [0.0; 3], 10);
    let p0 = Vec3::from_array(sim.character_state(h).position);
    let states = walk(&mut sim, h, [0.0; 3], 60);
    let s = states.last().unwrap();
    println!("55°: from {p0} to {:?}, state {:?}", s.position, s.ground);
    assert!(states.iter().any(|s| s.ground == GroundState::OnSteepGround), "steep ground reported");
    assert!(s.position[1] < p0.y - 0.5, "slid down the steep slope: {} -> {}", p0.y, s.position[1]);
}

#[test]
fn rides_a_descending_elevator_without_hovering() {
    let mut sim = world();
    ground(&mut sim);
    let kin = BodyFlags::Kinematic as u32 | BodyFlags::Awake as u32;
    let lift = add_box(&mut sim, body(Vec3::new(0.0, 4.75, 0.0), Quat::IDENTITY, 0.0, [0.0; 3], kin), [1.5, 0.25, 1.5]);
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 5.0), 0.0]);
    walk(&mut sim, h, [0.0; 3], 20);
    sim.set_kinematic_velocity(lift, [0.0, -2.0, 0.0], [0.0; 3]);
    let mut worst = 0.0f32;
    let mut grounded = 0;
    for _ in 0..60 {
        sim.step().expect("step");
        let s = sim.character_state(h);
        let top = sim.read_body(lift).position[1] + 0.25;
        worst = worst.max((s.position[1] - stand_y(&d, top)).abs());
        grounded += u32::from(s.ground == GroundState::OnGround);
    }
    println!("elevator: worst gap error {worst:.4} m, grounded {grounded}/60");
    assert!(worst < 0.03, "hovered or sank on the elevator: {worst}");
    assert!(grounded >= 58, "lost the ground on the elevator ({grounded}/60)");
}

#[test]
fn is_carried_by_a_moving_platform_and_jumps_and_lands() {
    let mut sim = world();
    ground(&mut sim);
    let kin = BodyFlags::Kinematic as u32 | BodyFlags::Awake as u32;
    let plat = add_box(&mut sim, body(Vec3::new(0.0, 0.25, 0.0), Quat::IDENTITY, 0.0, [0.0; 3], kin), [2.0, 0.25, 2.0]);
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 0.5), 0.0]);
    walk(&mut sim, h, [0.0; 3], 20);
    sim.set_kinematic_velocity(plat, [1.0, 0.0, 0.0], [0.0; 3]);
    walk(&mut sim, h, [0.0; 3], 60);
    let s = sim.character_state(h);
    let px = sim.read_body(plat).position[0];
    println!("platform at {px}, character at {:?}", s.position);
    assert!((s.position[0] - px).abs() < 0.05, "carried: character {} platform {px}", s.position[0]);
    sim.poll_motion_events();
    sim.set_character_input(h, CharacterInput { velocity: [0.0; 3], jump_seq: 1 });
    let mut peak = 0.0f32;
    for _ in 0..90 {
        sim.step().expect("step");
        peak = peak.max(sim.character_state(h).position[1]);
    }
    let rise = peak - stand_y(&d, 0.5);
    let expect = d.jump_speed * d.jump_speed / (2.0 * 10.0);
    println!("jump rise {rise:.3} (ballistic {expect:.3})");
    assert!((rise - expect).abs() < 0.1, "jump height {rise} vs {expect}");
    let events = sim.poll_motion_events();
    assert!(events.iter().any(|e| matches!(e, MotionEvent::Landed { character: 0, .. })), "{events:?}");
    assert_eq!(sim.character_state(h).ground, GroundState::OnGround);
}

#[test]
fn pushes_light_debris_is_blocked_by_heavy_and_weighs_on_a_plank() {
    let mut sim = world();
    ground(&mut sim);
    let d = desc();
    // A light crate in the way (10 kg) and a heavy one (2 t) further along.
    let light = dynamic_box(&mut sim, Vec3::new(1.5, 0.25, 0.0), [0.25; 3], 10.0);
    let heavy = dynamic_box(&mut sim, Vec3::new(1.5, 0.5, 4.0), [0.5; 3], 2000.0);
    let h = sim.add_character(&d, [0.0, stand_y(&d, 0.0), 0.0]);
    walk(&mut sim, h, [0.0; 3], 20);
    walk(&mut sim, h, [2.0, 0.0, 0.0], 90);
    let lx = sim.read_body(light).position[0];
    let cx = sim.character_state(h).position[0];
    println!("light crate pushed to {lx}, character at {cx}");
    assert!(lx > 2.5, "the character pushed the light crate: {lx}");
    assert!(cx > 2.0, "walked on through it: {cx}");
    // Walk into the heavy crate along z: blocked a skin away from its face.
    let h2 = sim.add_character(&d, [1.5, stand_y(&d, 0.0), 1.0]);
    walk(&mut sim, h2, [0.0; 3], 10);
    walk(&mut sim, h2, [0.0, 0.0, 2.0], 90);
    let cz = sim.character_state(h2).position[2];
    let hz = sim.read_body(heavy).position[2];
    println!("heavy crate at {hz}, character at {cz}");
    assert!(cz < hz - 0.5 - d.radius + 0.02, "blocked by the heavy crate: {cz} vs face {}", hz - 0.5);
    assert!((hz - 4.0).abs() < 0.05, "the heavy crate stayed: {hz}");
}

#[test]
fn standing_weighs_on_a_dynamic_plank() {
    let mut sim = world();
    ground(&mut sim);
    // A plank resting on a pivot: the character stands on one end and tips it.
    static_box(&mut sim, Vec3::new(0.0, 0.25, 0.0), Quat::IDENTITY, [0.1, 0.25, 1.0]);
    let plank = dynamic_box(&mut sim, Vec3::new(0.0, 0.55, 0.0), [2.0, 0.05, 0.5], 20.0);
    let d = desc();
    let h = sim.add_character(&d, [1.5, stand_y(&d, 0.6) + 0.05, 0.0]);
    let states = walk(&mut sim, h, [0.0; 3], 120);
    let s = states.last().unwrap();
    let pq = Quat::from_array(sim.read_body(plank).orientation);
    let tilt = (pq * Vec3::X).y;
    println!("plank end at {tilt:.3} (tilt sine), character {:?} push {:?}", s.position, s.push);
    assert!(states.iter().any(|s| s.push[1] < -100.0), "weight on the plank");
    assert!(tilt < -0.1, "the plank tipped under the character's weight: {tilt}");
}

#[test]
fn a_ccd_rocket_never_passes_through_a_character() {
    let mut sim = world();
    let mut p = sim.params();
    p.flags |= FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    ground(&mut sim);
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 0.0), 0.0]);
    walk(&mut sim, h, [0.0; 3], 10);
    // A 2 kg sphere rocket at 250 m/s, aimed through the character's chest.
    let r = 0.1;
    let i = 2.5 / (2.0 * r * r);
    let mut rb = body(Vec3::new(-12.0, 1.2, 0.0), Quat::IDENTITY, 0.5, [i; 3], 9 | BODY_CCD);
    rb.linear_vel = [250.0, 0.0, 0.0];
    let rocket = sim.add_body(rb);
    sim.add_collider(Collider {
        body: rocket,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [r, 0.0, 0.0, r],
    });
    let p0 = rb.linear_vel;
    let mut max_x = f32::MIN;
    for _ in 0..30 {
        let mut p = sim.params();
        p.gravity_y = -10.0;
        sim.set_params(p);
        sim.step().expect("step");
        max_x = max_x.max(sim.read_body(rocket).position[0]);
    }
    println!("rocket max x {max_x} (character at 0, radius {})", d.radius);
    let _ = p0;
    assert!(max_x < -(d.radius + r) + 0.05, "rocket passed into or through the character: {max_x}");
}

/// `compact` renumbers the character pool (character audit Phase 3): its inner
/// capsule and the crate it stands on move to new ids, a dropped body vanishes,
/// and the character keeps standing on the crate.
#[test]
fn compact_renumbers_characters_and_their_ground() {
    let mut sim = world();
    ground(&mut sim);
    let debris = dynamic_box(&mut sim, Vec3::new(20.0, 0.25, 20.0), [0.25; 3], 5.0);
    let crate_ = dynamic_box(&mut sim, Vec3::new(0.0, 0.5, 0.0), [0.75, 0.5, 0.75], 400.0);
    let d = desc();
    let h = sim.add_character(&d, [0.0, stand_y(&d, 1.0) + 0.02, 0.0]);
    let inner = sim.character_body(h);
    let s = walk(&mut sim, h, [0.0; 3], 60).pop().unwrap();
    assert_eq!(s.ground_body, Some(crate_), "{s:?}");
    // Drop the debris box (body 1, collider 1).
    let (bm, _) = sim.compact(&[0, crate_, inner], &[0, 2, 3]);
    assert_eq!(bm[debris as usize], u32::MAX);
    let (new_crate, new_inner) = (bm[crate_ as usize], bm[inner as usize]);
    assert_eq!(sim.character_body(h), new_inner);
    assert_eq!(sim.character_state(h).ground_body, Some(new_crate), "ground renumbered");
    let states = walk(&mut sim, h, [0.0; 3], 60);
    let s = states.last().unwrap();
    println!("after compact: {:?} on {:?}", s.position, s.ground_body);
    assert!(states.iter().all(|s| s.ground == GroundState::OnGround && s.ground_body == Some(new_crate)));
    assert!((s.position[1] - stand_y(&d, 1.0)).abs() < 0.02, "height {}", s.position[1]);
    let b = sim.read_body(new_inner);
    assert!((Vec3::from_array(b.position) - Vec3::from_array(s.position)).length() < 1e-3, "inner capsule follows");
}

/// A heavy block dropped into a standing character's head (overlapping it by
/// 0.15 m) reports `Crushed`; a block resting on the capsule does not.
#[test]
fn a_block_pressed_into_the_head_reports_crushed() {
    for overlap in [0.0f32, 0.15] {
        let mut sim = world();
        ground(&mut sim);
        let d = desc();
        let h = sim.add_character(&d, [0.0, stand_y(&d, 0.0), 0.0]);
        walk(&mut sim, h, [0.0; 3], 20);
        sim.poll_motion_events();
        let head = stand_y(&d, 0.0) + d.half_height + d.radius;
        dynamic_box(&mut sim, Vec3::new(0.0, head + 0.25 + d.skin - overlap, 0.0), [0.4, 0.25, 0.4], 1000.0);
        walk(&mut sim, h, [0.0; 3], 3);
        let crushed = sim
            .poll_motion_events()
            .iter()
            .any(|e| matches!(e, MotionEvent::Crushed { character: 0, .. }));
        println!("overlap {overlap}: crushed {crushed}");
        assert_eq!(crushed, overlap > 0.0, "overlap {overlap}");
    }
}
