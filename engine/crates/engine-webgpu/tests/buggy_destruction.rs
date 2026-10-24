//! The hybrid dune buggy (`engine_softbody::buggy`) as it ships, in rigid mode:
//! every way it can come apart, each from a crash a player makes. A rollover
//! bends the cage without crushing the cell; a T-bone by another buggy bends
//! the struck side, and the frames do not pass through each other; a hard drop
//! sheds parts without leaving the frame bent beyond the cell; a blow to the
//! engine tears it out and cuts the drive.

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::buggy::dune_buggy_soft;
use engine_softbody::{SoftCapacity, SoftHandle, SoftSettings, SpawnOptions};
use engine_vehicle::{presets, VehicleCapacity, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const HZ: f32 = 240.0;

fn world() -> WgpuSim {
    let cap = WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 1024,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    };
    let mut sim = WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    sim.set_params(engine_softbody::soft_scene_params(HZ, 10));
    sim.set_soft_settings(SoftSettings {
        rigid_mode: true,
        ..SoftSettings::calibrated_for(HZ)
    });
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim.set_material(
        9,
        Material {
            friction: 0.8,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    static_box(
        &mut sim,
        [0.0, -5.0, 0.0],
        [400.0, 5.0, 400.0],
        Quat::IDENTITY,
    );
    sim
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3], rot: Quat) {
    let rot = rot.to_array();
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: rot,
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: rot,
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], Vec3::from_array(half).length()],
    });
}

fn ball(sim: &mut WgpuSim, pos: Vec3, radius: f32, mass: f32, vel: [f32; 3]) {
    let mut b = BodyState::dynamic(
        pos,
        Quat::IDENTITY,
        mass,
        Vec3::splat(0.4 * mass * radius * radius),
    );
    b.linear_vel = vel;
    let body = sim.add_body(b);
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [radius, 0.0, 0.0, radius],
    });
}

fn spawn(sim: &mut WgpuSim, pos: [f32; 3], rot: Quat, vel: [f32; 3]) -> SoftHandle {
    sim.add_soft_vehicle(
        &dune_buggy_soft(),
        pos,
        rot.to_array(),
        &SpawnOptions {
            velocity: vel,
            ..Default::default()
        },
    )
    .expect("spawn")
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h)
        .iter()
        .map(|b| Vec3::from_array(b.position))
        .collect()
}

/// What came apart: beams yielded / broken, torn-off frame pieces (break groups),
/// loose bolted parts, core parts and wheels broken off, and the largest change
/// in distance between two safety-cell nodes (hoop and roof: frame nodes above
/// 1.3 m in the desc).
#[derive(Debug, Default)]
struct Damage {
    yielded: usize,
    broken: usize,
    groups: Vec<u32>,
    loose: Vec<String>,
    core_parts: Vec<String>,
    wheels: u32,
    cell: f32,
    engine_connected: bool,
}

fn damage(sim: &WgpuSim, h: SoftHandle, rest: &[Vec3]) -> Damage {
    let e = sim.soft_host().entry(h).clone();
    let beams = sim.read_soft_beams(h);
    let mut groups: Vec<u32> = beams
        .iter()
        .filter(|b| b.broken() && b.group != 0)
        .map(|b| b.group)
        .collect();
    groups.sort();
    groups.dedup();
    let core = e.core.unwrap().0;
    let st = sim.vehicle_state(core);
    let cdesc = &e.desc.core.as_ref().unwrap().vehicle;
    let p = positions(sim, h);
    let cell: Vec<usize> = (0..p.len())
        .filter(|&i| {
            e.desc.nodes[i].name.starts_with("frame:") && e.desc.nodes[i].position[1] > 1.3
        })
        .collect();
    let cell_change = cell
        .iter()
        .flat_map(|&i| cell.iter().map(move |&j| (i, j)))
        .map(|(i, j)| ((p[i] - p[j]).length() - (rest[i] - rest[j]).length()).abs())
        .fold(0.0f32, f32::max);
    Damage {
        yielded: beams
            .iter()
            .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
            .count(),
        broken: beams.iter().filter(|b| b.broken()).count(),
        groups,
        loose: e
            .loose_parts(&sim.download_joints())
            .into_iter()
            .map(|k| e.desc.rigid_parts[k].name.clone())
            .collect(),
        core_parts: (0..cdesc.parts.len())
            .filter(|k| st.parts_broken & (1 << k) != 0)
            .map(|k| cdesc.parts[k].name.clone())
            .collect(),
        wheels: st.wheels_broken,
        cell: cell_change,
        engine_connected: st.engine_connected,
    }
}

/// No particle flew off: every node within `r` m of the frame's median.
fn holds_together(p: &[Vec3], r: f32) -> bool {
    let mut xs: Vec<Vec3> = p.to_vec();
    let mid = |k: fn(&Vec3) -> f32, xs: &mut Vec<Vec3>| {
        xs.sort_by(|a, b| k(a).total_cmp(&k(b)));
        k(&xs[xs.len() / 2])
    };
    let c = Vec3::new(
        mid(|v| v.x, &mut xs),
        mid(|v| v.y, &mut xs),
        mid(|v| v.z, &mut xs),
    );
    p.iter().all(|x| (*x - c).length() < r)
}

#[test]
fn a_rollover_bends_the_cage_and_keeps_the_cell() {
    let mut sim = world();
    // A kicker under the left wheels only (+X is left): the buggy rolls right.
    static_box(
        &mut sim,
        [0.75, 0.3, 14.0],
        [0.5, 0.3, 2.0],
        Quat::from_axis_angle(Vec3::X, -0.3),
    );
    let h = spawn(&mut sim, [0.0, 0.05, 0.0], Quat::IDENTITY, [0.0, 0.0, 16.0]);
    let rest = positions(&sim, h);
    let core = sim.soft_host().entry(h).core.unwrap();
    sim.set_vehicle_input(
        core.0,
        VehicleInput {
            forward: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 200);
    let rigid_before = sim.read_soft_rigid(h);
    let mut min_up = 1.0f32;
    for _ in 0..(HZ * 3.0) as u32 {
        sim.step().expect("step");
        let q = Quat::from_array(sim.read_body(core.1).orientation);
        min_up = min_up.min((q * Vec3::Y).y);
    }
    let d = damage(&sim, h, &rest);
    eprintln!("rollover: rigid before {rigid_before}, lowest up·y {min_up:.2}; {d:?}");
    assert!(rigid_before, "rigid on the run-up");
    assert!(min_up < 0.0, "it rolled past its side");
    assert!(d.yielded > 0, "the cage bent");
    assert!(
        d.cell < 0.15,
        "the cell kept its shape within 15 cm ({} m)",
        d.cell
    );
    assert!(holds_together(&positions(&sim, h), 4.0));
}

#[test]
fn a_t_bone_by_another_buggy_bends_the_struck_side() {
    let mut sim = world();
    // The target parked across the path; the striker 16 m to its left (+X),
    // facing −X, at 50 km/h.
    let target = spawn(&mut sim, [0.0, 0.05, 0.0], Quat::IDENTITY, [0.0; 3]);
    let striker = spawn(
        &mut sim,
        [16.0, 0.05, -0.3],
        Quat::from_axis_angle(Vec3::Y, -std::f32::consts::FRAC_PI_2),
        [-14.0, 0.0, 0.0],
    );
    let (rt, rs) = (positions(&sim, target), positions(&sim, striker));
    run(&mut sim, 120);
    let both_rigid = sim.read_soft_rigid(target) && sim.read_soft_rigid(striker);
    run(&mut sim, (HZ * 2.0) as u32);
    let (dt, ds) = (damage(&sim, target, &rt), damage(&sim, striker, &rs));
    // No pass-through: the striker's frame nodes stay on the struck (+X) side of
    // the target's right-hand (−X) rail.
    let pt = positions(&sim, target);
    let ps = positions(&sim, striker);
    let target_right = pt.iter().map(|p| p.x).fold(f32::MAX, f32::min);
    let through = ps.iter().filter(|p| p.x < target_right - 0.1).count();
    eprintln!("t-bone: both rigid before {both_rigid}; target {dt:?}\n  striker {ds:?}\n  striker nodes past the target {through}");
    assert!(both_rigid, "both rigid before the hit");
    assert!(dt.yielded > 0 && ds.yielded > 0, "both frames bent");
    // The tyres stand proud of the frame: the struck one tears off its arms.
    assert!(
        dt.wheels != 0 || !dt.loose.is_empty() || !dt.groups.is_empty(),
        "something on the target came off"
    );
    assert_eq!(through, 0, "the frames did not pass through each other");
    assert!(holds_together(&pt, 4.0) && holds_together(&ps, 4.0));
}

#[test]
fn a_hard_drop_lands_on_the_suspension_without_damage() {
    let mut sim = world();
    // 4 m up, level: 8.9 m/s at the ground (a botched jump).
    let h = spawn(&mut sim, [0.0, 4.05, 0.0], Quat::IDENTITY, [0.0; 3]);
    let rest = positions(&sim, h);
    run(&mut sim, (HZ * 2.5) as u32);
    let d = damage(&sim, h, &rest);
    eprintln!("4 m drop: {d:?}");
    assert_eq!(
        (d.yielded, d.wheels, d.loose.len()),
        (0, 0, 0),
        "a 4 m drop lands on the springs and bump stops"
    );
    assert!(holds_together(&positions(&sim, h), 4.0));
}

#[test]
fn a_blow_to_the_engine_tears_it_out_and_cuts_the_drive() {
    let mut sim = world();
    let h = spawn(&mut sim, [0.0, 0.05, 0.0], Quat::IDENTITY, [0.0; 3]);
    run(&mut sim, 480);
    let e = sim.soft_host().entry(h).clone();
    let (core, chassis) = e.core.unwrap();
    assert!(sim.read_soft_rigid(h), "parked: rigid");
    // The engine sits at the back of the cage: a 400 kg ball at 30 m/s from behind.
    let cdesc = &e.desc.core.as_ref().unwrap().vehicle;
    let k = cdesc
        .parts
        .iter()
        .position(|p| p.name == "engine")
        .expect("engine part");
    let at = Vec3::from_array(cdesc.parts[k].colliders[0].position)
        - engine_softbody::pack::authored_core_origin(&e.desc, cdesc);
    let b = sim.read_body(chassis);
    let world_at = Vec3::from_array(b.position) + Quat::from_array(b.orientation) * at;
    let rest = positions(&sim, h);
    ball(
        &mut sim,
        world_at - Vec3::Z * 2.5,
        0.6,
        400.0,
        [0.0, 0.0, 30.0],
    );
    run(&mut sim, 240);
    let d = damage(&sim, h, &rest);
    sim.set_vehicle_input(
        core,
        VehicleInput {
            forward: 1.0,
            ..Default::default()
        },
    );
    run(&mut sim, 480);
    let speed = sim.vehicle_state(core).forward_speed;
    eprintln!("engine hit: {d:?}; speed under full throttle after {speed:.2} m/s");
    assert!(
        d.core_parts.iter().any(|n| n == "engine"),
        "the engine tore out"
    );
    assert!(!sim.vehicle_state(core).engine_connected, "drive cut");
}

/// The rigid preset's part names, for reading the probes above.
#[test]
fn core_part_names() {
    let d = engine_softbody::buggy::dune_buggy_soft();
    let names: Vec<&str> = d
        .core
        .as_ref()
        .unwrap()
        .vehicle
        .parts
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    eprintln!(
        "core parts {names:?}; rigid preset parts {:?}",
        presets::dune_buggy()
            .parts
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(names.contains(&"engine"));
}
