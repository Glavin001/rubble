//! The hybrid dune buggy (`engine_softbody::buggy`) on the native wgpu host:
//! a node–beam tube frame on the rigid buggy's raycast core, with its bolted
//! parts on breakable ball joints. It must stand and drive without bending,
//! bend in front and keep its safety cell in a wall crash, and shed a bolted
//! part that is struck.

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity, JOINT_FLAG_BROKEN};
use engine_softbody::buggy::dune_buggy_soft;
use engine_softbody::{SoftCapacity, SoftHandle, SpawnOptions};
use engine_vehicle::{VehicleCapacity, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::Vec3;

const HZ: f32 = 240.0;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3]) {
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
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

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new_with_soft(capacity(), VehicleCapacity::default(), SoftCapacity::default()).expect("wgpu");
    sim.set_params(engine_softbody::soft_scene_params(HZ, 10));
    sim.set_soft_settings(engine_softbody::SoftSettings::calibrated_for(HZ));
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(9, Material { friction: 0.8, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    static_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0]);
    sim
}

fn run(sim: &mut WgpuSim, steps: u32) {
    for _ in 0..steps {
        sim.step().expect("step");
    }
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h).iter().map(|b| Vec3::from_array(b.position)).collect()
}

fn yielded(sim: &WgpuSim, h: SoftHandle) -> usize {
    sim.read_soft_beams(h).iter().filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0).count()
}

fn broken(sim: &WgpuSim, h: SoftHandle) -> usize {
    sim.read_soft_beams(h).iter().filter(|b| b.broken()).count()
}

/// Bolted parts whose joints have all broken (loose), by name.
fn loose_parts(sim: &WgpuSim, h: SoftHandle) -> Vec<String> {
    let e = sim.soft_host().entry(h);
    let joints = sim.download_joints();
    e.rigid_parts
        .iter()
        .zip(&e.desc.rigid_parts)
        .filter(|((_, r), _)| r.clone().all(|j| joints[j as usize].flags & JOINT_FLAG_BROKEN != 0))
        .map(|(_, p)| p.name.clone())
        .collect()
}

fn spawn(sim: &mut WgpuSim, pos: [f32; 3], vel: [f32; 3]) -> SoftHandle {
    let opts = SpawnOptions { velocity: vel, ..Default::default() };
    sim.add_soft_vehicle(&dune_buggy_soft(), pos, [0.0, 0.0, 0.0, 1.0], &opts).expect("spawn")
}

#[test]
fn hybrid_buggy_stands_on_its_wheels_without_bending() {
    let mut sim = world();
    let h = spawn(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
    run(&mut sim, 720);
    let core = sim.soft_host().entry(h).core.expect("core").0;
    let st = sim.vehicle_state(core);
    let p = positions(&sim, h);
    let lowest = p.iter().map(|x| x.y).fold(f32::MAX, f32::min);
    eprintln!("contacts {} lowest node {lowest:.3} yielded {} broken {} loose {:?}", st.contacts, yielded(&sim, h), broken(&sim, h), loose_parts(&sim, h));
    assert_eq!(st.contacts, 4, "stands on four wheels");
    assert!(lowest > 0.15, "frame clear of the ground ({lowest})");
    assert_eq!(yielded(&sim, h), 0, "standing does not bend the frame");
    assert_eq!(broken(&sim, h), 0);
    assert!(loose_parts(&sim, h).is_empty());
    // Driving off and braking leaves it whole.
    sim.set_vehicle_input(core, VehicleInput { forward: 1.0, ..Default::default() });
    run(&mut sim, 720);
    let speed = sim.vehicle_state(core).forward_speed;
    sim.set_vehicle_input(core, VehicleInput { brake: 1.0, ..Default::default() });
    run(&mut sim, 480);
    eprintln!("drove at {speed:.1} m/s; yielded {} broken {} loose {:?}", yielded(&sim, h), broken(&sim, h), loose_parts(&sim, h));
    assert!(speed > 8.0, "drives ({speed} m/s)");
    assert_eq!(yielded(&sim, h), 0, "driving does not bend the frame");
    assert!(loose_parts(&sim, h).is_empty());
}

#[test]
fn hybrid_buggy_bends_and_loses_its_front_wheels_in_a_wall_crash() {
    let mut sim = world();
    // A concrete wall across the path, 6 m ahead of the front bumper.
    static_box(&mut sim, [0.0, 1.0, 8.0], [5.0, 1.0, 0.5]);
    let speed = 14.0; // 50 km/h
    let h = spawn(&mut sim, [0.0, 0.05, 0.0], [0.0, 0.0, speed]);
    let before = positions(&sim, h);
    run(&mut sim, (HZ * 2.0) as u32);
    let after = positions(&sim, h);
    let e = sim.soft_host().entry(h).clone();
    // Length of the frame proper (frame nodes only: bumpers and nose may be gone).
    let frame: Vec<usize> = (0..after.len()).filter(|&i| e.desc.nodes[i].name.starts_with("frame:")).collect();
    let extent = |p: &[Vec3]| frame.iter().map(|&i| p[i].z).fold(f32::MIN, f32::max) - frame.iter().map(|&i| p[i].z).fold(f32::MAX, f32::min);
    let (len0, len1) = (extent(&before), extent(&after));
    // The safety cell: the hoop and roof nodes (frame nodes above 1.3 m at rest).
    let cell: Vec<usize> = frame.iter().copied().filter(|&i| before[i].y > 1.3).collect();
    let cell_change = cell
        .iter()
        .flat_map(|&i| cell.iter().map(move |&j| (i, j)))
        .map(|(i, j)| ((after[i] - after[j]).length() - (before[i] - before[j]).length()).abs())
        .fold(0.0f32, f32::max);
    let core = e.core.unwrap().0;
    let st = sim.vehicle_state(core);
    let groups: Vec<u32> = sim.read_soft_beams(h).iter().filter(|b| b.broken()).map(|b| b.group).collect();
    eprintln!(
        "frame crush {:.3} m (length {len0:.2} → {len1:.2}); cell distortion {cell_change:.3} m; yielded {} broken {} (groups {:?}); loose {:?}; core parts broken {:#b} wheels {:#b}",
        len0 - len1,
        yielded(&sim, h),
        broken(&sim, h),
        groups,
        loose_parts(&sim, h),
        st.parts_broken,
        st.wheels_broken,
    );
    assert!(yielded(&sim, h) > 0, "the frame bent");
    assert!(cell_change < 0.05, "the safety cell holds its shape ({cell_change} m)");
    // Its tyres stand proud of the bumper: they take the wall and tear their arms out.
    assert_eq!(st.wheels_broken & 0b11, 0b11, "both front wheels torn off");
    assert_eq!(st.wheels_broken & 0b1100, 0, "rear wheels stay");
    assert!(!groups.contains(&2), "the rear bumper stays on in a frontal crash");
}

#[test]
fn a_struck_seat_tears_off_its_mounts() {
    let mut sim = world();
    let h = spawn(&mut sim, [0.0, 0.05, 0.0], [0.0; 3]);
    run(&mut sim, 480);
    let e = sim.soft_host().entry(h).clone();
    let k = e.desc.rigid_parts.iter().position(|p| p.name == "seat_l").unwrap();
    let seat = sim.read_body(e.rigid_parts[k].0).position;
    // The demo's cannonball (400 kg, r 0.6) at 25 m/s from the open left side (+X).
    let body = sim.add_body({
        let mut b = BodyState::dynamic(Vec3::new(seat[0] + 2.0, seat[1] + 0.25, seat[2]), glam::Quat::IDENTITY, 400.0, Vec3::splat(57.6));
        b.linear_vel = [-25.0, 0.0, 0.0];
        b
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Sphere as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [0.6, 0.0, 0.0, 0.6],
    });
    run(&mut sim, 240);
    let loose = loose_parts(&sim, h);
    eprintln!("loose {loose:?}; yielded {} broken {}", yielded(&sim, h), broken(&sim, h));
    assert!(loose.contains(&"seat_l".to_string()), "the struck seat came loose");
    assert!(!loose.contains(&"seat_r".to_string()), "the other seat stays");
}

/// Calibration probe (run by hand): frame crush and damage in a 50 km/h wall
/// crash across crumple-zone thresholds.
#[test]
#[ignore]
fn hybrid_buggy_crash_probe() {
    use engine_softbody::buggy::{dune_buggy_soft_with, BuggyBeams};
    let pole = std::env::var("POLE").is_ok();
    let springs: Vec<f32> = std::env::var("SPRINGS").ok().map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![3e6]);
    let cases: Vec<(f32, f32, f32, f32)> = springs.iter().flat_map(|&sp| [(12e3, 150e3, 40e3, sp), (6e3, 100e3, 30e3, sp)]).collect();
    for (cd, cs, celld, sp) in cases {
        if cd == 12e3 && std::env::var("ALL").is_err() { continue; }
        let k = BuggyBeams { crumple_deform: cd, crumple_strength: cs, cell_deform: celld, spring: sp, ..Default::default() };
        let mut sim = world();
        // A wall across the path, or (POLE=1) a 0.3 m post between the front wheels.
        if pole { static_box(&mut sim, [0.0, 1.5, 6.0], [0.15, 1.5, 0.15]) } else { static_box(&mut sim, [0.0, 1.0, 8.0], [5.0, 1.0, 0.5]) }
        let desc = dune_buggy_soft_with(&k);
        let h = sim.add_soft_vehicle(&desc, [0.0, 0.05, 0.0], [0.0, 0.0, 0.0, 1.0], &SpawnOptions { velocity: [0.0, 0.0, 14.0], ..Default::default() }).unwrap();
        let before = positions(&sim, h);
        run(&mut sim, (HZ * 2.0) as u32);
        let after = positions(&sim, h);
        let e = sim.soft_host().entry(h).clone();
        let frame: Vec<usize> = (0..after.len()).filter(|&i| e.desc.nodes[i].name.starts_with("frame:")).collect();
        let front = |p: &[Vec3]| frame.iter().map(|&i| p[i].z).fold(f32::MIN, f32::max);
        let back = |p: &[Vec3]| frame.iter().map(|&i| p[i].z).fold(f32::MAX, f32::min);
        let st = sim.vehicle_state(e.core.unwrap().0);
        let groups: std::collections::BTreeSet<u32> = sim.read_soft_beams(h).iter().filter(|b| b.broken()).map(|b| b.group).collect();
        let cell: Vec<usize> = frame.iter().copied().filter(|&i| before[i].y > 1.3).collect();
        let distortion = cell.iter().flat_map(|&i| cell.iter().map(move |&j| (i, j)))
            .map(|(i, j)| ((after[i] - after[j]).length() - (before[i] - before[j]).length()).abs()).fold(0.0f32, f32::max);
        eprintln!(
            "spring {sp:.1e} crumple {cd:>6.0}/{cs:>7.0} cell {celld:>6.0}: crush {:.3} m, cell {distortion:.3} m, yielded {}, broken {} groups {groups:?}, loose {:?}, wheels {:#b}",
            (front(&before) - back(&before)) - (front(&after) - back(&after)),
            yielded(&sim, h),
            broken(&sim, h),
            loose_parts(&sim, h),
            st.wheels_broken
        );
    }
}
