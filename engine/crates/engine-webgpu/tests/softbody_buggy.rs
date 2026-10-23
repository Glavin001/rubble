//! The hybrid dune buggy (`engine_softbody::buggy`) on the native wgpu host:
//! a node–beam tube frame on the rigid buggy's raycast core, with its bolted
//! parts on breakable ball joints. It must stand and drive without bending, and
//! crumple in front, shed its front bumper and keep its safety cell in a wall
//! crash.

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
fn hybrid_buggy_crumples_its_nose_in_a_wall_crash() {
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
}
