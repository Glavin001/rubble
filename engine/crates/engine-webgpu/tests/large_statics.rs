//! Oversized statics (wider than half a grid cell) are found through a list, not
//! the broadphase grid. The list used to hold 10 entries claimed by an atomic
//! counter: with more, a random subset was dropped each step, and a vehicle on a
//! long road box lost its wheel contacts on some steps.

use engine_exec::ExecError;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity, OVERFLOW_LARGE_STATICS};
use engine_vehicle::{presets, VehicleInput};
use engine_webgpu::WgpuSim;

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32_768,
        max_joints: 64,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim
}

fn static_collider(sim: &mut WgpuSim, pos: [f32; 3], shape: ShapeType, params: [f32; 4]) {
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
        shape: shape as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params,
    });
}

/// Parked 0.6 m spheres far from everything (vehicle.html's cannonballs): each
/// is wider than half the small grid cell, so each takes a list entry.
fn parked_spheres(sim: &mut WgpuSim, n: u32) {
    for i in 0..n {
        static_collider(sim, [-4000.0 - 5.0 * i as f32, -4000.0, -4000.0], ShapeType::Sphere, [0.6, 0.0, 0.0, 0.6]);
    }
}

#[test]
fn a_truck_on_a_long_road_keeps_its_wheels_with_many_large_statics() {
    let mut sim = world();
    // Ground first, then the parked spheres, then the road: the road comes after
    // more than ten other large statics in collider order.
    static_collider(&mut sim, [0.0, -5.0, 0.0], ShapeType::Box, [400.0, 5.0, 400.0, 565.7]);
    parked_spheres(&mut sim, 14);
    // A 60 m road 3 m up (four grid cells and more in every direction but x).
    static_collider(&mut sim, [0.0, 1.5, 0.0], ShapeType::Box, [2.5, 1.5, 30.0, 30.1]);
    let v = sim.add_vehicle(&presets::monster_truck(), [0.0, 5.0, -20.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..60 {
        sim.step().expect("step");
    }
    sim.set_vehicle_input(v, VehicleInput { forward: 0.6, ..Default::default() });
    let mut missing = 0;
    for _ in 0..180 {
        sim.step().expect("step");
        if sim.vehicle_state(v).contacts != 4 {
            missing += 1;
        }
    }
    let st = sim.vehicle_state(v);
    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
    eprintln!("steps without 4 contacts: {missing}/180, speed {:.2}, chassis {:?}", st.forward_speed, c.position);
    assert_eq!(missing, 0, "the road was dropped from the large-static list on some steps");
    assert!(st.forward_speed > 4.0, "drives on its wheels, not its shells: {}", st.forward_speed);
}

#[test]
fn more_large_statics_than_the_list_holds_is_reported() {
    let mut sim = world();
    static_collider(&mut sim, [0.0, -5.0, 0.0], ShapeType::Box, [400.0, 5.0, 400.0, 565.7]);
    parked_spheres(&mut sim, engine_schema::LARGE_STATIC_CAP + 8);
    // One mover so the step is not skipped as fully asleep.
    sim.add_body(BodyState::dynamic(glam::Vec3::new(0.0, 0.5, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::splat(0.1)));
    match sim.step() {
        Err(ExecError::Overflow(flags)) => assert_ne!(flags & OVERFLOW_LARGE_STATICS, 0, "flags {flags:#x}"),
        other => panic!("expected the large-static overflow flag, got {other:?}"),
    }
}
