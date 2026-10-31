//! Fingerprints of vehicle runs that never touch soil (deformable terrain T5).
//! Ignored: run by hand before and after a change to the vehicle kernels and
//! compare the printed hashes, which must be equal (soil off is bit-identical).
//!
//! `cargo test -p engine-webgpu --test soil_fingerprint -- --ignored --nocapture`

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{presets, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32_768,
        max_joints: 64,
        max_candidates: 65_536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn sim() -> WgpuSim {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(
        0,
        Material {
            friction: 1.0,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim
}

fn box_ground(sim: &mut WgpuSim) {
    let mut b = BodyState::static_plane_support();
    b.position = [0.0, -5.0, 0.0];
    b.initial_pos = b.position;
    let body = sim.add_body(b);
    let h = [400.0f32, 5.0, 400.0];
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [
            h[0],
            h[1],
            h[2],
            (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt(),
        ],
    });
}

fn bumpy_field() -> HeightField {
    HeightField::from_fn(161, 161, 0.25, 0.25, |x, z| {
        0.04 * (0.7 * x).sin() * (0.5 * z).cos()
    })
    .unwrap()
}

fn fingerprint(sim: &mut WgpuSim) -> u64 {
    let v = sim
        .add_vehicle(&presets::sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        .expect("vehicle");
    for step in 0..300u32 {
        let input = VehicleInput {
            forward: if step > 60 { 0.6 } else { 0.0 },
            steer: if step > 150 { 0.3 } else { 0.0 },
            brake: if step > 260 { 0.5 } else { 0.0 },
            ..Default::default()
        };
        sim.set_vehicle_input(v, input);
        sim.step().expect("step");
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: u32| {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    for b in sim.download_bodies() {
        for x in b
            .position
            .iter()
            .chain(&b.orientation)
            .chain(&b.linear_vel)
            .chain(&b.angular_vel)
        {
            eat(x.to_bits());
        }
    }
    for w in sim.vehicle_state(v).wheels {
        for x in [
            w.suspension_force,
            w.longitudinal_force,
            w.lateral_force,
            w.omega,
            w.suspension_length,
        ] {
            eat(x.to_bits());
        }
    }
    h
}

#[test]
#[ignore]
fn vehicle_runs_without_soil() {
    let mut a = sim();
    box_ground(&mut a);
    let on_box = fingerprint(&mut a);
    let mut b = sim();
    let mut arena = HullArena::new();
    b.add_height_field(&mut arena, &bumpy_field(), Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let on_field = fingerprint(&mut b);
    println!("fingerprint box {on_box:016x} heightfield {on_field:016x}");
}
