//! Vehicles on the Metal host, and Metal ↔ wgpu parity for the same scene.

use engine_exec::PhysicsBackend;
use engine_metal::MetalWorld;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{presets, VehicleInput};
use engine_webgpu::WgpuSim;

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

fn ground_body() -> (BodyState, Collider) {
    let b = BodyState {
        position: [0.0, -5.0, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0, -5.0, 0.0],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    };
    let c = Collider {
        body: 0,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    };
    (b, c)
}

const ASPHALT: Material = Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 };

/// Inputs per phase: (input, steps).
fn script() -> Vec<(VehicleInput, u32)> {
    vec![
        (VehicleInput::default(), 90),
        (VehicleInput { forward: 1.0, ..Default::default() }, 120),
        (VehicleInput { forward: 0.5, steer: 0.6, ..Default::default() }, 90),
        (VehicleInput { brake: 1.0, ..Default::default() }, 90),
    ]
}

fn run_metal() -> Vec<[f32; 3]> {
    let mut w = MetalWorld::new(capacity()).expect("metal");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    w.set_params(p);
    w.set_default_friction(ASPHALT.friction);
    let (b, c) = ground_body();
    w.add_body(b);
    w.add_collider(c);
    let v = w.add_vehicle(&presets::sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    let chassis = w.vehicle_host().chassis_body(v);
    let mut out = Vec::new();
    for (input, n) in script() {
        w.set_vehicle_input(v, input);
        for _ in 0..n {
            w.step_gpu().expect("step");
        }
        out.push(w.body(chassis).position);
    }
    let st = w.vehicle_state(v);
    assert_eq!(st.wheels.len(), 4);
    out
}

fn run_wgpu() -> Vec<[f32; 3]> {
    let mut sim = WgpuSim::new(capacity()).expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, ASPHALT);
    let (b, c) = ground_body();
    sim.add_body(b);
    sim.add_collider(c);
    let v = sim.add_vehicle(&presets::sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    let chassis = sim.vehicle_host().chassis_body(v);
    let mut out = Vec::new();
    for (input, n) in script() {
        sim.set_vehicle_input(v, input);
        for _ in 0..n {
            sim.step().expect("step");
        }
        out.push(sim.read_body(chassis).position);
    }
    out
}

#[test]
fn metal_vehicle_matches_wgpu_trajectory() {
    let m = run_metal();
    let g = run_wgpu();
    for (i, (a, b)) in m.iter().zip(&g).enumerate() {
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        eprintln!("phase {i}: metal {a:?} wgpu {b:?} |Δ| {d:.4}");
        // Same Slang source on both backends; f32 reduction order may differ slightly.
        let travelled = (b[0] * b[0] + b[2] * b[2]).sqrt();
        assert!(d < 0.05 + 0.03 * travelled, "phase {i}: backends diverge by {d}");
    }
    assert!(m[1][2] > 10.0, "Metal sedan drives forward: {:?}", m[1]);
}

#[test]
fn metal_detach_and_raycast() {
    let mut w = MetalWorld::new(capacity()).expect("metal");
    w.set_default_friction(1.0);
    let (b, c) = ground_body();
    w.add_body(b);
    w.add_collider(c);
    let v = w.add_vehicle(&presets::destructible_sedan(), [0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..90 {
        w.step_gpu().expect("step");
    }
    w.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
    for _ in 0..60 {
        w.step_gpu().expect("step");
    }
    w.detach_wheel(v, 1);
    w.detach_part(v, 3);
    w.step_gpu().expect("step");
    let st = w.vehicle_state(v);
    assert_eq!(st.wheel_mask & 2, 0);
    assert!(!st.engine_connected);
    assert!(w.body(w.vehicle_host().wheel_body(v, 1)).is_dynamic());
    let hits = w.raycast_batch(&[engine_vehicle::Ray {
        origin: [30.0, 4.0, 30.0],
        dir: [0.0, -1.0, 0.0],
        max_dist: 10.0,
        exclude_body: None,
    }]);
    let h = hits[0].expect("ground");
    assert!((h.distance - 4.0).abs() < 1e-3);
}
