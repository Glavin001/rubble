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

// ---- Two-way ground coupling: the same lab scenes on both backends ---------------

const GRIP: Material = Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 };
const ICE: Material = Material { friction: 0.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 };

fn box_collider(body: u32, local: [f32; 3], half: [f32; 3], material: u32) -> Collider {
    Collider {
        body,
        shape: ShapeType::Box as u32,
        material,
        generation: 0,
        local_pos: local,
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()],
    }
}

/// Builds and runs a coupling scene on one backend; returns the tracked bodies'
/// positions after each phase. `$step` advances one step, `$read` reads a body.
macro_rules! coupling_scene {
    ($w:ident, $scene:expr, $step:expr, $read:expr) => {{
        let mut p = AvbdParams::default();
        p.iterations = 10;
        $w.set_params(p);
        let (b, c) = ground_body();
        $w.add_body(b);
        $w.add_collider(c);
        let mut tracked = Vec::new();
        let mut script: Vec<(engine_vehicle::VehicleHandle, VehicleInput, u32)> = Vec::new();
        if $scene == "slab" {
            // Sedan on a 2.5 t slab with a grippy deck (material 1) on frictionless skids (2).
            let m = 2500.0f32;
            let (hx, hz) = (6.0f32, 30.0f32);
            let slab = $w.add_body(BodyState::dynamic(
                glam::Vec3::new(0.0, 0.25, 0.0),
                glam::Quat::IDENTITY,
                m,
                glam::Vec3::new(m * (0.25 + hz * hz) / 3.0, m * (hx * hx + hz * hz) / 3.0, m * (hx * hx + 0.25) / 3.0),
            ));
            $w.add_collider(box_collider(slab, [0.0, 0.1, 0.0], [hx, 0.15, hz], 1));
            $w.add_collider(box_collider(slab, [0.0, -0.1, 0.0], [hx, 0.15, hz], 2));
            let v = $w.add_vehicle(&presets::sedan(), [0.0, 2.0, -20.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
            tracked.push(slab);
            tracked.push($w.vehicle_host().chassis_body(v));
            script.push((v, VehicleInput::default(), 90));
            script.push((v, VehicleInput { forward: 0.5, ..Default::default() }, 120));
            script.push((v, VehicleInput { brake: 1.0, ..Default::default() }, 90));
        } else {
            // Parked kart (handbrake) on a monster truck's roof; the truck drives and turns.
            let truck = $w.add_vehicle(&presets::monster_truck(), [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("truck");
            let kart = $w.add_vehicle(&presets::kart(), [0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("kart");
            $w.set_vehicle_input(kart, VehicleInput { handbrake: 1.0, ..Default::default() });
            tracked.push($w.vehicle_host().chassis_body(truck));
            tracked.push($w.vehicle_host().chassis_body(kart));
            script.push((truck, VehicleInput::default(), 120));
            script.push((truck, VehicleInput { forward: 0.15, ..Default::default() }, 180));
            script.push((truck, VehicleInput { forward: 0.15, steer: 0.35, ..Default::default() }, 180));
        }
        let mut out: Vec<Vec<[f32; 3]>> = Vec::new();
        for (v, input, n) in script {
            $w.set_vehicle_input(v, input);
            for _ in 0..n {
                $step(&mut $w);
            }
            out.push(tracked.iter().map(|&b| $read(&$w, b).position).collect());
        }
        out
    }};
}

fn coupling_metal(scene: &str) -> Vec<Vec<[f32; 3]>> {
    let mut w = MetalWorld::new(capacity()).expect("metal");
    w.set_default_friction(1.0);
    assert_eq!(w.add_material(GRIP), 1);
    assert_eq!(w.add_material(ICE), 2);
    coupling_scene!(w, scene, |w: &mut MetalWorld| w.step_gpu().expect("step"), |w: &MetalWorld, b: u32| w.body(b))
}

fn coupling_wgpu(scene: &str) -> Vec<Vec<[f32; 3]>> {
    let mut w = WgpuSim::new(capacity()).expect("wgpu");
    w.set_material(0, ASPHALT);
    w.set_material(1, GRIP);
    w.set_material(2, ICE);
    coupling_scene!(w, scene, |w: &mut WgpuSim| w.step().expect("step"), |w: &WgpuSim, b: u32| w.read_body(b))
}

fn assert_coupling_parity(scene: &str) -> Vec<Vec<[f32; 3]>> {
    let m = coupling_metal(scene);
    let g = coupling_wgpu(scene);
    let dist = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let mut worst = Vec::new();
    for (i, (pm, pg)) in m.iter().zip(&g).enumerate() {
        // Coupled bodies share one error budget: drift in the car's drivetrain moves the
        // ground body too, so scale by the largest travel of any tracked body.
        let travelled = pg.iter().zip(&g[0]).map(|(&b, &b0)| dist(b, b0)).fold(0.0f32, f32::max);
        for (k, (&a, &b)) in pm.iter().zip(pg).enumerate() {
            // On frictionless ice nothing restores the slab sideways (neutral stability),
            // so backend f32 differences grow laterally; compare the along-track motion.
            let d = if scene == "slab" { (a[2] - b[2]).abs() } else { dist(a, b) };
            eprintln!("{scene} phase {i} body {k}: metal {a:?} wgpu {b:?} |Δ| {d:.4} (travel {travelled:.2})");
            worst.push((d / (0.05 + 0.03 * travelled), i, k, d));
        }
    }
    for (ratio, i, k, d) in worst {
        assert!(ratio < 1.0, "{scene} phase {i} body {k}: backends diverge by {d}");
    }
    g
}

#[test]
fn metal_matches_wgpu_car_pushing_a_sliding_slab() {
    let g = assert_coupling_parity("slab");
    // Throttle phase: the slab is pushed backward while the car moves forward.
    assert!(g[1][0][2] < -1.0, "slab pushed back: {:?}", g[1][0]);
    assert!(g[1][1][2] > -17.0, "car moved forward: {:?}", g[1][1]);
}

#[test]
fn metal_matches_wgpu_kart_carried_on_a_truck() {
    let g = assert_coupling_parity("carry");
    let last = g.last().unwrap();
    let (truck, kart) = (last[0], last[1]);
    let dx = ((truck[0] - kart[0]).powi(2) + (truck[2] - kart[2]).powi(2)).sqrt();
    assert!(dx < 0.5 && kart[1] > truck[1] + 0.5, "kart still on the roof: truck {truck:?} kart {kart:?}");
}
