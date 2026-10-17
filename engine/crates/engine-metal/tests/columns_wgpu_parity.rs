//! Metal vs wgpu columns: find where WGSL diverges.
use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{
    AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_REUSE_CONTACTS,
};
use engine_webgpu::WgpuSim;
use glam::Vec3;

fn capacity_for(n_dyn: u32) -> WorldCapacity {
    let bodies = (n_dyn + 1).next_power_of_two().max(4096);
    WorldCapacity {
        max_bodies: bodies,
        max_colliders: bodies,
        max_contacts: (bodies * 16).min(1 << 21),
        max_joints: 64,
        max_candidates: (bodies * 32).min(1 << 21),
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn add_columns_metal(world: &mut MetalWorld, n: u32, h: u32) {
    let mut b = SceneBuilder::new(world);
    b.add_ground_box_extent(0.5, 200.0);
    let half = Vec3::splat(0.5);
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                b.add_box(
                    Vec3::new(
                        (x as f32 - n as f32 / 2.0) * 1.5,
                        1.0 + y as f32,
                        (z as f32 - n as f32 / 2.0) * 1.5,
                    ),
                    half,
                    1.0,
                );
            }
        }
    }
}

fn add_columns_wgpu(sim: &mut WgpuSim, n: u32, h: u32) {
    let half_h = 25.0f32;
    let hx = 200.0f32;
    sim.add_body(BodyState {
        position: [0.0, 0.5 - half_h, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0; 3],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body: 0,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [hx, half_h, hx, hx.max(half_h)],
    });
    let hx_b = 0.5f32;
    let mass = 1.0f32;
    let inertia = (1.0 / 6.0) * mass;
    let r = (0.75f32).sqrt();
    for x in 0..n {
        for z in 0..n {
            for y in 0..h {
                let pos = [
                    (x as f32 - n as f32 / 2.0) * 1.5,
                    1.0 + y as f32,
                    (z as f32 - n as f32 / 2.0) * 1.5,
                ];
                let id = sim.add_body(BodyState {
                    position: pos,
                    inv_mass: 1.0 / mass,
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    inv_inertia: [1.0 / inertia; 3],
                    flags: 1 | 8,
                    linear_vel: [0.0; 3],
                    sleep_timer: 0.0,
                    angular_vel: [0.0; 3],
                    generation: 0,
                    initial_pos: pos,
                    initial_pad: 0.0,
                    initial_rot: [0.0, 0.0, 0.0, 1.0],
                });
                sim.add_collider(Collider {
                    body: id,
                    shape: ShapeType::Box as u32,
                    material: 0,
                    generation: 0,
                    local_pos: [0.0; 3],
                    density: 1.0,
                    local_rot: [0.0, 0.0, 0.0, 1.0],
                    params: [hx_b, hx_b, hx_b, r],
                });
            }
        }
    }
}

fn run_pair(n: u32, h: u32, steps: u32) {
    let n_dyn = n * n * h;
    let cap = capacity_for(n_dyn);
    let mut params = AvbdParams::default();
    params.iterations = 4;
    params.flags |= FLAG_REUSE_CONTACTS;

    let mut metal = MetalWorld::new(cap.clone()).unwrap();
    metal.set_params(params);
    add_columns_metal(&mut metal, n, h);

    let mut wg = WgpuSim::new(cap).unwrap();
    wg.set_params(params);
    add_columns_wgpu(&mut wg, n, h);

    for step in 0..steps {
        metal.step().expect("Metal step must not overflow");
        wg.step().expect("WGSL step must not overflow");
        if step % 30 != 29 && step + 1 != steps {
            continue;
        }
        let m = metal.download_bodies();
        let w = wg.download_bodies();
        assert_eq!(m.len(), w.len(), "backend body counts differ");
        let mut worst = 0.0f32;
        for id in 1..m.len() {
            let expected_y = 1.0 + ((id - 1) as u32 % h) as f32;
            for (backend, b) in [("Metal", &m[id]), ("WGSL", &w[id])] {
                assert!(
                    b.position.iter().all(|x| x.is_finite()),
                    "{backend}: step {step}, body {id} is nonfinite"
                );
                assert!((b.position[1] - expected_y).abs() < 0.15,
                    "{backend}: step {step}, body {id} lost its column layer: y={} expected={expected_y}", b.position[1]);
            }
            let delta = (0..3)
                .map(|axis| (m[id].position[axis] - w[id].position[axis]).abs())
                .fold(0.0, f32::max);
            worst = worst.max(delta);
            assert!(delta < 0.02,
                "step {step}, body {id}/{n_dyn}: Metal/WGSL position divergence {delta:.5} m exceeds 2 cm; Metal={:?}, WGSL={:?}", m[id].position, w[id].position);
        }
        println!("{n_dyn} bodies, step {step}: maximum Metal/WGSL position delta={worst:.6} m");
    }
}

#[test]
fn columns_16k_metal_vs_wgpu() {
    run_pair(40, 10, 180);
}

#[test]
fn columns_25k_metal_vs_wgpu() {
    run_pair(50, 10, 180);
}

#[test]
fn columns_100k_metal_vs_wgpu() {
    run_pair(100, 10, 180);
}
