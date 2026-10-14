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

fn max_y_metal(world: &MetalWorld) -> f32 {
    world
        .download_bodies()
        .iter()
        .filter(|b| b.is_dynamic())
        .map(|b| b.position[1])
        .fold(f32::NEG_INFINITY, f32::max)
}

fn max_y_wgpu(sim: &WgpuSim, n_dyn: u32) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for i in 1..=n_dyn {
        m = m.max(sim.read_body(i).position[1]);
    }
    m
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

    for s in 0..steps {
        let cm = metal.step().unwrap();
        let cw = wg.step().unwrap();
        if s == 0 || s == steps - 1 || s % 10 == 9 {
            let ym = max_y_metal(&metal);
            let yw = max_y_wgpu(&wg, n_dyn);
            println!(
                "n={n} t={s}: metal_y={ym:.2} wgpu_y={yw:.2} contacts m/w={}/{} awake m/w={}/{}",
                cm.contact_count, cw.contact_count, cm.awake_count, cw.awake_count
            );
        }
    }
    let ym = max_y_metal(&metal);
    let yw = max_y_wgpu(&wg, n_dyn);
    assert!(
        yw < ym * 1.5 + 2.0,
        "wgpu exploded vs metal: wgpu_y={yw} metal_y={ym}"
    );
}

#[test]
fn columns_16k_metal_vs_wgpu() {
    run_pair(40, 10, 30);
}

#[test]
fn columns_25k_metal_vs_wgpu() {
    run_pair(50, 10, 30);
}
