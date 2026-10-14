//! Smoke: small column grid holds height on the wgpu/WGSL path.
use engine_schema::{
    AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, FLAG_REUSE_CONTACTS,
};
use engine_webgpu::WgpuSim;

#[test]
fn columns_10k_holds_height() {
    let n = 32u32;
    let h = 10u32;
    let bodies = n * n * h + 1;
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: bodies.next_power_of_two().max(4096),
        max_colliders: bodies.next_power_of_two().max(4096),
        max_contacts: (bodies * 16).next_power_of_two().min(1 << 21),
        max_joints: 64,
        max_candidates: (bodies * 32).next_power_of_two().min(1 << 21),
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut params = AvbdParams::default();
    params.iterations = 4;
    params.flags |= FLAG_REUSE_CONTACTS;
    sim.set_params(params);

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
    let mut n_dyn = 0u32;
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
                n_dyn += 1;
            }
        }
    }

    let mut ymax0 = 10.0f32;
    for s in 0..60 {
        let c = sim.step().expect("step");
        if s == 0 || s == 59 {
            let mut ymax = f32::NEG_INFINITY;
            for i in 1..=n_dyn {
                ymax = ymax.max(sim.read_body(i).position[1]);
            }
            if s == 0 {
                ymax0 = ymax;
            }
            println!("t={s}: maxY={ymax:.2} contacts={} ov=0x{:x}", c.contact_count, c.overflow_flags);
            if s == 59 {
                assert!(
                    ymax > ymax0 * 0.7 && ymax < ymax0 * 1.5,
                    "unstable maxY={ymax} initial={ymax0}"
                );
            }
        }
    }
}
