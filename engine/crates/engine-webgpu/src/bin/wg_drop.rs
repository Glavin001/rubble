//! One box on a plane through the wgpu step — first fair-path smoke.

use engine_schema::{BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;

fn main() -> anyhow::Result<()> {
    let mut sim = WgpuSim::new(WorldCapacity::small())?;
    println!("adapter {}", sim.chip());
    let ground = sim.add_body(BodyState::static_plane_support());
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Plane as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [0.0, 1.0, 0.0, 0.5],
    });
    let mass = 1.0f32;
    let h = 0.5f32;
    let inertia = (1.0 / 6.0) * mass * (2.0 * h) * (2.0 * h);
    let box_id = sim.add_body(BodyState {
        position: [0.0, 2.0, 0.0],
        inv_mass: 1.0 / mass,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [1.0 / inertia; 3],
        flags: 1 | 8,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0; 3],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body: box_id,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h, h, h, (3.0 * h * h).sqrt()],
    });
    let mut params = engine_schema::AvbdParams::default();
    params.iterations = 4;
    sim.set_params(params);
    for i in 0..180 {
        match sim.step() {
            Ok(c) => {
                if i % 30 == 0 || i == 179 || std::env::var_os("WG_EVERY").is_some() {
                    println!(
                    "step {i} y={:.3} bodies={} contacts={} cand={} colors={} awake={} ms={:.2} ov=0x{:x}",
                    sim.body_y(box_id),
                    c.body_count,
                    c.contact_count,
                    c.candidate_count,
                    c.color_count,
                    c.awake_count,
                    sim.timings_ms(),
                    c.overflow_flags
                );
                }
            }
            Err(e) => {
                println!("step {i} ERR {e} y={:.3}", sim.body_y(box_id));
                break;
            }
        }
    }
    Ok(())
}
