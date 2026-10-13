//! Same scene through Metal (Slang→MSL) and wgpu (Slang→SPIR-V→WGSL): trajectories must match.

use engine_exec::PhysicsBackend;
use engine_metal::MetalWorld;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;

fn plane() -> (BodyState, Collider) {
    (
        BodyState::static_plane_support(),
        Collider {
            body: 0,
            shape: ShapeType::Plane as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, 0.5],
        },
    )
}

fn unit_box(body: u32, pos: [f32; 3]) -> (BodyState, Collider) {
    let h = 0.5f32;
    let inertia = (1.0 / 6.0) * (2.0 * h) * (2.0 * h);
    (
        BodyState {
            position: pos,
            inv_mass: 1.0,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [1.0 / inertia; 3],
            flags: 1 | 8,
            linear_vel: [0.0; 3],
            sleep_timer: 0.0,
            angular_vel: [0.0; 3],
            generation: 0,
        },
        Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 1.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [h, h, h, (3.0 * h * h).sqrt()],
        },
    )
}

#[test]
fn box_drop_matches_metal() {
    let mut metal = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut wg = WgpuSim::new(WorldCapacity::small()).unwrap();
    let mut params = AvbdParams::default();
    params.iterations = 4;
    let (pb, pc) = plane();
    let (bb, bc) = unit_box(1, [0.0, 2.0, 0.0]);
    metal.add_body(pb);
    metal.add_collider(pc);
    metal.add_body(bb);
    metal.add_collider(bc);
    wg.add_body(pb);
    wg.add_collider(pc);
    wg.add_body(bb);
    wg.add_collider(bc);
    metal.set_params(params);
    wg.set_params(params);
    let mut worst = 0.0f32;
    for i in 0..120 {
        metal.step().unwrap();
        wg.step().unwrap();
        let m = metal.body(1).position;
        let w = wg.read_body(1).position;
        let d = (0..3).map(|k| (m[k] - w[k]).abs()).fold(0.0, f32::max);
        if i % 20 == 0 || d > 1e-3 {
            println!("step {i} metal_y={:.4} wgpu_y={:.4} d={d:.2e}", m[1], w[1]);
        }
        worst = worst.max(d);
    }
    assert!(worst < 1e-3, "metal vs wgpu diverged by {worst}");
}
