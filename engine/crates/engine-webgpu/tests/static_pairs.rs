//! Broadphase ownership of dynamic–static pairs (the grid query's index rule).

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_webgpu::WgpuSim;

fn body(pos: [f32; 3], inv_mass: f32, flags: u32) -> BodyState {
    let i = if inv_mass > 0.0 { 6.0 * inv_mass / 0.25 } else { 0.0 };
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: [i; 3], flags,
        linear_vel: [0.0; 3], sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn boxc(b: u32, h: f32) -> Collider {
    Collider { body: b, shape: ShapeType::Box as u32, material: 0, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: [h, h, h, h * 1.7320508] }
}

/// An awake body owns its pairs with statics at any collider index: before the CCD
/// branch, the grid query kept a static only if its index was higher, so a box fell
/// straight through a small static pedestal added before it (y -10.4 m on the
/// uniform grid). More than 512 colliders, so the grid path runs.
#[test]
fn a_box_lands_on_a_lower_index_static_box() {
    for hier in [false, true] {
        let mut sim = WgpuSim::new(WorldCapacity { max_bodies: 2048, max_colliders: 2048, max_contacts: 65536, max_joints: 64, max_candidates: 131072, max_colors: 32, max_commands: 64, max_hull_vertices: 64 }).unwrap();
        let mut p = AvbdParams::default();
        p.dt = 1.0 / 60.0;
        sim.set_params(p);
        sim.set_grid_levels(Some(hier));
        // 600 small static boxes on a row far away (grid path: > 512 colliders), then
        // a static pedestal (low index), then the dynamic box above it.
        for k in 0..600u32 {
            let b = sim.add_body(body([100.0 + k as f32, 0.25, 100.0], 0.0, 0));
            sim.add_collider(boxc(b, 0.25));
        }
        let ped = sim.add_body(body([0.0, 0.25, 0.0], 0.0, 0));
        sim.add_collider(boxc(ped, 0.25));
        let dynb = sim.add_body(body([0.0, 1.0, 0.0], 1.0, 9));
        sim.add_collider(boxc(dynb, 0.25));
        for _ in 0..90 { sim.step().unwrap(); }
        let y = sim.read_body(dynb).position[1];
        println!("hier {hier}: dynamic box y {y:.3} (resting on the pedestal: 0.75)");
        assert!((y - 0.75).abs() < 0.02, "hier {hier}: the box did not land on the pedestal (y {y})");
    }
}
