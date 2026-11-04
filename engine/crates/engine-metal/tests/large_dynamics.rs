//! Box-only scenes run on the uniform broadphase grid, whose large-dynamics list
//! holds 16. A box scene with more (the fracture lab's cuboid suburb) takes the
//! hierarchical grid when the grid is chosen automatically, as on WebGPU
//! (engine-webgpu/tests/large_dynamics.rs), instead of failing the step with
//! `OVERFLOW_LARGE_DYNAMICS`.

use engine_exec::PhysicsBackend;
use engine_metal::MetalWorld;
use engine_schema::*;
use glam::{Quat, Vec3};

/// Planks: more than the 16 the uniform grid's large list holds.
const PLANKS: u32 = 24;
const PLANK_HALF: [f32; 3] = [3.0, 0.1, 3.0];
const BRICK_HALF: f32 = 0.2;

fn boxed(w: &mut MetalWorld, body: BodyState, half: [f32; 3]) -> u32 {
    let id = w.add_body(body);
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    w.add_collider(Collider {
        body: id,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], r],
    });
    id
}

fn dynamic_box(w: &mut MetalWorld, at: Vec3, half: [f32; 3]) -> u32 {
    let mass = 8.0 * half[0] * half[1] * half[2] * 500.0;
    let h = Vec3::from(half) * 2.0;
    let inertia = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) * (mass / 12.0);
    boxed(
        w,
        BodyState::dynamic(at, Quat::IDENTITY, mass, inertia),
        half,
    )
}

#[test]
fn a_box_scene_with_many_large_dynamics_steps_and_keeps_every_pair() {
    let mut w = MetalWorld::new(WorldCapacity {
        max_bodies: 256,
        max_colliders: 256,
        max_contacts: 16_384,
        max_joints: 1,
        max_candidates: 16_384,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("metal");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    w.set_params(p);
    let mut ground = BodyState::dynamic(Vec3::new(0.0, -1.0, 0.0), Quat::IDENTITY, 0.0, Vec3::ZERO);
    ground.flags = 0;
    boxed(&mut w, ground, [200.0, 1.0, 200.0]);
    let pairs: Vec<(u32, u32)> = (0..PLANKS)
        .map(|i| {
            let x = (i as f32 - PLANKS as f32 / 2.0) * 7.0;
            let plank = dynamic_box(&mut w, Vec3::new(x, PLANK_HALF[1] + 0.01, 0.0), PLANK_HALF);
            let brick = dynamic_box(
                &mut w,
                Vec3::new(x + 1.0, 2.0 * PLANK_HALF[1] + BRICK_HALF + 0.02, 0.5),
                [BRICK_HALF; 3],
            );
            (plank, brick)
        })
        .collect();
    for step in 0..90 {
        if let Err(e) = w.step() {
            panic!("step {step}: {e:?}");
        }
    }
    for (plank, brick) in pairs {
        let p = w.body(plank).position[1];
        let b = w.body(brick).position[1];
        assert!(
            (p - PLANK_HALF[1]).abs() < 0.05,
            "plank {plank} at y {p}: the plank-ground pair was missed"
        );
        let rest = 2.0 * PLANK_HALF[1] + BRICK_HALF;
        assert!(
            (b - rest).abs() < 0.05,
            "brick {brick} at y {b} (rest {rest}): the brick-plank pair was missed"
        );
    }
}
