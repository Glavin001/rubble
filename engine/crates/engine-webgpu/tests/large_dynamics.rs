//! Box-only scenes run on the uniform broadphase grid, whose cell fits the small
//! dynamics; a dynamic wider than a cell goes on a list that holds 16. The fracture
//! lab's cuboid suburb (96 houses of box pieces, each with planks wider than its
//! bricks' cell) has hundreds and failed its first step with
//! `Overflow(OVERFLOW_LARGE_DYNAMICS)`. With the grid chosen automatically, such a
//! scene takes the hierarchical grid (any size, every pair found) instead.

use engine_exec::ExecError;
use engine_schema::{
    AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity, OVERFLOW_LARGE_DYNAMICS,
};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

/// Planks: more than the 16 the uniform grid's large list holds.
const PLANKS: u32 = 24;
const PLANK_HALF: [f32; 3] = [3.0, 0.1, 3.0];
const BRICK_HALF: f32 = 0.2;

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 256,
        max_colliders: 256,
        max_contacts: 16_384,
        max_joints: 1,
        max_candidates: 16_384,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(
        0,
        Material {
            friction: 0.8,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    sim
}

fn boxed(sim: &mut WgpuSim, body: BodyState, half: [f32; 3]) -> u32 {
    let id = sim.add_body(body);
    let r = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
    sim.add_collider(Collider {
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

fn dynamic_box(sim: &mut WgpuSim, at: Vec3, half: [f32; 3]) -> u32 {
    let mass = 8.0 * half[0] * half[1] * half[2] * 500.0;
    let h = Vec3::from(half) * 2.0;
    let inertia = Vec3::new(
        h.y * h.y + h.z * h.z,
        h.x * h.x + h.z * h.z,
        h.x * h.x + h.y * h.y,
    ) * (mass / 12.0);
    boxed(
        sim,
        BodyState::dynamic(at, Quat::IDENTITY, mass, inertia),
        half,
    )
}

/// A ground box, `PLANKS` planks on it in a row, a brick on each plank.
fn planks_and_bricks(sim: &mut WgpuSim) -> Vec<(u32, u32)> {
    let mut ground = BodyState::dynamic(Vec3::new(0.0, -1.0, 0.0), Quat::IDENTITY, 0.0, Vec3::ZERO);
    ground.flags = 0;
    boxed(sim, ground, [200.0, 1.0, 200.0]);
    (0..PLANKS)
        .map(|i| {
            let x = (i as f32 - PLANKS as f32 / 2.0) * 7.0;
            let plank = dynamic_box(sim, Vec3::new(x, PLANK_HALF[1] + 0.01, 0.0), PLANK_HALF);
            let brick = dynamic_box(
                sim,
                Vec3::new(x + 1.0, 2.0 * PLANK_HALF[1] + BRICK_HALF + 0.02, 0.5),
                [BRICK_HALF; 3],
            );
            (plank, brick)
        })
        .collect()
}

#[test]
fn a_box_scene_with_many_large_dynamics_steps_and_keeps_every_pair() {
    let mut sim = world();
    let pairs = planks_and_bricks(&mut sim);
    for step in 0..90 {
        if let Err(e) = sim.step() {
            panic!("step {step}: {e:?}");
        }
    }
    for (plank, brick) in pairs {
        let p = sim.read_body(plank).position[1];
        let b = sim.read_body(brick).position[1];
        // Plank on the ground, brick on the plank: each pair was found.
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

/// The uniform grid, asked for explicitly, still fails such a scene with the flag
/// (never drops the 17th large dynamic's pairs).
#[test]
fn the_forced_uniform_grid_still_reports_the_overflow() {
    let mut sim = world();
    sim.set_grid_levels(Some(false));
    planks_and_bricks(&mut sim);
    match sim.step() {
        Err(ExecError::Overflow(flags)) => {
            assert_ne!(flags & OVERFLOW_LARGE_DYNAMICS, 0, "flags {flags:#x}")
        }
        other => panic!("expected the large-dynamics overflow flag, got {other:?}"),
    }
}
