//! Breakable joints on the Metal host (same Slang source as the wgpu host; see
//! engine-webgpu/tests/breakable_joints.rs).

use engine_exec::PhysicsBackend;
use engine_metal::{MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, BodyState, Collider, JointDesc, ShapeType, WorldCapacity, JOINT_BIG, JOINT_FLAG_BROKEN};
use glam::Vec3;

fn static_box(world: &mut MetalWorld, pos: [f32; 3], half: f32) -> u32 {
    let body = world.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    world.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half, half, half, half * 3f32.sqrt()],
    });
    body
}

/// A 10 kg box on a ball joint 1 m below a static block (load 100 N).
fn hanging_box(break_force: f32) -> (MetalWorld, u32, u32) {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut p = AvbdParams::default();
    p.iterations = 10;
    p.sleep_time = 1e6;
    world.set_params(p);
    let anchor = static_box(&mut world, [0.0, 6.0, 0.0], 0.2);
    let bob = SceneBuilder::new(&mut world).add_box(Vec3::new(0.0, 5.0, 0.0), Vec3::splat(0.25), 10.0);
    let never = |x: f32| if x > 0.0 { x } else { JOINT_BIG };
    let j = world.add_joint_desc(
        &JointDesc::ball(anchor, bob, Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 0.5, 0.0))
            .with_break(never(break_force), JOINT_BIG),
    );
    (world, bob, j)
}

#[test]
fn metal_joint_below_its_break_force_holds() {
    let (mut world, bob, j) = hanging_box(150.0);
    for _ in 0..300 {
        world.step().unwrap();
    }
    let joint = world.joint(j);
    let force = Vec3::from_slice(&joint.lambda[..3]).length();
    assert_eq!(joint.flags & JOINT_FLAG_BROKEN, 0, "the joint holds");
    assert!(world.poll_joint_breaks().is_empty());
    assert!((force - 100.0).abs() < 5.0, "converged force is the weight: {force}");
    assert!((world.body(bob).position[1] - 5.0).abs() < 0.01);
}

#[test]
fn metal_joint_above_its_break_force_breaks_once() {
    let (mut world, bob, j) = hanging_box(60.0);
    let mut breaks = Vec::new();
    for _ in 0..120 {
        world.step().unwrap();
        breaks.extend(world.poll_joint_breaks());
    }
    assert_ne!(world.joint(j).flags & JOINT_FLAG_BROKEN, 0);
    assert_eq!(breaks.len(), 1, "exactly one EVENT_BREAK: {breaks:?}");
    assert_eq!(breaks[0].joint, j);
    assert!(breaks[0].force > 60.0);
    assert!(world.body(bob).position[1] < 0.0, "the box fell");
}
