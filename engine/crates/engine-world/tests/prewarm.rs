//! Kernels are built when a step first runs them, or ahead of time by `prewarm`.

use rubble::*;

fn world() -> World {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 256,
            max_colliders: 256,
            max_contacts: 8_192,
            max_joints: 16,
            max_candidates: 16_384,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 1024,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [10.0, 0.5, 10.0])
        .unwrap();
    let b = w.create_body(&BodyDef::dynamic([0.0, 1.0, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5, 0.5, 0.5])
        .unwrap();
    w
}

#[test]
fn a_world_builds_only_the_kernels_it_runs_until_prewarmed() {
    let mut w = world();
    for _ in 0..4 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    // A world with no characters never ran their kernels, so never built them.
    assert!(!w.sim().kernel_is_built("char_move"));
    assert!(!w.sim().kernel_is_built("char_post"));

    let built = w.prewarm(KernelGroups::CHARACTERS);
    assert_eq!(built, 2, "char_move and char_post");
    assert!(w.sim().kernel_is_built("char_move"));
    assert!(w.sim().kernel_is_built("char_post"));
    // Other groups stay unbuilt, and a second prewarm has nothing left to do.
    assert!(!w.sim().kernel_is_built("projectile_step"));
    assert_eq!(w.prewarm(KernelGroups::CHARACTERS), 0);

    // Everything: afterwards no step compiles a kernel.
    assert!(w.prewarm(KernelGroups::ALL) > 0);
    assert_eq!(w.prewarm(KernelGroups::ALL), 0);
    for _ in 0..4 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
}

#[test]
fn a_background_prewarm_reports_progress_while_the_world_steps() {
    let mut w = world();
    let p = w.prewarm_async(KernelGroups::VEHICLES | KernelGroups::PROJECTILES);
    let (_, total) = p.progress();
    assert!(total > 2, "the vehicle kernels and projectile_step: {total}");
    // The world keeps stepping while the kernels compile.
    let mut steps = 0;
    while !p.is_done() {
        w.step().unwrap();
        w.poll().unwrap();
        steps += 1;
        assert!(steps < 100_000, "the prewarm never finished");
    }
    w.poll_blocking().unwrap();
    assert_eq!(p.progress(), (total, total));
    assert!(w.sim().kernel_is_built("projectile_step"));
    assert!(w.sim().kernel_is_built("vehicle_iterate"));
    assert!(!w.sim().kernel_is_built("char_move"));
    assert_eq!(w.prewarm(KernelGroups::VEHICLES | KernelGroups::PROJECTILES), 0);
}
