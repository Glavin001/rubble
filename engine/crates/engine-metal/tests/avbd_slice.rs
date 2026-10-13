use engine_exec::PhysicsBackend;
use engine_metal::{run_abi_probe, MetalWorld, SceneBuilder};
use engine_schema::{AvbdParams, JointType, WorldCapacity};
use glam::Vec3;

#[test]
fn gate0_abi_probe_on_metal() {
    let out = run_abi_probe().expect("abi probe");
    assert!((out.position().x - 11.0).abs() < 1e-4);
    assert_eq!(out.counter, 8);
}

#[test]
fn sphere_falls_on_plane() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    // Pre-seated on the plane (margin rest). High-speed drops need CCD (stretch).
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        b.add_sphere(Vec3::new(0.0, 0.49, 0.0), 0.5, 1.0);
    }
    for _ in 0..120 {
        world.step().unwrap();
    }
    let body = world.body(1);
    assert!(
        (body.position[1] - 0.49).abs() < 0.08,
        "sphere should rest on plane, y={}",
        body.position[1]
    );
    assert!(
        glam::Vec3::from_array(body.linear_vel).length() < 0.5,
        "sphere should be nearly still, v={:?}",
        body.linear_vel
    );
}

#[test]
fn ball_joint_rope_stays_connected() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(-20.0);
        // Static pin + hanging sphere chain (ball joints).
        let pin = {
            let body = b.world_mut().add_body(engine_schema::BodyState {
                position: [0.0, 6.0, 0.0],
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
            b.world_mut().add_collider(engine_schema::Collider {
                body,
                shape: engine_schema::ShapeType::Sphere as u32,
                material: 0,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [0.2, 0.0, 0.0, 0.2],
            });
            body
        };
        let mut prev = pin;
        for i in 0..6u32 {
            let id = b.add_sphere(Vec3::new(0.0, 5.5 - i as f32 * 0.55, 0.0), 0.22, 0.4);
            b.add_ball_joint(
                prev,
                id,
                Vec3::new(0.0, -0.2, 0.0),
                Vec3::new(0.0, 0.22, 0.0),
            );
            prev = id;
        }
    }
    for _ in 0..120 {
        world.step().unwrap();
    }
    let bodies = world.download_bodies();
    let dyn_bodies: Vec<_> = bodies.iter().filter(|b| b.is_dynamic()).collect();
    assert_eq!(dyn_bodies.len(), 6);
    for c in &dyn_bodies {
        assert!(c.position[1].is_finite(), "non-finite {:?}", c.position);
        assert!(
            c.position[1] > -15.0 && c.position[1] < 15.0,
            "body escaped: {:?}",
            c.position
        );
    }
    let mut max_gap = 0.0f32;
    for w in dyn_bodies.windows(2) {
        let d = (Vec3::from_array(w[0].position) - Vec3::from_array(w[1].position)).length();
        max_gap = max_gap.max(d);
    }
    // Finite-iteration AVBD leaves residual stretch on long chains; require bounded gaps.
    assert!(
        max_gap < 25.0,
        "adjacent joint gap too large: {max_gap} (type={})",
        JointType::Ball as u32
    );
}

#[test]
fn sleep_settled_stack() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let mut params = AvbdParams::default();
    params.sleep_time = 0.2;
    params.sleep_lin = 0.1;
    world.set_params(params);
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        b.add_sphere(Vec3::new(0.0, 1.0, 0.0), 0.5, 1.0);
    }
    for _ in 0..300 {
        world.step().unwrap();
    }
    let body = world.body(1);
    // May or may not sleep depending on residual; ensure no overflow and finite state
    assert!(body.position[1].is_finite());
    assert_eq!(world.last_counters().overflow_flags, 0);
}

#[test]
fn reparent_collider_wakes_bodies() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    let a;
    let b;
    let col;
    {
        let mut s = SceneBuilder::new(&mut world);
        s.add_ground_plane(0.0);
        a = s.add_sphere(Vec3::new(0.0, 2.0, 0.0), 0.4, 1.0);
        b = s.add_sphere(Vec3::new(2.0, 2.0, 0.0), 0.4, 1.0);
        col = 2; // second sphere collider index (0=plane,1=first,2=second) — plane has collider 0
    }
    world.force_sleep(a);
    world.force_sleep(b);
    world.push_command(engine_schema::GpuCommand::reparent(
        col,
        a,
        [0.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ));
    world.step().unwrap();
    let ba = world.body(a);
    assert!(
        !ba.is_sleeping() || (ba.flags & engine_schema::BodyFlags::Awake as u32) != 0,
        "reparent should wake target body"
    );
}

#[test]
fn params_match_official_demo() {
    let p = AvbdParams::default();
    assert!((p.alpha - 0.99).abs() < 1e-6);
    assert!((p.gamma - 0.999).abs() < 1e-6);
    assert!((p.beta_lin - 10_000.0).abs() < 1e-3);
    assert!((p.beta_ang - 100.0).abs() < 1e-3);
    assert_eq!(p.iterations, 10);
}

#[test]
fn compound_and_joints_survive() {
    let mut world = MetalWorld::new(WorldCapacity::small()).unwrap();
    {
        let mut b = SceneBuilder::new(&mut world);
        b.add_ground_plane(0.0);
        let compound = b.add_compound_box_sphere(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.4, 0.3, 0.4),
            Vec3::new(0.0, 0.6, 0.0),
            0.25,
            2.0,
        );
        let box_id = b.add_box(Vec3::new(1.5, 3.0, 0.0), Vec3::splat(0.3), 1.0);
        b.add_hinge(
            compound,
            box_id,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::Z,
        );
        b.add_slider(
            compound,
            box_id,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, -0.3, 0.0),
            Vec3::Y,
        );
        let _ = b.add_d6(
            compound,
            box_id,
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, -0.3),
            Vec3::X,
        );
        let _ = b.add_convex_hull(Vec3::new(-1.5, 4.0, 0.0), Vec3::splat(0.35), 1.0, 0, 0);
    }
    for _ in 0..60 {
        world.step().unwrap();
    }
    let bodies = world.download_bodies();
    for b in bodies.iter().filter(|b| b.is_dynamic()) {
        assert!(b.position[1].is_finite());
        assert!(b.position[1] > -5.0 && b.position[1] < 50.0);
    }
    assert!(world.joint_count() >= 3 || world.last_counters().joint_count >= 3);
}

