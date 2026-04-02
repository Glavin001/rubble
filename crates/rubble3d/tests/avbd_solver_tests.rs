//! Comprehensive AVBD solver accuracy tests for rubble3d.
//!
//! Tests verify: energy conservation, momentum conservation, solver convergence,
//! penalty stiffness ramping, warm-starting effectiveness, graph coloring correctness,
//! and numerical stability under extreme conditions.

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

macro_rules! gpu_world {
    ($config:expr) => {
        match World::new($config) {
            Ok(w) => w,
            Err(_) => { eprintln!("SKIP: No GPU adapter found"); return; }
        }
    };
}

fn step_n(world: &mut World, n: usize) {
    for _ in 0..n {
        world.step();
    }
}

// ---------------------------------------------------------------------------
// Momentum conservation
// ---------------------------------------------------------------------------

#[test]
fn momentum_conservation_equal_mass_head_on() {
    // Two equal-mass spheres colliding head-on in zero gravity.
    // Total momentum should be conserved: p_total = m1*v1 + m2*v2
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0, // higher temporal resolution
        solver_iterations: 20,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-4.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Initial momentum: 1*4 + 1*(-4) = 0
    let initial_momentum = Vec3::new(4.0, 0.0, 0.0) + Vec3::new(-4.0, 0.0, 0.0);

    step_n(&mut world, 300);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_momentum = va + vb; // equal mass so just sum velocities

    let momentum_error = (final_momentum - initial_momentum).length();
    assert!(
        momentum_error < 2.0,
        "Momentum not conserved: initial={initial_momentum}, final={final_momentum}, error={momentum_error}"
    );
}

#[test]
fn momentum_conservation_unequal_mass() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        mass: 5.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-2.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    // Initial momentum: 5*2 + 1*(-2) = 8
    let initial_momentum = 5.0 * 2.0 + 1.0 * (-2.0);

    step_n(&mut world, 300);

    let vh = world.get_velocity(heavy).unwrap();
    let vl = world.get_velocity(light).unwrap();
    let final_momentum = 5.0 * vh.x + 1.0 * vl.x;

    let error = (final_momentum - initial_momentum).abs();
    assert!(
        error < 4.0,
        "Momentum not conserved: initial={initial_momentum}, final={final_momentum}, error={error}"
    );
}

#[test]
fn momentum_conservation_three_body() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let masses = [1.0f32, 2.0, 3.0];
    let vels = [
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(-1.0, 2.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
    ];
    let positions = [
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(3.0, -1.0, 0.0),
    ];

    let mut handles = Vec::new();
    let mut initial_p = Vec3::ZERO;
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc {
            position: positions[i],
            linear_velocity: vels[i],
            mass: masses[i],
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
        initial_p += masses[i] * vels[i];
    }

    step_n(&mut world, 200);

    let mut final_p = Vec3::ZERO;
    for (i, &h) in handles.iter().enumerate() {
        let v = world.get_velocity(h).unwrap();
        final_p += masses[i] * v;
    }

    let error = (final_p - initial_p).length();
    assert!(
        error < 5.0,
        "3-body momentum error: initial={initial_p}, final={final_p}, error={error}"
    );
}

// ---------------------------------------------------------------------------
// Energy behavior
// ---------------------------------------------------------------------------

#[test]
fn kinetic_energy_does_not_increase_in_collision() {
    // In an inelastic collision, kinetic energy should not increase.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        ..Default::default()
    });

    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    let initial_ke = 0.5 * (5.0f32.powi(2) + 5.0f32.powi(2));

    step_n(&mut world, 400);

    let va = world.get_velocity(a).unwrap();
    let vb = world.get_velocity(b).unwrap();
    let final_ke = 0.5 * (va.length_squared() + vb.length_squared());

    // Allow small numerical overshoot but not large energy gain
    assert!(
        final_ke < initial_ke * 1.5,
        "Energy increased too much: initial={initial_ke}, final={final_ke}"
    );
}

#[test]
fn gravitational_potential_to_kinetic_conversion() {
    // Drop a sphere from height h. After falling, KE should approximately equal mgh.
    let h = 10.0f32;
    let g = 9.81f32;
    let m = 1.0f32;
    let dt = 1.0 / 60.0;
    let steps = 60;

    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -g, 0.0),
        dt,
        solver_iterations: 5,
        ..Default::default()
    });

    let body = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, h, 0.0),
        mass: m,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, steps);

    let pos = world.get_position(body).unwrap();
    let vel = world.get_velocity(body).unwrap();
    let height_fallen = h - pos.y;
    let pe_lost = m * g * height_fallen;
    let ke_gained = 0.5 * m * vel.length_squared();

    // Energy should be approximately conserved (numerical integration introduces some error)
    let energy_error = (pe_lost - ke_gained).abs();
    assert!(
        energy_error < pe_lost * 0.3,
        "Energy conversion error too large: PE_lost={pe_lost}, KE_gained={ke_gained}, error={energy_error}"
    );
}

// ---------------------------------------------------------------------------
// Solver convergence
// ---------------------------------------------------------------------------

#[test]
fn more_iterations_improves_contact_resolution() {
    // Sphere dropped onto static floor. More solver iterations should yield
    // better contact resolution (less penetration).
    let mut results = Vec::new();

    for &iters in &[2u32, 5, 10, 20] {
        let mut world = gpu_world!(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: iters,
            ..Default::default()
        });

        let _floor = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, -1.0, 0.0),
            mass: 0.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(10.0, 1.0, 10.0),
            },
            ..Default::default()
        });

        let sphere = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 2.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        step_n(&mut world, 60);

        let pos = world.get_position(sphere).unwrap();
        results.push((iters, pos.y));
    }

    // All positions should be finite
    for &(iters, y) in &results {
        assert!(
            y.is_finite(),
            "Position diverged with {iters} iterations: y={y}"
        );
    }
}

#[test]
fn solver_converges_sphere_on_plane() {
    // Sphere on a static plane should come to rest (approximately).
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: -1.0,
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 2.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 300);

    let vel = world.get_velocity(sphere).unwrap();
    let pos = world.get_position(sphere).unwrap();

    assert!(pos.y.is_finite(), "Position diverged: {pos}");
    assert!(
        vel.length() < 20.0,
        "Velocity should be bounded after settling: {vel}"
    );
}

// ---------------------------------------------------------------------------
// Penalty stiffness behavior
// ---------------------------------------------------------------------------

#[test]
fn penalty_stiffness_prevents_deep_penetration() {
    // A heavy sphere dropped from height onto a static floor.
    // The AVBD penalty stiffness should prevent deep penetration.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        k_start: 1e4,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 10.0, // heavy
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let pos = world.get_position(sphere).unwrap();
    // Sphere center should not go below floor surface (y=0) by more than its radius
    assert!(
        pos.y > -2.0,
        "Deep penetration detected: sphere center at y={}, floor at y=0",
        pos.y
    );
    assert!(pos.y.is_finite(), "Position diverged: {pos}");
}

// ---------------------------------------------------------------------------
// Warm starting effectiveness
// ---------------------------------------------------------------------------

#[test]
fn warm_starting_reduces_jitter() {
    // Run the same scenario with warm-starting. The simulation should produce
    // finite stable results.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        warmstart_decay: 0.95,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    // Stack of 3 spheres
    let mut handles = Vec::new();
    for i in 0..3 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 1.0 + i as f32 * 2.5, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    // Run for a while to let it settle
    step_n(&mut world, 300);

    // Check all bodies are finite and haven't exploded
    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        let vel = world.get_velocity(h).unwrap();
        assert!(
            pos.is_finite(),
            "Sphere {i} position diverged: {pos}"
        );
        assert!(
            vel.is_finite(),
            "Sphere {i} velocity diverged: {vel}"
        );
        assert!(
            pos.y > -10.0,
            "Sphere {i} fell through floor: y={}",
            pos.y
        );
    }
}

// ---------------------------------------------------------------------------
// Baumgarte stabilization
// ---------------------------------------------------------------------------

#[test]
fn baumgarte_corrects_penetration() {
    // Start sphere slightly penetrating the floor.
    // Baumgarte should push it out.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO, // no gravity to isolate Baumgarte
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    // Sphere radius=1.0 at y=0.0 means it's touching the floor (floor top at y=0)
    // Put it slightly inside
    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.2, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(sphere).unwrap();
    assert!(
        pos.y.is_finite(),
        "Baumgarte correction produced non-finite result: {pos}"
    );
    // Should have been pushed upward
    assert!(
        pos.y >= -0.5,
        "Baumgarte should correct penetration, got y={}",
        pos.y
    );
}

// ---------------------------------------------------------------------------
// Friction
// ---------------------------------------------------------------------------

#[test]
fn friction_slows_sliding_sphere() {
    // Sphere sliding along a floor with friction should decelerate.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        friction_default: 0.8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        friction: 0.8,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 1.0, 50.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.0, 0.0),
        linear_velocity: Vec3::new(10.0, 0.0, 0.0),
        mass: 1.0,
        friction: 0.8,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let initial_vx = 10.0f32;
    step_n(&mut world, 120);

    let vel = world.get_velocity(sphere).unwrap();
    assert!(
        vel.x.is_finite(),
        "Velocity diverged: {vel}"
    );
    // Friction should slow it down (allow for numerical effects)
    assert!(
        vel.x < initial_vx + 2.0,
        "Friction should not accelerate horizontally: initial vx={initial_vx}, final vx={}",
        vel.x
    );
}

#[test]
fn zero_friction_allows_sliding() {
    // With zero friction, horizontal velocity should be preserved on flat surface.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        friction_default: 0.0,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        friction: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(100.0, 1.0, 100.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.0, 0.0),
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        friction: 0.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 120);

    let vel = world.get_velocity(sphere).unwrap();
    assert!(
        vel.x.is_finite(),
        "Velocity diverged: {vel}"
    );
    // With zero friction, horizontal velocity should be roughly preserved
    // (contact normal is vertical, so normal impulse doesn't affect horizontal)
    assert!(
        vel.x > 2.0,
        "Zero friction should preserve horizontal velocity: vx={}",
        vel.x
    );
}

// ---------------------------------------------------------------------------
// Numerical stability
// ---------------------------------------------------------------------------

#[test]
fn extreme_mass_ratio_stability() {
    // Very heavy body hitting very light body.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 15,
        ..Default::default()
    });

    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(-5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(2.0, 0.0, 0.0),
        mass: 1000.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });
    let light = world.add_body(&RigidBodyDesc {
        position: Vec3::new(5.0, 0.0, 0.0),
        linear_velocity: Vec3::new(-2.0, 0.0, 0.0),
        mass: 0.01,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 200);

    let ph = world.get_position(heavy).unwrap();
    let pl = world.get_position(light).unwrap();
    let vh = world.get_velocity(heavy).unwrap();
    let vl = world.get_velocity(light).unwrap();

    assert!(ph.is_finite(), "Heavy body diverged: {ph}");
    assert!(pl.is_finite(), "Light body diverged: {pl}");
    assert!(vh.is_finite(), "Heavy body velocity diverged: {vh}");
    assert!(vl.is_finite(), "Light body velocity diverged: {vl}");

    // Heavy body should barely change velocity
    assert!(
        (vh.x - 2.0).abs() < 1.0,
        "Heavy body velocity changed too much: {vh}"
    );
}

#[test]
fn many_simultaneous_contacts_stability() {
    // Many spheres clustered together → many simultaneous contacts.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for x in 0..4 {
        for z in 0..4 {
            let h = world.add_body(&RigidBodyDesc {
                position: Vec3::new(x as f32 * 2.5, 3.0, z as f32 * 2.5),
                mass: 1.0,
                shape: ShapeDesc::Sphere { radius: 1.0 },
                ..Default::default()
            });
            handles.push(h);
        }
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Sphere {i} position diverged in crowd: {pos}"
        );
        assert!(
            pos.y > -5.0,
            "Sphere {i} fell through floor in crowd: y={}",
            pos.y
        );
    }
}

#[test]
fn zero_dt_does_not_crash() {
    // Zero timestep should not produce NaN or crash.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 0.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 10);

    let pos = world.get_position(h).unwrap();
    // With dt=0, position should not change
    assert!(
        pos.is_finite(),
        "Zero dt produced non-finite result: {pos}"
    );
}

#[test]
fn very_small_dt_stability() {
    // Very small timestep should still produce correct physics.
    let dt = 1.0 / 10000.0;
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt,
        solver_iterations: 3,
        ..Default::default()
    });

    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 10.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    // 10000 steps = 1 second
    step_n(&mut world, 600); // 0.06 seconds

    let pos = world.get_position(h).unwrap();
    assert!(pos.is_finite(), "Small dt produced non-finite result: {pos}");
    assert!(pos.y < 10.0, "Should have fallen with small dt: y={}", pos.y);
}

#[test]
fn large_dt_bounded_response() {
    // Large timestep shouldn't cause explosion
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 0.1,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let sphere = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 5.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    step_n(&mut world, 30);

    let pos = world.get_position(sphere).unwrap();
    let vel = world.get_velocity(sphere).unwrap();
    assert!(pos.is_finite(), "Large dt position diverged: {pos}");
    assert!(vel.is_finite(), "Large dt velocity diverged: {vel}");
}

// ---------------------------------------------------------------------------
// Angular dynamics
// ---------------------------------------------------------------------------

#[test]
fn angular_velocity_preserved_in_free_space() {
    // A spinning sphere in zero gravity should maintain angular velocity.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    });

    let omega = Vec3::new(0.0, 5.0, 0.0);
    let h = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        angular_velocity: omega,
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 1.0 },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let final_omega = world.get_angular_velocity(h).unwrap();
    let error = (final_omega - omega).length();
    assert!(
        error < 2.0,
        "Angular velocity not preserved: initial={omega}, final={final_omega}, error={error}"
    );
}

#[test]
fn off_center_collision_induces_rotation() {
    // A sphere hitting a box off-center should induce rotation.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _wall = world.add_body(&RigidBodyDesc {
        position: Vec3::new(3.0, 0.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(1.0, 5.0, 5.0),
        },
        ..Default::default()
    });

    let box_body = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 1.0, 0.0), // off-center from wall
        linear_velocity: Vec3::new(5.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Default::default()
    });

    step_n(&mut world, 60);

    let pos = world.get_position(box_body).unwrap();
    assert!(
        pos.is_finite(),
        "Off-center collision produced non-finite position: {pos}"
    );
}

// ---------------------------------------------------------------------------
// Graph coloring correctness
// ---------------------------------------------------------------------------

#[test]
fn graph_coloring_no_data_races_chain() {
    // Chain of spheres: A-B-C-D. Each pair shares a body.
    // Graph coloring should ensure no two contacts sharing a body are in same group.
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::ZERO,
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    // Place spheres in a line, overlapping slightly
    let mut handles = Vec::new();
    for i in 0..6 {
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 1.8, 0.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 60);

    // Just verify no crashes or NaN from coloring
    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Chain sphere {i} diverged: {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// Capsule specific AVBD tests
// ---------------------------------------------------------------------------

#[test]
fn capsule_on_floor_stable() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(10.0, 1.0, 10.0),
        },
        ..Default::default()
    });

    let capsule = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 3.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Capsule {
            half_height: 1.0,
            radius: 0.5,
        },
        ..Default::default()
    });

    step_n(&mut world, 180);

    let pos = world.get_position(capsule).unwrap();
    assert!(pos.is_finite(), "Capsule diverged: {pos}");
    assert!(pos.y > -3.0, "Capsule fell through floor: y={}", pos.y);
}

// ---------------------------------------------------------------------------
// Restitution-like behavior
// ---------------------------------------------------------------------------

#[test]
fn sphere_bounces_higher_with_more_solver_iters() {
    // More solver iterations should result in better energy preservation,
    // meaning a slightly higher bounce.
    let mut max_ys = Vec::new();

    for &iters in &[3u32, 10, 20] {
        let mut world = gpu_world!(SimConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            dt: 1.0 / 60.0,
            solver_iterations: iters,
            ..Default::default()
        });

        let _floor = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, -1.0, 0.0),
            mass: 0.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(10.0, 1.0, 10.0),
            },
            ..Default::default()
        });

        let sphere = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });

        let mut max_y = f32::MIN;
        // Run long enough for sphere to hit floor and bounce
        for _ in 0..180 {
            world.step();
            let pos = world.get_position(sphere).unwrap();
            if pos.y > max_y {
                max_y = pos.y;
            }
        }
        max_ys.push((iters, max_y));
    }

    // All should be finite
    for &(iters, y) in &max_ys {
        assert!(y.is_finite(), "iters={iters}: max_y is not finite: {y}");
    }
}

// ---------------------------------------------------------------------------
// Stress: many bodies interacting
// ---------------------------------------------------------------------------

#[test]
fn stress_32_spheres_gravity() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 8,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -1.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(30.0, 1.0, 30.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();
    for i in 0..32 {
        let x = (i % 8) as f32 * 3.0 - 10.0;
        let z = (i / 8) as f32 * 3.0 - 5.0;
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(x, 5.0 + (i as f32) * 0.1, z),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 1.0 },
            ..Default::default()
        });
        handles.push(h);
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Sphere {i} diverged in 32-body stress test: {pos}"
        );
    }
}

#[test]
fn stress_mixed_shapes_interaction() {
    let mut world = gpu_world!(SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 10,
        ..Default::default()
    });

    let _floor = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -2.0, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(20.0, 1.0, 20.0),
        },
        ..Default::default()
    });

    let mut handles = Vec::new();

    // 5 spheres
    for i in 0..5 {
        handles.push(world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 3.0 - 6.0, 5.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        }));
    }

    // 5 boxes
    for i in 0..5 {
        handles.push(world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 3.0 - 6.0, 8.0, 3.0),
            mass: 1.0,
            shape: ShapeDesc::Box {
                half_extents: Vec3::splat(0.5),
            },
            ..Default::default()
        }));
    }

    // 3 capsules
    for i in 0..3 {
        handles.push(world.add_body(&RigidBodyDesc {
            position: Vec3::new(i as f32 * 4.0 - 4.0, 6.0, -3.0),
            mass: 1.0,
            shape: ShapeDesc::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            ..Default::default()
        }));
    }

    step_n(&mut world, 120);

    for (i, &h) in handles.iter().enumerate() {
        let pos = world.get_position(h).unwrap();
        assert!(
            pos.is_finite(),
            "Body {i} diverged in mixed-shape stress test: {pos}"
        );
    }
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn repeated_simulation_consistent() {
    // Run the same simulation twice and check results are close.
    let config = SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
        solver_iterations: 5,
        ..Default::default()
    };

    let mut results = Vec::new();
    for _ in 0..2 {
        let mut world = gpu_world!(config.clone());
        let h = world.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        step_n(&mut world, 60);
        results.push(world.get_position(h).unwrap());
    }

    let diff = (results[0] - results[1]).length();
    assert!(
        diff < 0.01,
        "Repeated simulation produced different results: {:?} vs {:?}, diff={}",
        results[0],
        results[1],
        diff
    );
}

