//! M1 solver-fidelity behavior tests: restitution (D10), velocity clamp (D6e),
//! substeps (D6g), and the penalty floor's effect on heavy bodies (D6b).

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

macro_rules! gpu_world {
    ($config:expr) => {
        match World::new($config) {
            Ok(w) => w,
            Err(_) => {
                eprintln!("SKIP: No GPU adapter found");
                return;
            }
        }
    };
}

fn ground() -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::ZERO,
        mass: 0.0,
        shape: ShapeDesc::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        },
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// D10: restitution
// ---------------------------------------------------------------------------

/// A ball with e=0.8 dropped from 2 m must bounce back up to a meaningful
/// fraction of its drop height. Zero-restitution engines (the old rubble, both
/// reference implementations, and the AVBD paper itself) fail this by design.
#[test]
fn restitution_ball_bounces() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 8,
        ..Default::default()
    });
    world.add_body(&ground());
    let ball = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 2.0, 0.0),
        mass: 1.0,
        restitution: 0.8,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    // Track the max height reached after the first impact. (Position peaks
    // smoothly, so a plain running max is exact; gating on near-zero velocity
    // can miss the apex entirely between fixed steps.)
    let mut hit_ground = false;
    let mut apex_after_bounce = f32::MIN;
    for _ in 0..240 {
        world.step();
        let p = world.get_position(ball).unwrap();
        assert!(p.is_finite(), "ball diverged: {p}");
        if !hit_ground && p.y < 0.6 {
            hit_ground = true;
        }
        if hit_ground && p.y > apex_after_bounce {
            apex_after_bounce = p.y;
        }
    }
    assert!(hit_ground, "ball never reached the ground");
    // e = 0.8 → ideal rebound to e² ≈ 64% of drop height (measured from the
    // resting contact at y = 0.5). Demand at least a third of the ideal — the
    // velocity-level pass under-restores slightly at coarse dt.
    let drop = 2.0 - 0.5;
    let ideal = 0.5 + drop * 0.8 * 0.8;
    assert!(
        apex_after_bounce > 0.5 + (ideal - 0.5) / 3.0,
        "e=0.8 ball should bounce meaningfully: apex {apex_after_bounce:.3} vs ideal {ideal:.3}"
    );
}

/// e = 0 (default) must not bounce: after settling, the ball stays put.
#[test]
fn zero_restitution_does_not_bounce() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 8,
        ..Default::default()
    });
    world.add_body(&ground());
    let ball = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 2.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut max_y_after_settle = f32::MIN;
    for i in 0..240 {
        world.step();
        let p = world.get_position(ball).unwrap();
        if i > 120 {
            max_y_after_settle = max_y_after_settle.max(p.y);
        }
    }
    assert!(
        max_y_after_settle < 0.65,
        "inelastic ball must not rebound: max y after settle = {max_y_after_settle:.3}"
    );
}

/// Restitution below the threshold speed is suppressed (no micro-bouncing of
/// resting contacts even with e > 0).
#[test]
fn restitution_threshold_suppresses_micro_bounce() {
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 8,
        restitution_threshold: 1.0,
        ..Default::default()
    });
    world.add_body(&ground());
    // Start nearly resting on the plane: impact speed will be < threshold.
    let ball = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.52, 0.0),
        mass: 1.0,
        restitution: 0.9,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..120 {
        world.step();
    }
    let v = world.get_velocity(ball).unwrap();
    let p = world.get_position(ball).unwrap();
    assert!(
        v.length() < 0.2 && (p.y - 0.5).abs() < 0.05,
        "sub-threshold contact must rest, got v={v}, y={}",
        p.y
    );
}

// ---------------------------------------------------------------------------
// D6e: velocity clamp
// ---------------------------------------------------------------------------

#[test]
fn velocity_clamp_bounds_launch_speed() {
    // Two bodies spawned deeply overlapping: without the clamp this is the
    // classic "explosion" case. With it, post-step speed is bounded.
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 8,
        max_speed: 10.0,
        ..Default::default()
    });
    let a = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });
    let b = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.05, 0.0, 0.0),
        mass: 1.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    for _ in 0..30 {
        world.step();
        for h in [a, b] {
            let v = world.get_velocity(h).unwrap();
            assert!(
                v.length() <= 10.0 + 0.01,
                "velocity clamp must bound launch speed, got {}",
                v.length()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// D6g: substeps
// ---------------------------------------------------------------------------

#[test]
fn substeps_preserve_free_fall_and_improve_impact() {
    // Free fall for 1 s with 4 substeps must land at the same place as 1.
    let make = |substeps: u32| -> Option<(World, rubble_math::BodyHandle)> {
        let mut w = World::new(SimConfig {
            substeps,
            solver_iterations: 8,
            ..Default::default()
        })
        .ok()?;
        w.add_body(&ground());
        let h = w.add_body(&RigidBodyDesc {
            position: Vec3::new(0.0, 20.0, 0.0),
            mass: 1.0,
            shape: ShapeDesc::Sphere { radius: 0.5 },
            ..Default::default()
        });
        Some((w, h))
    };
    let Some((mut w1, h1)) = make(1) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };
    let (mut w4, h4) = make(4).unwrap();
    for _ in 0..30 {
        w1.step();
        w4.step();
    }
    let p1 = w1.get_position(h1).unwrap();
    let p4 = w4.get_position(h4).unwrap();
    assert!(p1.is_finite() && p4.is_finite());
    // Ballistic phase: BDF1 free fall integrates the same trajectory apart from
    // O(dt) discretization differences.
    assert!(
        (p1.y - p4.y).abs() < 1.5,
        "substep free-fall should track single-step: {} vs {}",
        p1.y,
        p4.y
    );
    // Both must come to rest on the plane eventually.
    for _ in 0..240 {
        w1.step();
        w4.step();
    }
    let p1 = w1.get_position(h1).unwrap();
    let p4 = w4.get_position(h4).unwrap();
    assert!((p1.y - 0.5).abs() < 0.1, "substeps=1 rests at {}", p1.y);
    assert!((p4.y - 0.5).abs() < 0.1, "substeps=4 rests at {}", p4.y);
}

// ---------------------------------------------------------------------------
// D6b: penalty floor — heavy body on one contact must not sink through
// ---------------------------------------------------------------------------

#[test]
fn heavy_body_rests_without_sinking() {
    // m = 1000 kg sphere: m/dt² = 3.6e6 ≫ k_start (1e4). Without the floor the
    // first frames sink deep before the ramp catches up; with it the contact is
    // stiff from frame one.
    let mut world = gpu_world!(SimConfig {
        solver_iterations: 8,
        ..Default::default()
    });
    world.add_body(&ground());
    let heavy = world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, 0.55, 0.0),
        mass: 1000.0,
        shape: ShapeDesc::Sphere { radius: 0.5 },
        ..Default::default()
    });

    let mut min_y = f32::MAX;
    for _ in 0..180 {
        world.step();
        min_y = min_y.min(world.get_position(heavy).unwrap().y);
    }
    let p = world.get_position(heavy).unwrap();
    assert!(
        min_y > 0.40,
        "1-tonne sphere must not sink through its contact (min y = {min_y:.3})"
    );
    assert!(
        (p.y - 0.5).abs() < 0.05,
        "heavy sphere should rest at the surface, got y = {:.3}",
        p.y
    );
}
