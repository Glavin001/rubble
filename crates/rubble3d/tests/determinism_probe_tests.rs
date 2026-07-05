//! D13 probes: two identical worlds stepped in lockstep must produce bitwise
//! identical trajectories on the same adapter.
//!
//! Determinism fixes landed with these probes (M1):
//! - Luby coloring mark/apply split (same-dispatch color read/write race made
//!   colorings improper run-to-run → primal races between adjacent bodies).
//! - LBVH pair overflow grow-and-retry (silent drop of a scheduling-dependent
//!   pair subset when demand exceeded `bodies*8`).
//! - Total-order contact sort before the solve (atomicAdd emission order fed
//!   float summation order).
//! - One compute pass per solve color (pass-boundary hazard ordering between
//!   Gauss-Seidel colors).
//!
//! One residual source remains (see the ignored full-iteration probe below).

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};

fn pyramid_world(solver_iterations: u32) -> Option<World> {
    let mut world = World::new(SimConfig {
        solver_iterations,
        max_bodies: 65536,
        ..Default::default()
    })
    .ok()?;
    world.add_body(&RigidBodyDesc {
        position: Vec3::new(0.0, -0.5, 0.0),
        mass: 0.0,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(50.0, 0.5, 50.0),
        },
        ..Default::default()
    });
    const SIZE: usize = 16;
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            world.add_body(&RigidBodyDesc {
                position: Vec3::new(
                    x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                    y as f32 * 0.85 + 0.5,
                    0.0,
                ),
                mass: 1.0,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(0.5, 0.25, 0.25),
                },
                ..Default::default()
            });
        }
    }
    Some(world)
}

fn run_lockstep(solver_iterations: u32, steps: usize) {
    let (Some(mut w1), Some(mut w2)) = (
        pyramid_world(solver_iterations),
        pyramid_world(solver_iterations),
    ) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let handles: Vec<_> = (0..w1.body_count() as u32)
        .map(|i| rubble_math::BodyHandle::new(i, 0))
        .collect();

    for step in 0..steps {
        w1.step();
        w2.step();
        for (bi, h) in handles.iter().enumerate() {
            let s1 = (
                w1.get_position(*h),
                w1.get_rotation(*h),
                w1.get_velocity(*h),
                w1.get_angular_velocity(*h),
            );
            let s2 = (
                w2.get_position(*h),
                w2.get_rotation(*h),
                w2.get_velocity(*h),
                w2.get_angular_velocity(*h),
            );
            let eq = s1.0.map(|v| v.to_array()) == s2.0.map(|v| v.to_array())
                && s1.1.map(|v| v.to_array()) == s2.1.map(|v| v.to_array())
                && s1.2.map(|v| v.to_array()) == s2.2.map(|v| v.to_array())
                && s1.3.map(|v| v.to_array()) == s2.3.map(|v| v.to_array());
            assert!(
                eq,
                "worlds diverged at step {step}, body {bi}:\n  w1 {s1:?}\n  w2 {s2:?}"
            );
        }
    }

    // Cross-process fingerprint: same binary + adapter must print the same
    // hash on every run (compare manually / in CI logs across runs).
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for h in &handles {
        if let Some(p) = w1.get_position(*h) {
            for b in p.to_array().iter().flat_map(|v| v.to_bits().to_le_bytes()) {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
        }
    }
    println!("final state fnv1a: {hash:016x}");
}

/// Regression gate: detection stack + one solve sweep, bit-deterministic.
#[test]
fn two_worlds_pyramid_bitwise_deterministic() {
    run_lockstep(1, 360);
}

/// Pinned repro of the one KNOWN residual nondeterminism source (kept ignored
/// until the M2 GPU-resident rewrite replaces the affected stages).
///
/// Findings from the M1 bisection, all on lavapipe (software Vulkan):
/// - `LP_NUM_THREADS=1` (single-threaded execution): bit-deterministic over
///   360 steps → the residual source is execution-scheduling-dependent.
/// - `solver_iterations` 0 and 1: bit-deterministic multithreaded.
/// - `solver_iterations` 12: diverges (~1e-10 first delta) around the settle
///   phase, even with one queue submit per compute pass — so it is not a
///   missing inter-pass barrier; some intra-dispatch or driver-level
///   scheduling dependence remains on multithreaded llvmpipe.
/// - Physically the divergence is round-off-scale per step; it only matters
///   for replay determinism and chaotic-scene test thresholds.
#[test]
#[ignore = "residual multithread scheduling nondeterminism; tracked for the M2 rewrite"]
fn two_worlds_pyramid_full_iterations_bitwise_deterministic() {
    run_lockstep(12, 360);
}
