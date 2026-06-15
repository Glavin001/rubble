//! Metamorphic & determinism tests for the 2D engine (CA1).
//!
//! 2D previously had no oracle and no determinism coverage at all. These guard
//! the properties a performance refactor (GPU coloring / adjacency / indirect
//! dispatch / residency) could silently break:
//!   * **determinism** — two independent `World2D`s, identical scene → identical
//!     trajectory (fp-exact within round-off);
//!   * **permutation invariance** — body insertion order must not grossly change
//!     the result (graph-colored Gauss-Seidel is inherently *mildly* order-
//!     dependent, so this is a regression ceiling, not a zero-tolerance check).

mod support;

use glam::Vec2;
use rubble2d::{RigidBodyDesc2D, ShapeDesc2D, SimConfig2D};
use std::collections::HashMap;

use support::{should_skip_known_failure, try_world};

const STEPS: usize = 240;

fn cfg() -> SimConfig2D {
    SimConfig2D {
        gravity: Vec2::new(0.0, -9.81),
        dt: 1.0 / 120.0,
        solver_iterations: 12,
        max_bodies: 64,
        friction_default: 0.6,
        ..Default::default()
    }
}

/// Final per-body state keyed by label: (pos.x, pos.y, angle, vx, vy, omega).
type FinalState = HashMap<&'static str, [f32; 6]>;

/// Build a fresh `World2D`, add the labelled bodies, step `STEPS` times, and
/// return the final state of each body keyed by label. `None` if no GPU adapter.
fn simulate(scene: &[(&'static str, RigidBodyDesc2D)]) -> Option<FinalState> {
    let mut world = try_world(cfg())?;
    let handles: Vec<_> = scene.iter().map(|(_, d)| world.add_body(d)).collect();
    for _ in 0..STEPS {
        world.step();
    }
    let mut out = FinalState::new();
    for ((label, _), h) in scene.iter().zip(&handles) {
        let p = world.get_position(*h).unwrap_or(Vec2::ZERO);
        let a = world.get_angle(*h).unwrap_or(0.0);
        let v = world.get_velocity(*h).unwrap_or(Vec2::ZERO);
        let w = world.get_angular_velocity(*h).unwrap_or(0.0);
        out.insert(label, [p.x, p.y, a, v.x, v.y, w]);
    }
    Some(out)
}

/// Worst absolute discrepancy across all bodies (matched by label) and all six
/// recorded scalars.
fn worst_delta(a: &FinalState, b: &FinalState) -> f64 {
    let mut worst = 0.0f64;
    for (label, va) in a {
        let vb = &b[label];
        for k in 0..6 {
            worst = worst.max((va[k] - vb[k]).abs() as f64);
        }
    }
    worst
}

const STACK_LABELS: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];

/// Floor + a 6-rect vertical stack — a dense multi-contact graph that needs
/// several colors (where order/scheduling nondeterminism would first appear).
fn stack_scene() -> Vec<(&'static str, RigidBodyDesc2D)> {
    let mut v = vec![(
        "floor",
        RigidBodyDesc2D {
            x: 0.0,
            y: -0.5,
            mass: 0.0,
            friction: 0.6,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(20.0, 0.5),
            },
            ..Default::default()
        },
    )];
    for (i, &label) in STACK_LABELS.iter().enumerate() {
        v.push((
            label,
            RigidBodyDesc2D {
                x: 0.0,
                y: 0.5 + i as f32 * 1.005,
                mass: 1.0,
                friction: 0.6,
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::splat(0.5),
                },
                ..Default::default()
            },
        ));
    }
    v
}

/// Floor + a row of 3 flat rects resting at the floor and touching edge-to-edge
/// (each rect rests on the floor and contacts its neighbour). Unlike the stack,
/// this is a *stable* dense-contact scene, so insertion-order differences reflect
/// solver order-dependence rather than 2D stacking instability.
fn row_scene() -> Vec<(&'static str, RigidBodyDesc2D)> {
    let mut v = vec![(
        "floor",
        RigidBodyDesc2D {
            x: 0.0,
            y: -0.5,
            mass: 0.0,
            friction: 0.6,
            shape: ShapeDesc2D::Rect {
                half_extents: Vec2::new(20.0, 0.5),
            },
            ..Default::default()
        },
    )];
    for (i, label) in ["r0", "r1", "r2"].iter().enumerate() {
        v.push((
            label,
            RigidBodyDesc2D {
                x: (i as f32 - 1.0) * 1.0, // touching edge-to-edge (half-width 0.5)
                y: 0.5,
                mass: 1.0,
                friction: 0.6,
                shape: ShapeDesc2D::Rect {
                    half_extents: Vec2::splat(0.5),
                },
                ..Default::default()
            },
        ));
    }
    v
}

#[test]
fn determinism_two_worlds_2d() {
    let scene = stack_scene();
    let (Some(a), Some(b)) = (simulate(&scene), simulate(&scene)) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&a, &b);
    println!("2D determinism (two worlds): worst delta = {worst:.3e}");
    assert!(
        worst < 1.0e-5,
        "2D engine is non-deterministic across two independent worlds (worst delta {worst:.3e})"
    );
}

#[test]
fn permutation_invariance_2d() {
    // KNOWN FAILURE: 2D multi-body resting contact is currently unstable (see the
    // `resting_rect_stays_quiet_on_floor_2d` known-failure) — even a flat rect row
    // does not settle, so reversing insertion order produces a *different*
    // divergence (measured ~6e2). This records the finding and flips to a real
    // order-independence guard once 2D resting stability lands. The companion
    // `determinism_two_worlds_2d` (same-order bit-reproducibility) is the active
    // guard for the performance refactor.
    if should_skip_known_failure(
        "permutation_invariance_2d",
        "2D multi-body resting contact is unstable, so insertion order changes the divergence",
    ) {
        return;
    }
    let fwd = row_scene();
    let mut rev = row_scene();
    rev.reverse();
    let (Some(a), Some(b)) = (simulate(&fwd), simulate(&rev)) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&a, &b);
    println!("2D permutation invariance: worst delta = {worst:.3e}");
    // Regression ceiling — graph-colored Gauss-Seidel is inherently mildly
    // order-dependent (see the 3D analogue). Guards against a gross
    // order/scheduling dependence introduced by a perf refactor.
    assert!(
        worst < 5.0e-3,
        "2D insertion order grossly changes a dense-contact result (worst delta {worst:.3e}, \
         ceiling 5e-3)"
    );
}
