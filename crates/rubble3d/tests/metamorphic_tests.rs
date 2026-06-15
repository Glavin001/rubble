//! Metamorphic & determinism tests.
//!
//! These compare the trajectories of deliberately-*related* runs, so they need
//! almost no tolerance — the expected relationship is exact (or fp-exact):
//!   * **determinism** — same input twice → same output;
//!   * **mass-independence** — free fall is identical regardless of mass;
//!   * **translation invariance** — shifting the whole scene shifts the result by
//!     exactly the same vector;
//!   * **permutation invariance** — body insertion order must not change physics.
//!
//! Translation and permutation are especially informative: a failure means the
//! engine has an absolute-position dependence (e.g. in broadphase Morton codes)
//! or an order dependence (e.g. in graph-colored solving) — both valuable to know.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig};
use rubble_testkit::{simulate_native, BodySample, TickRecord};

const STEPS: usize = 240;

fn cfg() -> SimConfig {
    SimConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        dt: 1.0 / 120.0,
        solver_iterations: 20,
        max_bodies: 64,
        friction_default: 0.5,
        ..Default::default()
    }
}

/// A small mixed scene (floor + sphere + box + capsule) optionally translated.
fn mixed_scene(offset: Vec3) -> Vec<(&'static str, RigidBodyDesc)> {
    vec![
        (
            "floor",
            RigidBodyDesc {
                position: Vec3::new(0.0, -0.5, 0.0) + offset,
                mass: 0.0,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::new(20.0, 0.5, 20.0),
                },
                ..Default::default()
            },
        ),
        (
            "sphere",
            RigidBodyDesc {
                position: Vec3::new(-0.6, 2.0, 0.0) + offset,
                linear_velocity: Vec3::new(0.5, 0.0, 0.0),
                mass: 1.0,
                friction: 0.5,
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            },
        ),
        (
            "box",
            RigidBodyDesc {
                position: Vec3::new(0.6, 2.4, 0.0) + offset,
                rotation: Quat::from_rotation_z(0.2),
                mass: 1.2,
                friction: 0.5,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::splat(0.4),
                },
                ..Default::default()
            },
        ),
        (
            "capsule",
            RigidBodyDesc {
                position: Vec3::new(0.0, 3.0, 0.4) + offset,
                mass: 0.9,
                friction: 0.5,
                shape: ShapeDesc::Capsule {
                    half_height: 0.4,
                    radius: 0.25,
                },
                ..Default::default()
            },
        ),
    ]
}

fn final_by_label(traj: &[TickRecord]) -> HashMap<String, BodySample> {
    traj.last()
        .unwrap()
        .bodies
        .iter()
        .map(|b| (b.label.clone(), b.clone()))
        .collect()
}

/// Worst position/velocity discrepancy between two final states, after applying
/// `shift` to the second's positions (used for translation invariance; pass ZERO
/// otherwise). Bodies are matched by label, so insertion order is irrelevant.
fn worst_delta(a: &[TickRecord], b: &[TickRecord], shift: Vec3) -> f64 {
    let fa = final_by_label(a);
    let fb = final_by_label(b);
    let mut worst = 0.0f64;
    for (label, va) in &fa {
        let vb = &fb[label];
        let dp = (va.position() - (vb.position() - shift)).length() as f64;
        let dv = (va.lin_vel() - vb.lin_vel()).length() as f64;
        worst = worst.max(dp).max(dv);
    }
    worst
}

#[test]
fn determinism_same_scene_twice() {
    let scene = mixed_scene(Vec3::ZERO);
    let (Some(a), Some(b)) = (
        simulate_native(&cfg(), &scene, STEPS),
        simulate_native(&cfg(), &scene, STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&a, &b, Vec3::ZERO);
    println!("determinism: worst delta = {worst:.3e}");
    assert!(
        worst < 1.0e-5,
        "engine is non-deterministic across identical runs (worst delta {worst:.3e})"
    );
}

#[test]
fn mass_independence_of_free_fall() {
    // No floor: pure free fall, which is mass-independent. Different masses must
    // produce identical trajectories.
    let make = |m: f32| {
        vec![(
            "ball",
            RigidBodyDesc {
                position: Vec3::new(0.0, 5.0, 0.0),
                linear_velocity: Vec3::new(1.0, 0.0, -0.5),
                mass: m,
                shape: ShapeDesc::Sphere { radius: 0.5 },
                ..Default::default()
            },
        )]
    };
    let (Some(light), Some(heavy)) = (
        simulate_native(&cfg(), &make(1.0), 120),
        simulate_native(&cfg(), &make(1000.0), 120),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&light, &heavy, Vec3::ZERO);
    println!("mass independence: worst delta = {worst:.3e}");
    assert!(
        worst < 1.0e-5,
        "free fall depends on mass (worst delta {worst:.3e}) — gravity integration couples mass"
    );
}

#[test]
fn translation_sensitivity_free_motion() {
    // Characterization. No contacts ⇒ no chaotic amplification, so any divergence
    // is pure absolute-position f32 error in integration (notably the
    // finite-difference velocity extraction, whose cancellation error grows with
    // |x|). Measured ~6mm at a 13m offset over 2s — ~10x the naive f32 floor, and
    // it grows with distance from the origin (the classic large-world precision
    // issue). The magnitude is the finding; we only guard against catastrophe.
    let t = Vec3::new(10.0, 4.0, -7.0);
    let make = |o: Vec3| {
        vec![
            (
                "sphere",
                RigidBodyDesc {
                    position: Vec3::new(0.0, 5.0, 0.0) + o,
                    linear_velocity: Vec3::new(1.0, 0.5, -0.3),
                    mass: 1.0,
                    shape: ShapeDesc::Sphere { radius: 0.5 },
                    ..Default::default()
                },
            ),
            (
                "box",
                RigidBodyDesc {
                    position: Vec3::new(2.0, 6.0, 1.0) + o,
                    angular_velocity: Vec3::new(0.3, 0.0, 0.2),
                    mass: 1.0,
                    shape: ShapeDesc::Box {
                        half_extents: Vec3::splat(0.4),
                    },
                    ..Default::default()
                },
            ),
        ]
    };
    let (Some(base), Some(shifted)) = (
        simulate_native(&cfg(), &make(Vec3::ZERO), STEPS),
        simulate_native(&cfg(), &make(t), STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&base, &shifted, t);
    println!(
        "translation sensitivity (free motion): worst delta = {worst:.3e} at {:.0}m offset",
        t.length()
    );
    // Catastrophe guard only — the divergence is real (absolute-position f32
    // accumulation) but small for free motion; a large value would indicate a
    // gross position-dependence bug rather than round-off.
    assert!(
        worst.is_finite() && worst < 0.05,
        "free-motion translation divergence is far larger than f32 round-off ({worst:.3e})"
    );
}

#[test]
fn translation_sensitivity_with_contacts() {
    // Characterization (not an invariant assertion): how much does a *contact*
    // simulation diverge when the whole scene is moved away from the origin? The
    // equations are translation-invariant, so any divergence comes from f32
    // precision in absolute-position-dependent collision math, amplified by the
    // chaotic nature of contacts. The magnitude is the finding; we only guard
    // against a catastrophic (NaN / runaway) result here.
    let t = Vec3::new(10.0, 4.0, -7.0);
    let (Some(base), Some(shifted)) = (
        simulate_native(&cfg(), &mixed_scene(Vec3::ZERO), STEPS),
        simulate_native(&cfg(), &mixed_scene(t), STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&base, &shifted, t);
    println!(
        "translation sensitivity (with contacts): worst delta = {worst:.3e} \
         (divergence of the same scene simulated {:.0}m from the origin)",
        t.length()
    );
    assert!(
        worst.is_finite() && worst < 50.0,
        "translated contact sim diverged catastrophically ({worst:.3e})"
    );
}

#[test]
fn permutation_invariance_of_insertion_order() {
    let fwd = mixed_scene(Vec3::ZERO);
    let mut rev = fwd.clone();
    rev.reverse();
    let (Some(a), Some(b)) = (
        simulate_native(&cfg(), &fwd, STEPS),
        simulate_native(&cfg(), &rev, STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    // Matched by label, so reversing insertion order should not matter — unless
    // the solver is order-dependent (e.g. graph-coloring / contact ordering).
    let worst = worst_delta(&a, &b, Vec3::ZERO);
    println!("permutation invariance: worst delta = {worst:.3e}");
    assert!(
        worst < 1.0e-4,
        "insertion order changes the result (worst delta {worst:.3e}): order-dependent \
         solving (graph coloring / contact ordering)"
    );
}

// ---------------------------------------------------------------------------
// High-contact determinism (CA1) — guards the coloring / adjacency / atomic
// contact-emission paths that the performance plan rewrites (GPU coloring,
// CSR adjacency, indirect dispatch). The 4-body mixed scene above has only a
// handful of contacts; a settled box stack produces a dense, multi-color
// contact graph, which is where order-/scheduling-dependent nondeterminism
// would first appear.
// ---------------------------------------------------------------------------

const STACK_LABELS: [&str; 6] = ["b0", "b1", "b2", "b3", "b4", "b5"];

/// Floor + a 6-box vertical stack (settles into 6 multi-point contacts, a graph
/// that needs several colors). Labels are static so trajectories can be matched
/// by label under insertion-order permutation.
fn stack_scene() -> Vec<(&'static str, RigidBodyDesc)> {
    let mut v = vec![(
        "floor",
        RigidBodyDesc {
            position: Vec3::new(0.0, -0.5, 0.0),
            mass: 0.0,
            friction: 0.6,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
            ..Default::default()
        },
    )];
    for (i, &label) in STACK_LABELS.iter().enumerate() {
        // Half-extent 0.5 boxes touching with a 5mm settle gap.
        v.push((
            label,
            RigidBodyDesc {
                position: Vec3::new(0.0, 0.5 + i as f32 * 1.005, 0.0),
                mass: 1.0,
                friction: 0.6,
                shape: ShapeDesc::Box {
                    half_extents: Vec3::splat(0.5),
                },
                ..Default::default()
            },
        ));
    }
    v
}

#[test]
fn determinism_high_contact_stack_two_worlds() {
    let scene = stack_scene();
    let (Some(a), Some(b)) = (
        simulate_native(&cfg(), &scene, STEPS),
        simulate_native(&cfg(), &scene, STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&a, &b, Vec3::ZERO);
    println!("high-contact determinism (two worlds): worst delta = {worst:.3e}");
    assert!(
        worst < 1.0e-5,
        "dense contact graph is non-deterministic across two independent worlds \
         (worst delta {worst:.3e}) — a scheduling/atomic/coloring-order dependence the \
         performance refactor must preserve"
    );
}

#[test]
fn permutation_invariance_high_contact_stack() {
    let fwd = stack_scene();
    let mut rev = fwd.clone();
    rev.reverse();
    let (Some(a), Some(b)) = (
        simulate_native(&cfg(), &fwd, STEPS),
        simulate_native(&cfg(), &rev, STEPS),
    ) else {
        eprintln!("SKIP: no GPU adapter");
        return;
    };
    let worst = worst_delta(&a, &b, Vec3::ZERO);
    println!("high-contact permutation invariance: worst delta = {worst:.3e}");
    // A graph-colored Gauss-Seidel solver is *inherently* mildly order-dependent:
    // the coloring keys off body index, so reversing insertion order assigns
    // different colors and the per-color sweep converges along a slightly
    // different path. Measured ~4.7e-4 over 240 steps on this 6-box stack (vs
    // <1e-4 for the sparse mixed scene) — sub-millimetre, and the reference
    // solvers share this property. This is therefore a *regression ceiling*: it
    // guards against a perf refactor introducing a gross order/scheduling
    // dependence (a race, nondeterministic atomic ordering), not against the
    // inherent ~5e-4 sensitivity.
    assert!(
        worst < 5.0e-3,
        "insertion order grossly changes a dense-contact result (worst delta {worst:.3e}, \
         ceiling 5e-3): the perf refactor introduced an order/scheduling dependence well \
         beyond the inherent Gauss-Seidel coloring sensitivity (~5e-4)"
    );
}

