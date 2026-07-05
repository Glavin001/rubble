//! Detectors for the M3 contact-stability failure: contact-identity churn
//! (SAT axis flip-flop + order-dependent clip-vertex feature ids) causes
//! warm-start misses that pump kinetic energy into resting piles.
//!
//! Two scene tiers:
//!   - The 10-box stack settles correctly today: those tests GATE, protecting
//!     against regressions in manifold stability.
//!   - The 16-row pyramid exhibits the failure today: those tests are
//!     registered known failures (owned by M3) with tight thresholds, so they
//!     flip to gates the moment M3 lands. Run with RUBBLE_RUN_KNOWN_FAILURES=1.
//!
//! Metric choice: warm-start key survival (fraction of contacts whose
//! `(pair, feature_id)` key existed the previous frame) and settle-window
//! kinetic energy. Both are robust to the engine's run-to-run scheduling
//! nondeterminism (a few percent) while the failure moves them by an order of
//! magnitude (survival 0.67 vs ~1.0; tail KE 200+ J vs <1 J). See
//! pyramid_energy_diagnostic.rs for the full per-frame instrumentation.

mod support;

use glam::Vec3;
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig, World};
use support::{
    add_tracked_body, collect_stability_frames, settle_stats, should_skip_known_failure,
    try_world, TrackedBody3D,
};

#[allow(clippy::too_many_arguments)]
fn box_desc(x: f32, y: f32, z: f32, w: f32, h: f32, d: f32, mass: f32) -> RigidBodyDesc {
    RigidBodyDesc {
        position: Vec3::new(x, y, z),
        mass,
        friction: 0.5,
        shape: ShapeDesc::Box {
            half_extents: Vec3::new(w * 0.5, h * 0.5, d * 0.5),
        },
        ..Default::default()
    }
}

/// Same 10-box stack as demo_repro_tests (a scene that settles today).
fn build_stack() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        max_bodies: 32,
        ..Default::default()
    })?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0),
    ));
    for i in 0..10 {
        tracked.push(add_tracked_body(
            &mut world,
            "box",
            box_desc(0.0, i as f32 * 1.5 + 1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        ));
    }
    Some((world, tracked, gravity))
}

/// Same 16-row pyramid as demo_repro_tests (the visible energy-pump repro).
fn build_pyramid() -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    const SIZE: usize = 16;
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        max_bodies: 256,
        ..Default::default()
    })?;
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, -0.5, 0.0, 100.0, 1.0, 100.0, 0.0),
    ));
    for y in 0..SIZE {
        for x in 0..(SIZE - y) {
            tracked.push(add_tracked_body(
                &mut world,
                "box",
                box_desc(
                    x as f32 * 1.01 + y as f32 * 0.5 - SIZE as f32 / 2.0,
                    y as f32 * 0.85 + 0.5,
                    0.0,
                    1.0,
                    0.5,
                    0.5,
                    1.0,
                ),
            ));
        }
    }
    Some((world, tracked, gravity))
}

// ---------------------------------------------------------------------------
// Gates (pass today): stack manifold stability + energy decay
// ---------------------------------------------------------------------------

#[test]
fn stack_contacts_keep_identity_and_energy_decays_3d() {
    let Some((mut world, tracked, gravity)) = build_stack() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let frames = collect_stability_frames(&mut world, &tracked, gravity, 480);
    let stats = settle_stats(&frames, 240);
    eprintln!("stack settle stats: {stats:?}");

    assert!(
        stats.mean_key_survival > 0.90,
        "settled stack is losing contact identities (warm-start misses): \
         mean_key_survival={:.3} min={:.3}",
        stats.mean_key_survival,
        stats.min_key_survival
    );
    assert!(
        stats.max_ke < 1.0,
        "settled stack shows kinetic energy where it should be at rest: \
         max_ke={:.3} J final_ke={:.3} J",
        stats.max_ke,
        stats.final_ke
    );
    assert!(
        stats.final_ke < 0.05,
        "stack failed to damp to near-zero kinetic energy: final_ke={:.4} J",
        stats.final_ke
    );
}

// ---------------------------------------------------------------------------
// Known failures (flip to gates with M3): pyramid identity churn + energy pump
// ---------------------------------------------------------------------------

#[test]
fn pyramid_contacts_keep_identity_3d() {
    // Detects the root cause directly, independent of how much energy it
    // happens to inject: in a settled pile, contact keys must persist.
    // Measured today: mean_key_survival 0.68-0.74 across runs (threshold
    // 0.95) — SAT axis flip-flop and clip-vertex index churn rewrite feature
    // ids every frame. This is the most stable of the detectors.
    if should_skip_known_failure(
        "pyramid_contacts_keep_identity_3d",
        "box-box feature ids churn (~33%/frame); owned by M3 (SAT hysteresis + stable feature ids)",
    ) {
        return;
    }
    let Some((mut world, tracked, gravity)) = build_pyramid() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let frames = collect_stability_frames(&mut world, &tracked, gravity, 360);
    let stats = settle_stats(&frames, 240);
    eprintln!("pyramid settle stats: {stats:?}");

    assert!(
        stats.mean_key_survival > 0.95,
        "settled pyramid is losing contact identities (warm-start misses): \
         mean_key_survival={:.3} min={:.3}",
        stats.mean_key_survival,
        stats.min_key_survival
    );
}

#[test]
fn pyramid_does_not_pump_kinetic_energy_3d() {
    // Detects the symptom: a resting pyramid must dissipate, not gain, kinetic
    // energy. Measured today: max_ke in the settle window was 52/107/192 J
    // across three runs (threshold 5) — noisy in magnitude but always an order
    // above healthy (<1 J). max_ke is the primary assert; final_ke alone is
    // NOT reliable (one run damped to 0.12 J after pumping to 192 J).
    if should_skip_known_failure(
        "pyramid_does_not_pump_kinetic_energy_3d",
        "warm-start misses pump KE to 200+ J in the settle window; owned by M3",
    ) {
        return;
    }
    let Some((mut world, tracked, gravity)) = build_pyramid() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let frames = collect_stability_frames(&mut world, &tracked, gravity, 360);
    let stats = settle_stats(&frames, 240);
    eprintln!("pyramid settle stats: {stats:?}");

    assert!(
        stats.max_ke < 5.0,
        "settled pyramid gained kinetic energy instead of dissipating: \
         max_ke={:.2} J final_ke={:.2} J",
        stats.max_ke,
        stats.final_ke
    );
    assert!(
        stats.final_ke < 0.5,
        "pyramid failed to damp toward rest: final_ke={:.3} J",
        stats.final_ke
    );
}
