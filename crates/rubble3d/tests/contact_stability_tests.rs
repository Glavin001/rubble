//! Detectors for the M3 contact-stability failure: contact-identity churn
//! (SAT axis flip-flop + order-dependent clip-vertex feature ids) causes
//! warm-start misses that pump kinetic energy into resting piles.
//!
//! All tests here are gates. The pyramid tests were registered known failures
//! until the M3 contact-stability fix landed (SAT face-axis hysteresis +
//! line-pair clip-vertex feature ids in narrowphase box_box_test); with the
//! fix, pyramid key survival went from 0.67-0.74 to 0.97+ and the settle-tail
//! energy pump (52-192 J across runs) disappeared (tail decays to ~0.001 J).
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
use support::{add_tracked_body, collect_stability_frames, settle_stats, try_world, TrackedBody3D};

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

/// Dense, deep dropped grid — the "Grid Boxes" demo scene shape. At a single
/// step per frame this compacts so fast it tunnels into deep overlap and the
/// penalty solver explodes it (energy injection, boxes launched). With adaptive
/// substepping (as the demo scenes ship) it stays bounded and settles.
fn build_grid(nx: usize, ny: usize, nz: usize, max_substeps: u32) -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut world = try_world(SimConfig {
        gravity,
        max_bodies: nx * ny * nz + 4,
        ..Default::default()
    })?;
    world.set_adaptive_substeps(max_substeps);
    tracked_floor_and_grid(world, nx, ny, nz, gravity)
}

fn tracked_floor_and_grid(
    mut world: World,
    nx: usize,
    ny: usize,
    nz: usize,
    gravity: Vec3,
) -> Option<(World, Vec<TrackedBody3D>, Vec3)> {
    let mut tracked = Vec::new();
    tracked.push(add_tracked_body(
        &mut world,
        "floor",
        box_desc(0.0, 0.0, 0.0, 100.0, 1.0, 100.0, 0.0),
    ));
    let side = 0.42_f32;
    let pitch = side + 0.08;
    let half = side * 0.5;
    let ox = -((nx - 1) as f32 * pitch) * 0.5;
    let oz = -((nz - 1) as f32 * pitch) * 0.5;
    let base_y = 0.5 + half + 0.03;
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                let x = ox + i as f32 * pitch;
                let y = base_y + j as f32 * pitch;
                let z = oz + k as f32 * pitch;
                tracked.push(add_tracked_body(
                    &mut world,
                    "box",
                    box_desc(x, y, z, side, side, side, 1.0),
                ));
            }
        }
    }
    Some((world, tracked, gravity))
}

// ---------------------------------------------------------------------------
// Gates (pass today): stack manifold stability + energy decay
// ---------------------------------------------------------------------------

#[test]
fn grid_does_not_explode_with_adaptive_substeps_3d() {
    // Regression gate for the deep-grid explosion (see grid_explosion_diagnostic
    // for the full T1-T6 investigation). Root cause: box-box contacts carry no
    // speculative margin, so a deep stack compacting at several m/s tunnels into
    // ~0.2 m overlap (half a box) in one 1/60 s step; the penalty solver then
    // converts that overlap into an explosion (measured 7478 J of total-energy
    // INJECTION, boxes fired to 22 m/s / y=27 on a 6x24x6 grid at substeps=1).
    //
    // Substeps prevent the deep overlap by shrinking per-step travel. We ship it
    // as per-scene adaptive substepping (World::set_adaptive_substeps), NOT a
    // global default — substeps recalibrate the per-step solver and destabilise
    // other scenes (the offset pyramid regresses from 0.03 J to 500+ J of
    // injection under substeps; see grid_explosion_diagnostic::
    // pyramid_regression_probe). This gate uses the shipped adaptive config and
    // asserts the injection stays ~2 orders of magnitude below the broken level.
    let Some((mut world, tracked, gravity)) = build_grid(4, 20, 4, 8) else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let frames = collect_stability_frames(&mut world, &tracked, gravity, 300);
    let stats = settle_stats(&frames, 60);
    eprintln!("grid (adaptive) settle stats: {stats:?}");

    // The bug injects thousands of joules; adaptive holds it to tens. 200 J is
    // ~1% of this 320-box grid's ~5000 J budget: comfortably above the healthy
    // band, ~40x below the substeps=1 explosion.
    assert!(
        stats.max_total_energy_gain < 200.0,
        "grid is injecting energy (deep-overlap explosion): \
         max_total_energy_gain={:.1} J (max_ke={:.1} J, final_ke={:.1} J)",
        stats.max_total_energy_gain,
        stats.max_ke,
        stats.final_ke
    );
}

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
// Pyramid gates: identity churn + energy pump (fixed by M3; see module docs)
// ---------------------------------------------------------------------------

#[test]
fn pyramid_contacts_keep_identity_3d() {
    // Detects the root cause directly, independent of how much energy it
    // happens to inject: in a settled pile, contact keys must persist.
    // Pre-fix: mean_key_survival 0.68-0.74 (SAT axis flip-flop + clip-vertex
    // index churn rewrote feature ids every frame). Post-fix: 0.969-0.978
    // mean across 12 runs. The residual ~2.5% is genuine manifold-membership
    // fluctuation (contacts appearing/vanishing as flank boxes micro-rock;
    // new contacts count as misses here), so the threshold sits at 0.90 —
    // comfortably above the broken band, comfortably below the healthy one.
    let Some((mut world, tracked, gravity)) = build_pyramid() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    // 16-row pyramid dropped with ~0.35 m inter-layer gaps carries real
    // impact energy that legitimately takes a few hundred frames to dissipate;
    // measure identity only in the settled tail.
    let frames = collect_stability_frames(&mut world, &tracked, gravity, 600);
    let stats = settle_stats(&frames, 420);
    eprintln!("pyramid settle stats: {stats:?}");

    assert!(
        stats.mean_key_survival > 0.90,
        "settled pyramid is losing contact identities (warm-start misses): \
         mean_key_survival={:.3} min={:.3}",
        stats.mean_key_survival,
        stats.min_key_survival
    );
}

#[test]
fn pyramid_does_not_pump_kinetic_energy_3d() {
    // Detects the symptom: the solver must not ADD energy to a settled pile.
    // The gate is on total energy (KE + PE) rising above its running minimum:
    // legitimate metastable slips (a flank box sliding to a lower spot) show
    // KE spikes but total energy keeps falling; solver injection raises total
    // energy. Pre-fix the pump raised KE 52-192 J in the settle tail (total
    // gain of that order). Post-fix, across 10 calibration runs:
    // max_total_energy_gain 0.01-0.14 J — including a run with a 34 J KE slip
    // spike that gained only 0.14 J total, confirming slips are PE->KE
    // conversion, not injection. Threshold 5 J: ~35x above measured noise,
    // ~50x below the broken regime. final_ke is deliberately not asserted:
    // a slip near the window end can still be decaying at the last frame.
    let Some((mut world, tracked, gravity)) = build_pyramid() else {
        eprintln!("SKIP: No GPU adapter found");
        return;
    };

    let frames = collect_stability_frames(&mut world, &tracked, gravity, 600);
    let stats = settle_stats(&frames, 420);
    eprintln!("pyramid settle stats: {stats:?}");

    assert!(
        stats.max_total_energy_gain < 5.0,
        "solver injected energy into a settled pyramid: \
         max_total_energy_gain={:.2} J (max_ke={:.2} J, final_ke={:.2} J)",
        stats.max_total_energy_gain,
        stats.max_ke,
        stats.final_ke
    );
}
