//! The **gap registry**: the structured record of scenarios that currently fail
//! the harness's derived-tolerance checks. This is the deliverable — an explicit,
//! categorized list of what the engine does *not* yet get right — and it doubles
//! as a quarantine so CI stays green on *known* gaps while still breaking on a
//! *new* regression (an unregistered failure) or a *fixed* gap (a registered
//! scenario that now passes), keeping the registry honest.
//!
//! A gap here is NOT "a test we gave up on": each entry records the measured
//! symptom and a hypothesis for where in the pipeline it lives.

use serde::{Deserialize, Serialize};

/// Where in the pipeline a gap most likely lives. Used to group the gap report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapCategory {
    /// Contact / constraint solve (AVBD primal/dual).
    Solver,
    /// Contact generation (SAT/GJK/EPA).
    Narrowphase,
    /// Tangential / Coulomb friction.
    Friction,
    /// Restitution / bounce.
    Restitution,
    /// Inertia tensor construction.
    Inertia,
    /// Rotational (quaternion) integration / angular-velocity extraction.
    AngularIntegration,
    /// Reproduces natively (lavapipe) — an algorithm/shader-logic bug.
    Native,
    /// Only reproduces in the browser (Chrome+SwiftShader): a toolchain (naga vs
    /// Tint), WASM-binding, async-path, or device-limit issue.
    BrowserOnly,
}

/// One catalogued gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnownGap {
    pub scenario: &'static str,
    pub category: GapCategory,
    /// The measured symptom and the where-it-lives hypothesis.
    pub reason: &'static str,
}

/// The current registry. Add an entry when the harness surfaces a real engine gap
/// (after ruling out an oracle/tolerance error); remove it when the gap is fixed.
pub fn known_gaps() -> &'static [KnownGap] {
    &[
        KnownGap {
            scenario: "zero_g_sphere_constant_spin",
            category: GapCategory::AngularIntegration,
            reason: "Angular velocity drifts ~3.3e-3 over 180 zero-torque steps, \
                     identically for a sphere and a box. Because a sphere is \
                     isotropic and cannot precess, the shape-independence shows \
                     this is a systematic quaternion integration/extraction bias, \
                     NOT gyroscopic free precession. It exceeds the random-walk \
                     f32 floor (~2e-4) by ~17x. (Refines the existing \
                     `zero_gravity_shapes_preserve_velocity_and_spin_3d` \
                     known-failure, which mis-attributes it to gyroscopics and \
                     incorrectly expects boxes to keep constant ω.)",
        },
        KnownGap {
            scenario: "static_friction_holds",
            category: GapCategory::Solver,
            reason: "A box resting on the floor sinks ~0.15m below the surface from \
                     the first tick under a tilted-gravity (combined normal + \
                     tangential) load. Straight-gravity resting \
                     (`resting_box_settles`) does NOT penetrate, so the contact \
                     solver under-resolves the normal constraint when a large \
                     tangential load is present. Needs the Parry depth oracle to \
                     confirm magnitude independently.",
        },
        KnownGap {
            scenario: "torque_free_box_angular_momentum",
            category: GapCategory::AngularIntegration,
            reason: "Angular momentum drifts ~14% of |L₀| over 180 zero-torque \
                     steps for a tumbling box, even though energy stays bounded. \
                     A torque-free body MUST conserve L exactly, so this is a \
                     real rotational-integration error — the conservation-law \
                     quantification of the same root cause as \
                     `zero_g_sphere_constant_spin`.",
        },
        KnownGap {
            scenario: "deep_overlap_separates",
            category: GapCategory::Solver,
            reason: "Two spheres started 0.6m deep in overlap separate explosively \
                     at ~70 m/s (vs a 15 m/s sanity bound). The engine has no \
                     penetration-recovery velocity clamp, so a large initial \
                     overlap converts directly into an unphysical velocity spike. \
                     Bodies spawned overlapping (a common case) will be launched.",
        },
    ]
}

/// Look up a registered gap by scenario name.
pub fn lookup(scenario: &str) -> Option<&'static KnownGap> {
    known_gaps().iter().find(|g| g.scenario == scenario)
}
