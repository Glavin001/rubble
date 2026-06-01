//! The scenario ladder: tiny, fast scenarios each chosen to isolate **one**
//! pipeline stage so a failure localizes the bug. Scenarios are plain data so the
//! exact same definition drives both the native and (future) wasm runners.

use glam::{Quat, Vec3};
use rubble3d::{RigidBodyDesc, ShapeDesc, SimConfig};

use crate::oracle::EndpointCheck;

/// One body plus a stable label used in oracles and reports.
pub struct BodyDef {
    pub label: &'static str,
    pub desc: RigidBodyDesc,
}

/// Which oracles / invariants apply to a scenario.
#[derive(Default)]
pub struct ScenarioChecks {
    /// Total energy must not increase (passive systems).
    pub energy_non_increase: bool,
    /// Linear momentum conserved (isolated systems only — no statics, no gravity).
    pub momentum_conserve: bool,
    /// Height of a floor plane (top surface) for the non-penetration invariant.
    pub floor_y: Option<f32>,
    /// Endpoint / whole-trajectory analytic oracles.
    pub endpoints: Vec<EndpointCheck>,
    /// Optional explicit explosion bounds (otherwise derived from initial state).
    pub v_escape: Option<f64>,
    pub omega_escape: Option<f64>,
}

/// A complete, self-contained scenario.
pub struct Scenario {
    pub name: &'static str,
    pub config: SimConfig,
    pub steps: usize,
    pub bodies: Vec<BodyDef>,
    pub checks: ScenarioChecks,
}

// --- builders ---------------------------------------------------------------

fn cfg(gravity: Vec3, dt: f32, solver_iterations: u32, friction_default: f32) -> SimConfig {
    SimConfig {
        gravity,
        dt,
        solver_iterations,
        max_bodies: 64,
        friction_default,
        ..Default::default()
    }
}

fn sphere(label: &'static str, p: Vec3, v: Vec3, r: f32, m: f32, friction: f32) -> BodyDef {
    BodyDef {
        label,
        desc: RigidBodyDesc {
            position: p,
            linear_velocity: v,
            mass: m,
            friction,
            shape: ShapeDesc::Sphere { radius: r },
            ..Default::default()
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn boxd(
    label: &'static str,
    p: Vec3,
    v: Vec3,
    w: Vec3,
    rot: Quat,
    he: Vec3,
    m: f32,
    friction: f32,
) -> BodyDef {
    BodyDef {
        label,
        desc: RigidBodyDesc {
            position: p,
            rotation: rot,
            linear_velocity: v,
            angular_velocity: w,
            mass: m,
            friction,
            shape: ShapeDesc::Box { half_extents: he },
        },
    }
}

/// Axis-aligned cube as a convex hull (8 corners), for exercising the convex path.
fn cube_hull(h: f32) -> ShapeDesc {
    ShapeDesc::ConvexHull {
        vertices: vec![
            Vec3::new(-h, -h, -h),
            Vec3::new(h, -h, -h),
            Vec3::new(h, h, -h),
            Vec3::new(-h, h, -h),
            Vec3::new(-h, -h, h),
            Vec3::new(h, -h, h),
            Vec3::new(h, h, h),
            Vec3::new(-h, h, h),
        ],
    }
}

/// Static box floor whose top surface sits at `top_y`.
fn floor(top_y: f32, friction: f32) -> BodyDef {
    BodyDef {
        label: "floor",
        desc: RigidBodyDesc {
            position: Vec3::new(0.0, top_y - 0.5, 0.0),
            mass: 0.0,
            friction,
            shape: ShapeDesc::Box {
                half_extents: Vec3::new(20.0, 0.5, 20.0),
            },
            ..Default::default()
        },
    }
}

/// The full ladder. Keep scenarios tiny (≤ a few bodies) and short (just enough
/// ticks to verify the property) so both lanes stay fast.
pub fn scenarios() -> Vec<Scenario> {
    let dt = 1.0 / 120.0;
    let g = Vec3::new(0.0, -9.81, 0.0);

    vec![
        // 1. Integration only — free fall. Isolates predict/extract.
        Scenario {
            name: "free_fall_sphere",
            config: cfg(g, dt, 8, 0.5),
            steps: 90,
            bodies: vec![sphere(
                "ball",
                Vec3::new(0.0, 5.0, 0.0),
                Vec3::ZERO,
                0.5,
                1.0,
                0.5,
            )],
            checks: ScenarioChecks {
                endpoints: vec![EndpointCheck::Ballistic {
                    label: "ball",
                    pos_tol: 5.0e-4,
                    vel_tol: 5.0e-4,
                }],
                ..Default::default()
            },
        },
        // 2. Integration with skew gravity + initial velocity. Isolates predict.
        Scenario {
            name: "projectile_box",
            config: cfg(Vec3::new(0.35, -9.81, 0.6), dt, 8, 0.3),
            steps: 90,
            bodies: vec![boxd(
                "proj",
                Vec3::new(-1.25, 4.5, 0.8),
                Vec3::new(1.5, -0.25, 0.75),
                Vec3::ZERO,
                Quat::from_rotation_y(0.35) * Quat::from_rotation_x(-0.2),
                Vec3::new(0.5, 0.4, 0.6),
                2.0,
                0.3,
            )],
            checks: ScenarioChecks {
                endpoints: vec![EndpointCheck::Ballistic {
                    label: "proj",
                    pos_tol: 5.0e-4,
                    vel_tol: 5.0e-4,
                }],
                ..Default::default()
            },
        },
        // 3a. Zero-g sphere: isotropic inertia ⇒ ω is genuinely constant. Isolates
        //     angular integration with a *valid* constant-spin oracle.
        Scenario {
            name: "zero_g_sphere_constant_spin",
            config: cfg(Vec3::ZERO, dt, 8, 0.0),
            steps: 180,
            bodies: vec![BodyDef {
                label: "spinner",
                desc: RigidBodyDesc {
                    position: Vec3::new(1.5, -0.25, 2.0),
                    rotation: Quat::from_rotation_z(0.25) * Quat::from_rotation_y(-0.35),
                    linear_velocity: Vec3::new(1.1, -0.7, 0.35),
                    angular_velocity: Vec3::new(0.4, -0.25, 0.6),
                    mass: 1.5,
                    friction: 0.0,
                    shape: ShapeDesc::Sphere { radius: 0.5 },
                },
            }],
            checks: ScenarioChecks {
                energy_non_increase: true,
                endpoints: vec![
                    EndpointCheck::Ballistic {
                        label: "spinner",
                        pos_tol: 3.0e-4,
                        vel_tol: 3.0e-4,
                    },
                    EndpointCheck::ConstantSpin {
                        label: "spinner",
                        tol: 6.0e-4,
                    },
                ],
                ..Default::default()
            },
        },
        // 3b. Zero-g tumbling box: anisotropic inertia ⇒ ω evolves (free precession,
        //     the tennis-racket theorem), so ω is *not* expected to be constant.
        //     The valid invariant is energy conservation (torque-free KE is
        //     constant) plus translational ballistics.
        Scenario {
            name: "zero_g_box_energy_conserved",
            config: cfg(Vec3::ZERO, dt, 8, 0.0),
            steps: 180,
            bodies: vec![boxd(
                "tumbler",
                Vec3::new(1.5, -0.25, 2.0),
                Vec3::new(1.1, -0.7, 0.35),
                Vec3::new(0.4, -0.25, 0.6),
                Quat::from_rotation_z(0.25) * Quat::from_rotation_y(-0.35),
                Vec3::new(0.5, 0.35, 0.45),
                1.5,
                0.0,
            )],
            checks: ScenarioChecks {
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::Ballistic {
                    label: "tumbler",
                    pos_tol: 3.0e-4,
                    vel_tol: 3.0e-4,
                }],
                ..Default::default()
            },
        },
        // 3c. Torque-free anisotropic body: angular momentum MUST be conserved (ω
        //     evolving is fine). Quantifies the rotational-integration gap via a
        //     conservation law — pure rotation, no translation, to isolate L.
        Scenario {
            name: "torque_free_box_angular_momentum",
            config: cfg(Vec3::ZERO, dt, 8, 0.0),
            steps: 180,
            bodies: vec![boxd(
                "tumbler",
                Vec3::ZERO,
                Vec3::ZERO,
                Vec3::new(0.4, -0.25, 0.6),
                Quat::from_rotation_z(0.25) * Quat::from_rotation_y(-0.35),
                Vec3::new(0.5, 0.35, 0.45),
                1.5,
                0.0,
            )],
            checks: ScenarioChecks {
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::AngularMomentumConserved { rel_tol: 0.02 }],
                ..Default::default()
            },
        },
        // 4. Two-body elastic-ish collision, isolated (no gravity, no statics).
        //    Isolates pairwise contact solve via momentum conservation.
        Scenario {
            name: "two_body_head_on",
            config: cfg(Vec3::ZERO, dt, 24, 0.0),
            steps: 120,
            bodies: vec![
                sphere(
                    "a",
                    Vec3::new(-1.5, 0.0, 0.0),
                    Vec3::new(3.0, 0.0, 0.0),
                    0.5,
                    1.0,
                    0.0,
                ),
                sphere(
                    "b",
                    Vec3::new(1.5, 0.0, 0.0),
                    Vec3::new(-3.0, 0.0, 0.0),
                    0.5,
                    1.0,
                    0.0,
                ),
            ],
            checks: ScenarioChecks {
                momentum_conserve: true,
                energy_non_increase: true,
                ..Default::default()
            },
        },
        // 5. Restitution / energy bound on a bounce. Isolates normal-impulse solve.
        //    e=1 makes this assumption-free: a passive bounce can never exceed the
        //    drop height (super-elastic rebound = energy creation = bug).
        Scenario {
            name: "drop_bounce_energy_bound",
            config: cfg(g, dt, 20, 0.3),
            steps: 180,
            bodies: vec![
                floor(0.0, 0.3),
                sphere("ball", Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO, 0.5, 1.0, 0.3),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::RestitutionMaxBounce {
                    label: "ball",
                    e: 1.0,
                    drop_height: 1.5, // center 2.0 -> rest center 0.5
                    floor_y: 0.5,
                    slack: 0.05,
                }],
                ..Default::default()
            },
        },
        // 6. Resting box must settle to rest with *derived* (slop-based) jitter
        //    bounds. Directly challenges the previously fitted resting tolerances.
        Scenario {
            name: "resting_box_settles",
            config: cfg(g, dt, 24, 0.8),
            steps: 360,
            bodies: vec![
                floor(0.0, 0.8),
                boxd(
                    "box",
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.8,
                ),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::SettledAtRest {
                    label: "box",
                    after_frac: 0.5,
                }],
                ..Default::default()
            },
        },
        // 7. Static friction holds (tilted gravity emulates an incline; μ>tanθ).
        //    Isolates the tangential solver with an unambiguous "must not slide".
        Scenario {
            name: "static_friction_holds",
            config: cfg(Vec3::new(3.0, -9.81, 0.0), dt, 24, 0.8), // tanθ≈0.31 < μ=0.8
            steps: 240,
            bodies: vec![
                floor(0.0, 0.8),
                boxd(
                    "slider",
                    Vec3::new(0.0, 0.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.8,
                ),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                endpoints: vec![EndpointCheck::StaticFrictionNoSlide {
                    label: "slider",
                    max_slide: 0.05,
                }],
                ..Default::default()
            },
        },
        // 7b. Gentle incline (small tangential load): isolates the *tangential
        //     hold* property of friction. Non-penetration is intentionally NOT
        //     checked here — the box still sinks ~0.026m (gap #3 is load-
        //     proportional and covered by `static_friction_holds`); this scenario
        //     verifies friction prevents sliding, and serves as the *passing*
        //     friction case the fault matrix needs to detect ZeroFriction.
        Scenario {
            name: "gentle_incline_friction_holds",
            config: cfg(Vec3::new(0.5, -9.81, 0.0), dt, 24, 0.5), // tanθ≈0.051 ≪ μ=0.5
            steps: 240,
            bodies: vec![
                floor(0.0, 0.5),
                boxd(
                    "slider",
                    Vec3::new(0.0, 0.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.5,
                ),
            ],
            checks: ScenarioChecks {
                endpoints: vec![EndpointCheck::StaticFrictionNoSlide {
                    label: "slider",
                    max_slide: 0.05,
                }],
                ..Default::default()
            },
        },
        // 8. Small stack — stresses warm-start + graph-coloring cross-frame state
        //    (the transient-glitch risk). Invariant-only: no NaN, no penetration,
        //    energy non-increase, and the top box must settle.
        Scenario {
            name: "stack_two_boxes",
            config: cfg(g, dt, 24, 0.6),
            steps: 300,
            bodies: vec![
                floor(0.0, 0.6),
                boxd(
                    "lower",
                    Vec3::new(0.0, 0.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
                boxd(
                    "upper",
                    Vec3::new(0.0, 1.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::SettledAtRest {
                    label: "upper",
                    after_frac: 0.6,
                }],
                ..Default::default()
            },
        },
        // 9. Newton's-cradle: a moving sphere strikes a resting equal-mass sphere.
        //    Momentum + energy + "the struck body must move" (no assumption about
        //    the restitution value), and the striker must not speed up.
        Scenario {
            name: "newtons_cradle",
            config: cfg(Vec3::ZERO, 1.0 / 240.0, 24, 0.0),
            steps: 220,
            bodies: vec![
                sphere(
                    "striker",
                    Vec3::new(-1.5, 0.0, 0.0),
                    Vec3::new(3.0, 0.0, 0.0),
                    0.5,
                    1.0,
                    0.0,
                ),
                sphere(
                    "target",
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::ZERO,
                    0.5,
                    1.0,
                    0.0,
                ),
            ],
            checks: ScenarioChecks {
                momentum_conserve: true,
                energy_non_increase: true,
                endpoints: vec![
                    EndpointCheck::FinalSpeed {
                        label: "target",
                        min: 0.5,
                        max: f64::INFINITY,
                    },
                    EndpointCheck::FinalSpeed {
                        label: "striker",
                        min: 0.0,
                        max: 3.05,
                    },
                ],
                ..Default::default()
            },
        },
        // 10. Unequal-mass collision (5:1). Momentum conservation across a mass
        //     ratio; the light body must be kicked forward.
        Scenario {
            name: "unequal_mass_collision",
            config: cfg(Vec3::ZERO, 1.0 / 240.0, 24, 0.0),
            steps: 220,
            bodies: vec![
                sphere(
                    "heavy",
                    Vec3::new(-1.5, 0.0, 0.0),
                    Vec3::new(2.0, 0.0, 0.0),
                    0.5,
                    5.0,
                    0.0,
                ),
                sphere("light", Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO, 0.5, 1.0, 0.0),
            ],
            checks: ScenarioChecks {
                momentum_conserve: true,
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::FinalSpeed {
                    label: "light",
                    min: 1.0,
                    max: f64::INFINITY,
                }],
                ..Default::default()
            },
        },
        // 11. Resting equilibrium: a box dropped flat must settle at the exact
        //     analytic height (floor + half-extent) and not drift sideways.
        Scenario {
            name: "box_rests_at_correct_height",
            config: cfg(g, dt, 24, 0.6),
            steps: 300,
            bodies: vec![
                floor(0.0, 0.6),
                boxd(
                    "box",
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![
                    EndpointCheck::RestHeight {
                        label: "box",
                        expected_y: 0.5,
                        tol: 0.03, // ≤ slop + contact_offset band
                    },
                    EndpointCheck::LateralDriftBounded {
                        label: "box",
                        max_drift: 0.02,
                    },
                ],
                ..Default::default()
            },
        },
        // 12. Taller stack (4 boxes) — heavier stress on warm-start + coloring.
        //     Invariant-only plus "top box settles".
        Scenario {
            name: "stack_four_boxes",
            config: cfg(g, dt, 24, 0.6),
            steps: 360,
            bodies: vec![
                floor(0.0, 0.6),
                boxd(
                    "b1",
                    Vec3::new(0.0, 0.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
                boxd(
                    "b2",
                    Vec3::new(0.0, 1.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
                boxd(
                    "b3",
                    Vec3::new(0.0, 2.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
                boxd(
                    "b4",
                    Vec3::new(0.0, 3.5, 0.0),
                    Vec3::ZERO,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                    1.0,
                    0.6,
                ),
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::SettledAtRest {
                    label: "b4",
                    after_frac: 0.7,
                }],
                ..Default::default()
            },
        },
        // 13. Deep-penetration recovery: two spheres start heavily overlapping and
        //     must push apart (separate) without exploding or NaN-ing. Energy is
        //     NOT checked (separating from overlap legitimately releases energy).
        Scenario {
            name: "deep_overlap_separates",
            config: cfg(Vec3::ZERO, 1.0 / 240.0, 24, 0.0),
            steps: 160,
            bodies: vec![
                sphere("left", Vec3::new(-0.2, 0.0, 0.0), Vec3::ZERO, 0.5, 1.0, 0.0),
                sphere("right", Vec3::new(0.2, 0.0, 0.0), Vec3::ZERO, 0.5, 1.0, 0.0),
            ],
            checks: ScenarioChecks {
                v_escape: Some(15.0), // separating 0.6m of overlap should not fling them
                endpoints: vec![EndpointCheck::MinSeparation {
                    a: "left",
                    b: "right",
                    min_dist: 0.95, // sum of radii = 1.0; ~separated
                }],
                ..Default::default()
            },
        },
        // 14. Convex hull (cube) resting on the floor. convex↔box narrowphase is
        //     exact, so a convex cube should rest like a box at floor+half_extent.
        Scenario {
            name: "convex_cube_rests",
            config: cfg(g, dt, 24, 0.6),
            steps: 300,
            bodies: vec![
                floor(0.0, 0.6),
                BodyDef {
                    label: "hull",
                    desc: RigidBodyDesc {
                        position: Vec3::new(0.0, 1.0, 0.0),
                        mass: 1.0,
                        friction: 0.6,
                        shape: cube_hull(0.5),
                        ..Default::default()
                    },
                },
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                energy_non_increase: true,
                endpoints: vec![EndpointCheck::RestHeight {
                    label: "hull",
                    expected_y: 0.5,
                    tol: 0.05,
                }],
                ..Default::default()
            },
        },
        // 15. Compound body (two boxes) dropped on the floor. Compound is supported
        //     via CPU pair expansion but is a known faller; RestHeight catches a
        //     fall-through (final center far below the resting height).
        Scenario {
            name: "compound_box_rests",
            config: cfg(g, dt, 24, 0.6),
            steps: 240,
            bodies: vec![
                floor(0.0, 0.6),
                BodyDef {
                    label: "compound",
                    desc: RigidBodyDesc {
                        position: Vec3::new(0.0, 2.0, 0.0),
                        mass: 2.0,
                        friction: 0.6,
                        shape: ShapeDesc::Compound {
                            children: vec![
                                (
                                    ShapeDesc::Box {
                                        half_extents: Vec3::new(0.5, 0.25, 0.5),
                                    },
                                    Vec3::new(0.0, -0.25, 0.0),
                                    Quat::IDENTITY,
                                ),
                                (
                                    ShapeDesc::Box {
                                        half_extents: Vec3::splat(0.25),
                                    },
                                    Vec3::new(0.0, 0.25, 0.0),
                                    Quat::IDENTITY,
                                ),
                            ],
                        },
                        ..Default::default()
                    },
                },
            ],
            checks: ScenarioChecks {
                floor_y: Some(0.0),
                // Lowest child point is 0.5 below center ⇒ rest center at floor+0.5.
                endpoints: vec![EndpointCheck::RestHeight {
                    label: "compound",
                    expected_y: 0.5,
                    tol: 0.1,
                }],
                ..Default::default()
            },
        },
    ]
}

/// Look up a scenario by name (used by the wasm entry point and targeted runs).
pub fn scenario_by_name(name: &str) -> Option<Scenario> {
    scenarios().into_iter().find(|s| s.name == name)
}

/// All scenario names, for enumerating lanes.
pub fn scenario_names() -> Vec<&'static str> {
    scenarios().iter().map(|s| s.name).collect()
}
