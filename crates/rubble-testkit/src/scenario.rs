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
