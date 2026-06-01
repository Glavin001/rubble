//! Analytic ("closed-form") oracles. Every expected value is computed in code from
//! the integrator's *own* discrete scheme or from a conservation law — never a
//! hand-transcribed number — so a failure is unambiguous.

use glam::{DVec3, Vec3};
use rubble3d::SimConfig;

use crate::metrics::{TickRecord, F32_EPS};
use crate::report::Violation;

/// Position after `steps` semi-implicit (symplectic) Euler steps under constant
/// gravity, matching the engine's predict scheme `v += g·dt; x += v·dt`:
///   x_n = x0 + v0·n·dt + g·dt²·n(n+1)/2.
/// Compared against the *continuous* ½gt² this removes the integrator's own
/// discretization error from the tolerance budget.
pub fn discrete_ballistic_position(x0: Vec3, v0: Vec3, g: Vec3, dt: f32, steps: usize) -> Vec3 {
    let n = steps as f32;
    x0 + v0 * (dt * n) + g * (dt * dt * n * (n + 1.0) * 0.5)
}

/// Velocity after `steps` symplectic-Euler steps: v_n = v0 + g·n·dt.
pub fn discrete_ballistic_velocity(v0: Vec3, g: Vec3, dt: f32, steps: usize) -> Vec3 {
    v0 + g * (dt * steps as f32)
}

/// An endpoint / whole-trajectory oracle. Evaluated by the runner after the run
/// against the recorded trajectory (tick 0 = initial state).
#[derive(Debug, Clone)]
pub enum EndpointCheck {
    /// Contact-free motion must match the discrete ballistic recurrence from the
    /// body's own initial conditions.
    Ballistic {
        label: &'static str,
        pos_tol: f64,
        vel_tol: f64,
    },
    /// Torque-free body must preserve its initial angular velocity.
    ConstantSpin { label: &'static str, tol: f64 },
    /// With μ > tan(θ) (encoded via tilted gravity), the body must not slide:
    /// net displacement along the tangential (gravity-projected-onto-floor)
    /// direction must stay below `max_slide`.
    StaticFrictionNoSlide { label: &'static str, max_slide: f64 },
    /// Energy bound on a bounce: peak rebound height above the floor must not
    /// exceed e²·h (a passive collision cannot create a higher bounce).
    RestitutionMaxBounce {
        label: &'static str,
        e: f64,
        drop_height: f64,
        floor_y: f64,
        slack: f64,
    },
    /// After settling, the body must be at rest within bounds *derived* from the
    /// allowed penetration slop (a resting contact bobs within ~one slop depth).
    SettledAtRest {
        label: &'static str,
        after_frac: f64,
    },
    /// System angular momentum (world frame) must be conserved for a torque-free
    /// system. Integration-accuracy tier: bound is relative to |L₀|.
    AngularMomentumConserved { rel_tol: f64 },
    /// Body must settle to a known analytic resting height (e.g. floor +
    /// half-extent) within `tol` — a stronger oracle than "low jitter".
    RestHeight {
        label: &'static str,
        expected_y: f64,
        tol: f64,
    },
    /// Symmetry: a body's lateral (perpendicular-to-up) displacement from its
    /// initial position must stay below `max_drift` (a box dropped flat must not
    /// slide sideways).
    LateralDriftBounded { label: &'static str, max_drift: f64 },
    /// Two bodies that start overlapping must end up separated: final
    /// center-to-center distance ≥ `min_dist`.
    MinSeparation {
        a: &'static str,
        b: &'static str,
        min_dist: f64,
    },
    /// A body's final speed must lie within `[min, max]` (e.g. a struck body must
    /// move; a striker must have slowed — without assuming a restitution value).
    FinalSpeed {
        label: &'static str,
        min: f64,
        max: f64,
    },
}

fn body_index(traj: &[TickRecord], label: &str) -> Option<usize> {
    traj.first()?.bodies.iter().position(|b| b.label == label)
}

/// Evaluate one endpoint check against the full trajectory.
pub fn evaluate_endpoint(
    check: &EndpointCheck,
    cfg: &SimConfig,
    steps: usize,
    traj: &[TickRecord],
    out: &mut Vec<Violation>,
) {
    if traj.len() < 2 {
        return;
    }
    let last = traj.len() - 1;
    let g = cfg.gravity;
    let dt = cfg.dt;

    match check {
        EndpointCheck::Ballistic {
            label,
            pos_tol,
            vel_tol,
        } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let b0 = &traj[0].bodies[i];
            let bn = &traj[last].bodies[i];
            let exp_p = discrete_ballistic_position(b0.position(), b0.lin_vel(), g, dt, steps);
            let exp_v = discrete_ballistic_velocity(b0.lin_vel(), g, dt, steps);

            // Derived bounds (the passed-in values act as floors). Position
            // accumulates integration round-off ~ |x|·ε·√steps. Velocity is
            // *extracted* by finite-differencing f32 positions, v=(xₙ−xₙ₋₁)/dt,
            // which amplifies position round-off by ~1/dt (near-cancellation of
            // nearly-equal positions), so its floor scales with |x|·ε/dt.
            let pos_scale = traj
                .iter()
                .map(|r| r.bodies[i].position().length() as f64)
                .fold(1.0, f64::max);
            let pos_bound = pos_tol.max(pos_scale * F32_EPS * (steps as f64).sqrt() * 8.0);
            let vel_bound = vel_tol.max(pos_scale * F32_EPS / dt as f64 * 8.0);

            let dp = (bn.position() - exp_p).length() as f64;
            if dp > pos_bound {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "ballistic_position",
                    dp,
                    pos_bound,
                    format!("'{label}' drifted from discrete ballistic position (exp={exp_p:?}, got={:?})", bn.position()),
                ));
            }
            let dv = (bn.lin_vel() - exp_v).length() as f64;
            if dv > vel_bound {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "ballistic_velocity",
                    dv,
                    vel_bound,
                    format!("'{label}' drifted from discrete ballistic velocity (exp={exp_v:?}, got={:?})", bn.lin_vel()),
                ));
            }
        }

        EndpointCheck::ConstantSpin { label, tol } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let w0 = traj[0].bodies[i].ang_vel();
            let wn = traj[last].bodies[i].ang_vel();
            let d = (wn - w0).length() as f64;
            // NOTE: only valid for *isotropic* inertia (spheres) or spin about a
            // principal axis. For anisotropic bodies ω genuinely evolves under
            // torque-free motion (free precession), so this must not be applied to
            // boxes/capsules — use energy/angular-momentum conservation instead.
            // ω is finite-differenced from the quaternion, so the floor scales as
            // ε/dt.
            let bound = tol.max(F32_EPS / dt as f64 * 8.0);
            if d > bound {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "constant_spin",
                    d,
                    bound,
                    format!("'{label}' angular velocity changed under zero torque (isotropic body; w0={w0:?}, wn={wn:?})"),
                ));
            }
        }

        EndpointCheck::StaticFrictionNoSlide { label, max_slide } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            // Tangential direction = gravity projected onto the floor plane (up = -ĝ wrt floor normal +Y).
            let tangent = Vec3::new(g.x, 0.0, g.z);
            let p0 = traj[0].bodies[i].position();
            let pn = traj[last].bodies[i].position();
            let slide = if tangent.length_squared() > 1e-9 {
                ((pn - p0).dot(tangent.normalize())).abs() as f64
            } else {
                (pn - p0).length() as f64
            };
            if slide > *max_slide {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "static_friction_slip",
                    slide,
                    *max_slide,
                    format!("'{label}' slid although μ should hold it (static friction)"),
                ));
            }
        }

        EndpointCheck::RestitutionMaxBounce {
            label,
            e,
            drop_height,
            floor_y,
            slack,
        } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            // Peak height reached after the body first reaches its lowest point.
            let mut min_y = f32::INFINITY;
            let mut reached_bottom = false;
            let mut peak_after = f32::NEG_INFINITY;
            for r in traj {
                let y = r.bodies[i].position().y;
                if y < min_y {
                    min_y = y;
                }
                if !reached_bottom && r.bodies[i].lin_vel().y > 0.0 && y <= min_y + 0.01 {
                    reached_bottom = true;
                }
                if reached_bottom {
                    peak_after = peak_after.max(y);
                }
            }
            if reached_bottom {
                let rebound = (peak_after as f64) - floor_y;
                let bound = e * e * drop_height + slack;
                if rebound > bound {
                    out.push(Violation::new(
                        last,
                        Some(i),
                        "restitution_energy",
                        rebound,
                        bound,
                        format!(
                            "'{label}' bounced higher than e²·h allows (e={e}, h={drop_height})"
                        ),
                    ));
                }
            }
        }

        EndpointCheck::SettledAtRest { label, after_frac } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let start = ((*after_frac).clamp(0.0, 0.99) * steps as f64) as usize;
            // Derived rest bounds: a resting contact may bob within ~one penetration
            // slop, giving speed ≤ ~sqrt(2·g·slop) and height variation ≤ a few slop.
            let gmag = g.length() as f64;
            let slop = cfg.penetration_slop as f64;
            let speed_bound = 1.5 * (2.0 * gmag * slop).sqrt();
            let drift_bound = 4.0 * slop;

            let mut max_speed = 0.0f64;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for r in &traj[start..] {
                let b = &r.bodies[i];
                max_speed = max_speed.max(b.lin_vel().length() as f64);
                let y = b.position().y as f64;
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            if max_speed > speed_bound {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "rest_speed",
                    max_speed,
                    speed_bound,
                    format!("'{label}' still jittering after settling (bound ≈ 1.5·√(2g·slop))"),
                ));
            }
            let drift = max_y - min_y;
            if drift > drift_bound {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "rest_drift",
                    drift,
                    drift_bound,
                    format!("'{label}' height drift after settling exceeds 4·slop"),
                ));
            }
        }

        EndpointCheck::AngularMomentumConserved { rel_tol } => {
            let l0 = DVec3::from_array(traj[0].metrics.angular_momentum);
            let ln = DVec3::from_array(traj[last].metrics.angular_momentum);
            let d = (ln - l0).length();
            let bound = rel_tol * l0.length() + 1.0e-6;
            if d > bound {
                out.push(Violation::new(
                    last,
                    None,
                    "angular_momentum_drift",
                    d,
                    bound,
                    format!(
                        "angular momentum not conserved under zero torque (L0={l0:?}, Ln={ln:?})"
                    ),
                ));
            }
        }

        EndpointCheck::RestHeight {
            label,
            expected_y,
            tol,
        } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let y = traj[last].bodies[i].position().y as f64;
            let d = (y - expected_y).abs();
            if d > *tol {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "rest_height",
                    d,
                    *tol,
                    format!("'{label}' settled at y={y:.5}, expected {expected_y:.5}"),
                ));
            }
        }

        EndpointCheck::LateralDriftBounded { label, max_drift } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let up = if g.length_squared() > 1e-12 {
                -g.normalize()
            } else {
                Vec3::Y
            };
            let delta = traj[last].bodies[i].position() - traj[0].bodies[i].position();
            let lateral = (delta - up * delta.dot(up)).length() as f64;
            if lateral > *max_drift {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "lateral_drift",
                    lateral,
                    *max_drift,
                    format!("'{label}' drifted sideways {lateral:.5} (symmetry broken)"),
                ));
            }
        }

        EndpointCheck::MinSeparation { a, b, min_dist } => {
            let (Some(ia), Some(ib)) = (body_index(traj, a), body_index(traj, b)) else {
                return;
            };
            let dist = (traj[last].bodies[ia].position() - traj[last].bodies[ib].position())
                .length() as f64;
            if dist < *min_dist {
                out.push(Violation::new(
                    last,
                    None,
                    "min_separation",
                    dist,
                    *min_dist,
                    format!("'{a}' and '{b}' failed to separate (center dist {dist:.4})"),
                ));
            }
        }

        EndpointCheck::FinalSpeed { label, min, max } => {
            let Some(i) = body_index(traj, label) else {
                return;
            };
            let s = traj[last].bodies[i].lin_vel().length() as f64;
            if s < *min || s > *max {
                out.push(Violation::new(
                    last,
                    Some(i),
                    "final_speed",
                    s,
                    if s < *min { *min } else { *max },
                    format!("'{label}' final speed {s:.4} outside [{min:.4}, {max:.4}]"),
                ));
            }
        }
    }
}
