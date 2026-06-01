//! Per-tick invariant checks, run on **every** recorded tick (not just endpoints)
//! so transient mid-simulation glitches are caught and pinpointed.
//!
//! Every tolerance here is *derived* — from f32 round-off accumulation or from a
//! config-defined physical bound (`penetration_slop`, `contact_offset`) — never
//! fitted to current behavior. Bounds split into two tiers:
//!   * **tight, derived** (quaternion norm, static-frozen, non-penetration,
//!     energy non-increase, momentum) — a violation is a real bug or precision
//!     regression; and
//!   * **loose, sanity** (explosion / teleport guards) — physically obvious
//!     catastrophe detectors that should never trip in correct behavior.

use glam::{DVec3, Quat, Vec3};
use rubble3d::ShapeDesc;

use crate::metrics::{TickRecord, F32_EPS};
use crate::report::Violation;

/// Toggles + derived bounds for the invariant pass of a single scenario. Built by
/// the runner from the scenario config and the recorded initial (tick-0) state.
pub struct InvariantSpec {
    pub dt: f32,
    pub gravity: Vec3,

    // --- universal (always checked) ---
    /// |‖q‖−1| bound. Derived: one renorm round-trips through f32 with < ~6 ulp
    /// error; 1e-5 is a ~10x margin on that.
    pub quat_norm_tol: f64,
    /// Loose explosion guard on linear speed (sanity tier).
    pub v_escape: f64,
    /// Loose explosion guard on angular speed (sanity tier).
    pub omega_escape: f64,
    /// Extra slack added to the no-teleport bound (`v_max·dt + this`).
    pub teleport_extra: f64,

    // --- static-frozen (bodies that must never move) ---
    pub static_indices: Vec<usize>,
    pub static_pos0: Vec<Vec3>,
    /// Static bodies are uploaded once and never integrated; only f32 noise is
    /// allowed.
    pub static_tol: f64,

    // --- floor non-penetration (opt-in: scenarios with a known floor plane) ---
    pub floor_y: Option<f32>,
    /// slop + contact_offset.
    pub penetration_bound: f64,
    pub shapes: Vec<ShapeDesc>,

    // --- energy non-increase (opt-in: passive systems) ---
    pub energy_non_increase: bool,
    pub baseline_energy: f64,
    /// Relative tolerance, derived ≈ 8·N_iter·eps_f32.
    pub energy_rel: f64,
    /// Absolute floor for catastrophic cancellation in PE, derived ≈ 1e-3·m·g·L.
    pub energy_abs: f64,

    // --- momentum conservation (opt-in: isolated systems) ---
    pub momentum_conserve: bool,
    pub baseline_momentum: DVec3,
    pub momentum_tol: f64,
}

/// Coordinate of the lowest surface point of `shape` along `up` (i.e. the minimum
/// of `point · up` over the shape). Returns `None` for shapes we don't model here.
fn lowest_along(shape: &ShapeDesc, pos: Vec3, rot: Quat, up: Vec3) -> Option<f32> {
    let c = pos.dot(up);
    let ext = match shape {
        ShapeDesc::Sphere { radius } => *radius,
        ShapeDesc::Box { half_extents } => {
            // Support extent of an oriented box along -up.
            let d_local = rot.inverse() * (-up);
            d_local.x.abs() * half_extents.x
                + d_local.y.abs() * half_extents.y
                + d_local.z.abs() * half_extents.z
        }
        ShapeDesc::Capsule {
            half_height,
            radius,
        } => {
            let axis = rot * Vec3::Y; // capsule axis is local +Y
            (axis * *half_height).dot(up).abs() + *radius
        }
        ShapeDesc::ConvexHull { vertices } => {
            let mut min_proj = f32::INFINITY;
            for v in vertices {
                min_proj = min_proj.min((rot * *v).dot(up));
            }
            return Some(c + min_proj);
        }
        _ => return None,
    };
    Some(c - ext)
}

/// Run all enabled invariants for tick `cur` (with `prev` for differential checks).
pub fn check_tick(
    spec: &InvariantSpec,
    prev: Option<&TickRecord>,
    cur: &TickRecord,
    out: &mut Vec<Violation>,
) {
    let up = if spec.gravity.length_squared() > 1e-12 {
        -spec.gravity.normalize()
    } else {
        Vec3::Y
    };
    let dt = spec.dt as f64;

    for (i, b) in cur.bodies.iter().enumerate() {
        // 1. No NaN/Inf — exact.
        if !b.all_finite() {
            out.push(Violation::new(
                cur.tick,
                Some(i),
                "non_finite",
                1.0,
                0.0,
                format!("body '{}' has non-finite state", b.label),
            ));
            continue; // further numeric checks are meaningless on this body
        }

        // 2. Quaternion unit-norm.
        let qn = b.quat().length() as f64;
        if (qn - 1.0).abs() > spec.quat_norm_tol {
            out.push(Violation::new(
                cur.tick,
                Some(i),
                "quat_norm",
                (qn - 1.0).abs(),
                spec.quat_norm_tol,
                format!(
                    "body '{}' quaternion not unit length (‖q‖={qn:.8})",
                    b.label
                ),
            ));
        }

        if !b.is_dynamic {
            continue;
        }

        // 3. Explosion guard (loose sanity tier).
        let speed = b.lin_vel().length() as f64;
        if speed > spec.v_escape {
            out.push(Violation::new(
                cur.tick,
                Some(i),
                "speed_explosion",
                speed,
                spec.v_escape,
                format!("body '{}' linear speed exceeded sanity bound", b.label),
            ));
        }
        let omega = b.ang_vel().length() as f64;
        if omega > spec.omega_escape {
            out.push(Violation::new(
                cur.tick,
                Some(i),
                "spin_explosion",
                omega,
                spec.omega_escape,
                format!("body '{}' angular speed exceeded sanity bound", b.label),
            ));
        }

        // 4. No-teleport (differential).
        if let Some(p) = prev {
            if let Some(pb) = p.bodies.get(i) {
                let dx = (b.position() - pb.position()).length() as f64;
                let v_max = speed.max(pb.lin_vel().length() as f64);
                let bound = v_max * dt + spec.teleport_extra;
                if dx > bound {
                    out.push(Violation::new(
                        cur.tick,
                        Some(i),
                        "teleport",
                        dx,
                        bound,
                        format!(
                            "body '{}' moved farther in one tick than v·dt allows",
                            b.label
                        ),
                    ));
                }
            }
        }

        // 5. Floor non-penetration.
        if let Some(fy) = spec.floor_y {
            if let Some(shape) = spec.shapes.get(i) {
                if let Some(low) = lowest_along(shape, b.position(), b.quat(), up) {
                    let pen = (fy as f64) - (low as f64); // >0 means below the floor
                    if pen > spec.penetration_bound {
                        out.push(Violation::new(
                            cur.tick,
                            Some(i),
                            "penetration",
                            pen,
                            spec.penetration_bound,
                            format!(
                                "body '{}' penetrated the floor (slop+offset bound)",
                                b.label
                            ),
                        ));
                    }
                }
            }
        }
    }

    // 6. Static bodies must not move.
    for (k, &idx) in spec.static_indices.iter().enumerate() {
        if let (Some(b), Some(&p0)) = (cur.bodies.get(idx), spec.static_pos0.get(k)) {
            let d = (b.position() - p0).length() as f64;
            if d > spec.static_tol {
                out.push(Violation::new(
                    cur.tick,
                    Some(idx),
                    "static_moved",
                    d,
                    spec.static_tol,
                    format!(
                        "static body '{}' drifted from its initial position",
                        b.label
                    ),
                ));
            }
        }
    }

    // 7. Energy non-increase (passive systems).
    if spec.energy_non_increase {
        let e = cur.metrics.total_energy;
        let bound =
            spec.baseline_energy + spec.energy_rel * spec.baseline_energy.abs() + spec.energy_abs;
        if e > bound {
            out.push(Violation::new(
                cur.tick,
                None,
                "energy_increase",
                e,
                bound,
                format!(
                    "total energy rose above E0 (E0={:.6}, rel={:.2e}, abs={:.2e})",
                    spec.baseline_energy, spec.energy_rel, spec.energy_abs
                ),
            ));
        }
    }

    // 8. Momentum conservation (isolated systems).
    if spec.momentum_conserve {
        let p = DVec3::from_array(cur.metrics.linear_momentum);
        let d = (p - spec.baseline_momentum).length();
        if d > spec.momentum_tol {
            out.push(Violation::new(
                cur.tick,
                None,
                "momentum_drift",
                d,
                spec.momentum_tol,
                "linear momentum drifted from its conserved baseline".to_string(),
            ));
        }
    }
}

/// Derived relative tolerance for energy non-increase: ≈ 8·N_iter·eps_f32.
pub fn energy_rel_tol(solver_iterations: u32) -> f64 {
    8.0 * (solver_iterations.max(1) as f64) * F32_EPS
}

/// Derived relative tolerance for momentum conservation: ≈ 2·N_iter·eps_f32.
pub fn momentum_rel_tol(solver_iterations: u32) -> f64 {
    2.0 * (solver_iterations.max(1) as f64) * F32_EPS
}
