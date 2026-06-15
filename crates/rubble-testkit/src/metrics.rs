//! Per-tick body samples and derived system metrics (momentum, energy).
//!
//! All serialized structs use plain arrays / scalars (no `glam` types) so the
//! report serializes cleanly to JSON for the browser lane via `serde-wasm-bindgen`.
//! Aggregate quantities are accumulated in `f64` to keep the oracle's own math
//! from losing precision relative to the engine's `f32` state.

use glam::{DVec3, Mat3, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// f32 machine epsilon (2^-23). Used to *derive* tolerances from accumulated
/// round-off rather than fitting them to observed behavior.
pub const F32_EPS: f64 = 1.192_092_9e-7;

/// Static per-body information captured once at scenario setup (does not change
/// per tick), used to compute rotational energy / angular momentum against the
/// engine's *actual* inertia tensor.
#[derive(Debug, Clone)]
pub struct BodyMeta {
    pub label: String,
    pub mass: f32,
    pub is_dynamic: bool,
    /// Local-frame inertia tensor (NOT inverse). `Mat3::ZERO` for static bodies.
    pub inertia_local: Mat3,
}

/// One body's state at one tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodySample {
    pub label: String,
    pub pos: [f32; 3],
    /// Quaternion in (x, y, z, w) order.
    pub rot: [f32; 4],
    pub lin: [f32; 3],
    pub ang: [f32; 3],
    pub mass: f32,
    pub is_dynamic: bool,
}

impl BodySample {
    pub fn position(&self) -> Vec3 {
        Vec3::from_array(self.pos)
    }
    pub fn quat(&self) -> Quat {
        Quat::from_array(self.rot)
    }
    pub fn lin_vel(&self) -> Vec3 {
        Vec3::from_array(self.lin)
    }
    pub fn ang_vel(&self) -> Vec3 {
        Vec3::from_array(self.ang)
    }
    pub fn all_finite(&self) -> bool {
        self.position().is_finite()
            && self.lin_vel().is_finite()
            && self.ang_vel().is_finite()
            && self.rot.iter().all(|c| c.is_finite())
    }
}

/// Aggregate conserved quantities for the dynamic subsystem at one tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_mass: f64,
    pub linear_momentum: [f64; 3],
    pub angular_momentum: [f64; 3],
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub total_energy: f64,
    pub max_speed: f64,
    pub max_ang_speed: f64,
    /// Height of the lowest dynamic body along the up axis (-gravity direction).
    pub min_height: f64,
}

/// Full snapshot of the system at one tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickRecord {
    pub tick: usize,
    pub bodies: Vec<BodySample>,
    pub metrics: SystemMetrics,
}

/// Compute conserved quantities for the dynamic bodies. Potential energy uses a
/// fixed reference so it is comparable across ticks of the same scenario.
pub fn compute_metrics(samples: &[BodySample], metas: &[BodyMeta], gravity: Vec3) -> SystemMetrics {
    let up = if gravity.length_squared() > 1e-12 {
        -gravity.normalize()
    } else {
        Vec3::Y
    };

    let mut total_mass = 0.0f64;
    let mut com = Vec3::ZERO;
    for (s, m) in samples.iter().zip(metas) {
        if m.is_dynamic {
            total_mass += m.mass as f64;
            com += s.position() * m.mass;
        }
    }
    let com = if total_mass > 0.0 {
        com / total_mass as f32
    } else {
        Vec3::ZERO
    };

    let mut p = DVec3::ZERO;
    let mut l = DVec3::ZERO;
    let mut ke = 0.0f64;
    let mut pe = 0.0f64;
    let mut max_speed = 0.0f64;
    let mut max_w = 0.0f64;
    let mut min_h = f64::INFINITY;

    for (s, m) in samples.iter().zip(metas) {
        if !m.is_dynamic {
            continue;
        }
        let v = s.lin_vel();
        let w = s.ang_vel();
        let mass = m.mass as f64;
        let mom = v.as_dvec3() * mass;
        p += mom;
        ke += 0.5 * mass * v.length_squared() as f64;

        let r = Mat3::from_quat(s.quat());
        let i_world = r * m.inertia_local * r.transpose();
        let spin = (i_world * w).as_dvec3();
        l += (s.position() - com).as_dvec3().cross(mom) + spin;
        ke += 0.5 * w.as_dvec3().dot(spin);

        pe += mass * (-gravity).as_dvec3().dot(s.position().as_dvec3());
        max_speed = max_speed.max(v.length() as f64);
        max_w = max_w.max(w.length() as f64);
        min_h = min_h.min(s.position().dot(up) as f64);
    }
    if !min_h.is_finite() {
        min_h = 0.0;
    }

    SystemMetrics {
        total_mass,
        linear_momentum: p.to_array(),
        angular_momentum: l.to_array(),
        kinetic_energy: ke,
        potential_energy: pe,
        total_energy: ke + pe,
        max_speed,
        max_ang_speed: max_w,
        min_height: min_h,
    }
}
