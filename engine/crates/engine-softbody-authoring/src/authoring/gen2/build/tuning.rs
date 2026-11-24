//! The build's environment tuning multipliers and the weakest-axis helper (split from `build.rs`).

use super::*;

pub fn env_f(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// The crash structure's yield multiple over the body's 1.15 % (rails, cans,
/// aprons, rear rails), `GEN2_CRUSH`: calibrated on wall56 against NHTSA's
/// frontal residual crush (`crash-reference-targets.md` §1).
pub fn crush_mult() -> f32 {
    // gen3 (2026-11-06): the census yield (the first lattice needed 4.5).
    env_f("GEN2_CRUSH", 1.0)
}

/// The crush cans' yield against the rails' (`GEN2_CAN`): their trigger.
pub fn can_mult() -> f32 {
    env_f("GEN2_CAN", 0.6)
}

/// The cabin's yield multiple (sills, pillars, roof rails), `GEN2_CABIN`.
pub fn cabin_mult() -> f32 {
    // gen3 (2026-11-06): 1.2 (the first lattice needed 6).
    env_f("GEN2_CABIN", 1.2)
}

/// The Gershgorin row cap as a multiple of sedan1's (`GEN2_ROW`, default 2.0:
/// the census's BeamNG parts run at p50 1.1–1.6, p90 1.4–2.6 in JBeam mode).
pub fn row_cap() -> f32 {
    env_f("GEN2_ROW", 3.0)
}

/// Stiffness multiple on every census band (`GEN2_STIFF`, the margin sweep).
pub fn stiff_mult() -> f32 {
    env_f("GEN2_STIFF", 1.0)
}

/// The smallest eigenvalue of a symmetric 3 × 3 and its axis (power
/// iteration on the shifted matrix).
pub fn weakest_axis(m: glam::Mat3) -> (f32, Vec3) {
    let tr = m.x_axis.x + m.y_axis.y + m.z_axis.z;
    let shifted = glam::Mat3::from_diagonal(Vec3::splat(tr + 1e-3)) - m;
    // The largest eigenvector of (tr·I − M) is M's smallest.
    let mut v = Vec3::new(0.577, 0.577, 0.577);
    for _ in 0..60 {
        v = (shifted * v).normalize_or(Vec3::X);
    }
    (v.dot(m * v), v)
}
